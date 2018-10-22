import os
import argparse
import copy
import collections
import operator

import networkx as nx
import numpy as np
import yaml
from scipy import optimize

import google_parser as gp


def load_graph(tree_dict, allocations):
    G = nx.DiGraph()
    for node_name, children in tree_dict.items():
        if node_name in allocations:
            node_allocation = float(allocations[node_name]) / 100
        else:
            node_allocation = None

        G.add_node(node_name, allocation=node_allocation)
        for child in children:
            if child in allocations:
                child_allocation = float(allocations[child]) / 100
            else:
                child_allocation = None
            G.add_node(child, allocation=child_allocation)
            G.add_edge(node_name, child)
    return G


def graph_root(graph):
    root_nodes = set(x for x in graph.nodes()
                     if graph.out_degree(x) > 0 and graph.in_degree(x) == 0)

    assert len(root_nodes) == 1
    root = root_nodes.pop()
    return root


def check_consistent_portfolio(portfolio, graph):
    portfolio_tickers = set(portfolio)
    leaf_nodes = set(x for x in graph.nodes()
                     if graph.out_degree(x) == 0 and graph.in_degree(x) == 1)
    assert portfolio_tickers == leaf_nodes


def add_percentages(graph, root):
    # Add percentages from allocations.
    graph.nodes[root]['percentage'] = 1.0
    for edge in nx.bfs_edges(graph, root):
        parent_percent = graph.nodes[edge[0]]['percentage']
        child_allocation = graph.nodes[edge[1]]['allocation']
        if parent_percent is None or child_allocation is None:
            percent = None
        else:
            percent = parent_percent * child_allocation
        graph.nodes[edge[1]]['percentage'] = percent


def node_by_distance_to_root(graph, root):
    length_by_node = nx.shortest_path_length(graph, source=root)
    node_by_length = collections.defaultdict(list)
    for node, length in length_by_node.items():
        node_by_length[length].append(node)
    return node_by_length


def price_per_ticker(portfolio, hardcoded_ticker_values):
    value_by_ticker = copy.deepcopy(hardcoded_ticker_values)
    symbols = set(portfolio) - set(value_by_ticker)

    value_by_ticker.update(
        {symbol: gp.get_quote(symbol).value
         for symbol in symbols})
    return value_by_ticker


def add_equity(graph, root, portfolio, price_by_ticker, graph_field_name):
    node_by_length = node_by_distance_to_root(graph, root)
    for length, nodes in sorted(
            node_by_length.items(), key=operator.itemgetter(0), reverse=True):
        for node in nodes:
            if node in portfolio:
                equity = portfolio[node] * price_by_ticker[node]
                graph.nodes[node][graph_field_name] = equity
            else:
                equity = sum(graph.nodes[child][graph_field_name]
                             for child in graph.successors(node))
                graph.nodes[node][graph_field_name] = equity


def node_max_allocation(node_allocation, parent_max_allocation, parent_equity,
                        node_equity):
    return max(
        min(
            node_allocation * (parent_max_allocation + parent_equity) -
            node_equity, parent_max_allocation), 0)


def add_max_allocation(graph, root, amount_to_allocate):
    graph.nodes[root]['max_allocation'] = amount_to_allocate
    for edge in nx.bfs_edges(graph, root):
        parent_max_allocation = graph.nodes[edge[0]]['max_allocation']
        parent_equity = graph.nodes[edge[0]]['equity']
        child_allocation = graph.nodes[edge[1]]['allocation']
        if child_allocation is None:
            child_allocation = 1
        child_equity = graph.nodes[edge[1]]['equity']
        graph.nodes[edge[1]]['max_allocation'] = node_max_allocation(
            node_allocation=child_allocation,
            parent_max_allocation=parent_max_allocation,
            parent_equity=parent_equity,
            node_equity=child_equity)


def generate_constraints(graph, root, tickers_to_allocate, amount_to_allocate):
    # Compute the maximum that we could allocate to each ticker given percentages.

    add_max_allocation(
        graph=graph, root=root, amount_to_allocate=amount_to_allocate)

    nodes_max_pair_list = set([((ticker, ),
                                graph.nodes[ticker]['max_allocation'])
                               for ticker in tickers_to_allocate])
    for edge in nx.bfs_edges(graph, root):
        node = edge[0]
        node_list = tuple([
            ticker for ticker in tickers_to_allocate
            if ticker in nx.descendants(graph, node)
        ])
        max_amount = graph.nodes[node]['max_allocation']
        if len(node_list) > 2:
            nodes_max_pair_list.add((node_list, max_amount))

    return list(nodes_max_pair_list)


def optimize_allocations(tickers_to_allocate, price_by_ticker, constraints):
    # linear program
    # vector of stock prices
    stock_prices = np.array(
        [price_by_ticker[ticker] for ticker in tickers_to_allocate])
    c = -1 * stock_prices

    # Matrix of latent space:
    A = np.zeros((len(constraints), len(tickers_to_allocate)))
    for i, constraint in enumerate(constraints):
        A[i, :] = [
            price_by_ticker[tick] if tick in constraint[0] else 0
            for tick in tickers_to_allocate
        ]

    # vector of constraints
    b = np.array([constraint[1] for constraint in constraints])

    bounds = [[0, None] for _ in tickers_to_allocate]
    res = optimize.linprog(c, A_ub=A, b_ub=b, bounds=bounds)
    # branch and bound:
    for idx in range(len(tickers_to_allocate)):
        new_bounds_left = copy.deepcopy(bounds)
        new_bounds_left[idx][0] = int(res.x[idx]) + 1
        res_left = optimize.linprog(c, A_ub=A, b_ub=b, bounds=new_bounds_left)
        new_bounds_right = copy.deepcopy(bounds)
        new_bounds_right[idx][1] = int(res.x[idx])
        res_right = optimize.linprog(
            c, A_ub=A, b_ub=b, bounds=new_bounds_right)
        if res_left.fun < res_right.fun:
            res = res_left
            bounds = copy.deepcopy(new_bounds_left)
            bounds[idx][1] = bounds[idx][0]
        else:
            res = res_right
            bounds = copy.deepcopy(new_bounds_right)
            bounds[idx][0] = bounds[idx][1]
    return dict(zip(tickers_to_allocate, res.x))


ENDC = '\033[0m'


def red_str(str_):
    return '\033[91m' + str_ + ENDC


def reverse_str(str_):
    def reverse_stringlet(stringlet):
        return '\033[;7m' + stringlet + '\033[0m'

    return ''.join(map(reverse_stringlet, str_.split(ENDC)))


def partly_colored_str_in_cell(str1, str2, cell_size, color_func):
    extra_width = len(repr(color_func('')))
    red_str2 = color_func(str2)
    tentative_str = str1 + red_str2
    num_char = cell_size + extra_width - len(repr(tentative_str))
    return ' ' * num_char + tentative_str


ASSET_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'asset.yml')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--asset_file', type=str, required=False, default=ASSET_FILE)
    parser.add_argument('amount_to_allocate', type=float)
    cli_args = parser.parse_args()

    with open(cli_args.asset_file, 'r') as fh:
        asset_file = yaml.load(fh)

    tree = asset_file['tree']
    allocations = asset_file['allocations']
    portfolio = asset_file['portfolio']
    hardcoded_ticker_values = asset_file['hardcoded_ticker_values']
    tickers_to_allocate = asset_file['tickers_to_allocate']
    amount_to_allocate = cli_args.amount_to_allocate

    G = load_graph(tree, allocations)
    root = graph_root(G)
    check_consistent_portfolio(portfolio, G)
    add_percentages(G, root)

    value_by_ticker = price_per_ticker(portfolio, hardcoded_ticker_values)
    add_equity(G, root, portfolio, value_by_ticker, 'equity')

    constraints = generate_constraints(G, root, tickers_to_allocate,
                                       amount_to_allocate)

    optimal_allocations = optimize_allocations(tickers_to_allocate,
                                               value_by_ticker, constraints)

    node_by_length = node_by_distance_to_root(G, root)

    portfolio_change = {
        sym: int(optimal_allocations.get(sym, 0))
        for sym in portfolio
    }
    add_equity(G, root, portfolio_change, value_by_ticker, 'equity_change')

    fmt = '| {:^15} | {:^40} | {:^20} | {:^20} | {:^20} |'
    print(fmt.format('Asset', 'Equity', 'Target Percentage',
                     'Actual Percentage', 'Shares'))
    print(fmt.format('=' * 15, '=' * 40, '=' * 20, '=' * 20, '=' * 20))
    root_equity = G.nodes[root]['equity']
    root_equity_change = G.nodes[root]['equity_change']
    for length in range(max(node_by_length) + 1):
        nodes = node_by_length[length]
        for idx, node in enumerate(nodes):
            percentage = G.nodes[node]['percentage']
            if percentage is not None:
                percentage = round(100 * percentage, 2)
            node_equity = G.nodes[node]['equity']
            equity_change = G.nodes[node]['equity_change']
            node_equity_str = '{:,.2f}'.format(node_equity)
            if equity_change > 0:
                node_equity_str = partly_colored_str_in_cell(
                    node_equity_str, ' + {:,.2f}'.format(equity_change), 40,
                    red_str)

            actual_percentage = 100 * float(node_equity) / root_equity
            actual_percentage_str = '{:.2f}'.format(actual_percentage)
            percentage_change = 100 * float(node_equity + equity_change) / (
                root_equity + root_equity_change) - actual_percentage
            sign = '+' if percentage_change >= 0 else '-'
            actual_percentage_str = partly_colored_str_in_cell(
                actual_percentage_str, ' {} {:.2f}'.format(
                    sign, abs(percentage_change)), 20, red_str)

            shares = str(portfolio.get(node, ''))
            share_increase = portfolio_change.get(node, 0)
            if share_increase >= 1:
                shares = partly_colored_str_in_cell(
                    shares, ' + {}'.format(share_increase), 20, red_str)

            line_str = fmt.format('{:<15}'.format(node),
                                  '{:>40}'.format(node_equity_str),
                                  '{:>20}'.format(percentage),
                                  '{:>20}'.format(actual_percentage_str),
                                  '{:>20}'.format(shares))
            if idx % 2 == 1:
                print(reverse_str(line_str))
            else:
                print(line_str)

        print(fmt.format('-' * 15, '-' * 40, '-' * 20, '-' * 20, '-' * 20))

    print('\n After purchase, ${:,.2f} will be left.'.format(
        amount_to_allocate - sum(value_by_ticker[sym] * portfolio_change[sym]
                                 for sym in portfolio_change)))


if __name__ == '__main__':
    run()
