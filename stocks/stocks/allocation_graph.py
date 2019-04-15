import functools

import numpy as np
import networkx as nx

from . import graph_common
from . import optimize


def from_df(allocation_graph_df):
    return nx.from_pandas_edgelist(allocation_graph_df,
                                   edge_attr='relative_allocation',
                                   create_using=nx.DiGraph())


def compute_target_percentage(node_target_relative_percentage,
                              parent_target_percentage):
    if node_target_relative_percentage is None or parent_target_percentage is None:
        return None
    return node_target_relative_percentage * parent_target_percentage


def check_consistent_portfolio(portfolio, graph):
    assert set(portfolio) == set(graph_common.leaf_nodes(graph))


def _node_equity(graph, node, assets_df):
    if node in assets_df.symbol.values:
        asset_row = assets_df[assets_df.symbol == node].iloc[0]
        return asset_row.price * asset_row.number
    return sum(graph.nodes[child]['equity']
               for child in graph.successors(node))


def add_equity(graph, assets_df):
    node_equity = functools.partial(_node_equity, assets_df=assets_df)
    graph_common.reverse_tree_traversal(graph, 'equity', node_equity)


def _node_equity_delta(graph, node, assets_df):
    if node in assets_df.symbol.values:
        asset_row = assets_df[assets_df.symbol == node].iloc[0]
        return asset_row.price * asset_row.delta_number
    return sum(graph.nodes[child]['equity_delta']
               for child in graph.successors(node))


def add_equity_delta(graph, assets_df):
    node_equity_delta = functools.partial(_node_equity_delta,
                                          assets_df=assets_df)
    graph_common.reverse_tree_traversal(graph, 'equity_delta',
                                        node_equity_delta)


def _node_target_percentage(graph, edge):
    # Add percentages from allocations.
    relative_allocation = float(
        graph.edges[edge]['relative_allocation']) / 100.0
    parent_name, child_name = edge
    parent_target_percentage = graph.nodes[parent_name]['target_percentage']
    return relative_allocation * parent_target_percentage


def add_target_percentage(graph):
    graph_common.tree_bfs(graph, 100.0, 'target_percentage',
                          _node_target_percentage)


def add_current_percentage(graph):
    root = graph_common.root(graph)
    for node in graph.nodes:
        graph.nodes[node]['current_percentage'] = (float(
            graph.nodes[node]['equity']) / graph.nodes[root]['equity']) * 100.0


def add_new_percentage(graph):
    root = graph_common.root(graph)
    for node in graph.nodes:
        graph.nodes[node]['new_percentage'] = (
            float(graph.nodes[node]['equity'] +
                  graph.nodes[node]['equity_delta']) /
            (graph.nodes[root]['equity'] +
             graph.nodes[root]['equity_delta'])) * 100.0


def add_delta_percentage(graph):
    for node in graph.nodes:
        graph.nodes[node]['delta_percentage'] = graph.nodes[node][
            'new_percentage'] - graph.nodes[node]['current_percentage']


def _node_max_allocation(graph, edge):
    parent_name, child_name = edge
    parent_max_allocation = graph.nodes[parent_name]['max_allocation']
    parent_equity = graph.nodes[parent_name]['equity']
    child_equity = graph.nodes[child_name]['equity']
    relative_allocation = float(
        graph.edges[edge]['relative_allocation']) / 100.0
    if np.isnan(relative_allocation):
        relative_allocation = 1.0
    return max(
        min(
            relative_allocation * (parent_max_allocation + parent_equity) -
            child_equity, parent_max_allocation), 0)


def add_max_allocation(graph, amount_to_allocate):
    graph_common.tree_bfs(graph, amount_to_allocate, 'max_allocation',
                          _node_max_allocation)


def generate_constraints(graph, tickers_to_allocate):
    # Compute the maximum that we could allocate to each ticker given percentages.
    nodes_max_pair_list = set([((ticker, ),
                                graph.nodes[ticker]['max_allocation'])
                               for ticker in tickers_to_allocate])
    root = graph_common.root(graph)
    for edge in nx.bfs_edges(graph, root):
        node = edge[0]
        node_list = tuple([
            ticker for ticker in tickers_to_allocate
            if ticker in nx.descendants(graph, node)
        ])
        max_amount = graph.nodes[node]['max_allocation']
        if len(node_list) >= 2:
            nodes_max_pair_list.add((node_list, max_amount))

    return list(nodes_max_pair_list)


def optimize_allocations(tickers_to_allocate, assets_df, constraints):
    # vector of stock prices
    stock_prices = assets_df.set_index(
        'symbol').loc[tickers_to_allocate].price.values
    c = -1 * stock_prices

    # Matrix of latent space:
    A = np.zeros((len(constraints), len(tickers_to_allocate)))
    for i, constraint in enumerate(constraints):
        A[i, :] = [
            price if tick in constraint[0] else 0
            for price, tick in zip(stock_prices, tickers_to_allocate)
        ]

    # vector of constraints
    b = np.array([constraint[1] for constraint in constraints])

    bounds = [[0, None] for _ in tickers_to_allocate]
    res = optimize.branch_and_bound(c=c, A_ub=A, b_ub=b, bounds=bounds)
    return dict(zip(tickers_to_allocate, res.x))


def distribute_allocations(allocation_graph_df, assets_df, amount_to_allocate,
                           tickers_to_allocate):
    graph = from_df(allocation_graph_df)
    add_equity(graph=graph, assets_df=assets_df)
    add_max_allocation(graph=graph, amount_to_allocate=amount_to_allocate)
    constraints = generate_constraints(graph, tickers_to_allocate)
    return optimize_allocations(tickers_to_allocate, assets_df, constraints)
