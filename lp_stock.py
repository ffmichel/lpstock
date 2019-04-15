#! /usr/bin/env python
import argparse
import operator

import pandas as pd

from stocks import securities, allocation_graph, graph_common

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


class StoreDict(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kv = {}
        if not isinstance(values, (list, )):
            values = (values, )
        for value in values:
            n, v = value.split('=')
            kv[n] = v
        setattr(namespace, self.dest, kv)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets_csv', type=str, required=True)
    parser.add_argument('--allocation_graph_csv', type=str, required=True)
    parser.add_argument('-a',
                        '--amount_to_allocate',
                        type=float,
                        required=True)
    parser.add_argument('-t',
                        '--tickers_to_allocate',
                        nargs='*',
                        type=str,
                        required=True)
    parser.add_argument('--ticker_prices', nargs='*', action=StoreDict)
    cli_args = parser.parse_args()

    allocation_graph_df = pd.read_csv(cli_args.allocation_graph_csv)
    assets_df = pd.read_csv(cli_args.assets_csv)

    assets_df['price'] = securities.asset_quotes(
        assets_df, asset_prices=cli_args.ticker_prices)
    optimal_allocations = allocation_graph.distribute_allocations(
        allocation_graph_df=allocation_graph_df,
        assets_df=assets_df,
        amount_to_allocate=cli_args.amount_to_allocate,
        tickers_to_allocate=cli_args.tickers_to_allocate)

    delta = assets_df.set_index('symbol').number.copy()
    delta[:] = 0
    delta[optimal_allocations.keys()] = list(optimal_allocations.values())
    assets_df['delta_number'] = delta.reset_index().number

    graph = allocation_graph.from_df(allocation_graph_df)
    allocation_graph.add_equity(graph, assets_df)
    allocation_graph.add_equity_delta(graph, assets_df)
    allocation_graph.add_current_percentage(graph)
    allocation_graph.add_equity(graph, assets_df)
    allocation_graph.add_target_percentage(graph)
    allocation_graph.add_new_percentage(graph)
    allocation_graph.add_delta_percentage(graph)

    fmt = '| {:^15} | {:^40} | {:^20} | {:^20} | {:^20} |'
    print(
        fmt.format('Asset', 'Equity', 'Target Percentage', 'Actual Percentage',
                   'Shares'))
    print(fmt.format('=' * 15, '=' * 40, '=' * 20, '=' * 20, '=' * 20))
    for distances, nodes in sorted(
            graph_common.node_by_distance_to_root(graph).items(),
            key=operator.itemgetter(0)):
        for idx, node in enumerate(nodes):
            graph_node = graph.nodes[node]
            target_percentage = graph_node['target_percentage']
            current_percentage = graph_node['current_percentage']
            delta_percentage = graph_node['delta_percentage']
            equity = graph_node['equity']
            equity_delta = graph_node['equity_delta']

            equity_str = '{:,.2f}'.format(equity)
            if equity_delta > 0:
                equity_str = partly_colored_str_in_cell(
                    equity_str, ' + {:,.2f}'.format(equity_delta), 40, red_str)

            target_percentage_str = '{:.2f}'.format(target_percentage)
            current_percentage_str = '{:.2f}'.format(current_percentage)
            sign = '+' if delta_percentage >= 0 else '-'
            current_percentage_str = partly_colored_str_in_cell(
                current_percentage_str,
                ' {} {:.2f}'.format(sign, abs(delta_percentage)), 20, red_str)

            shares = ''
            share_increase = 0
            if node in assets_df.symbol.values:
                asset_row = assets_df[assets_df.symbol == node].iloc[0]
                shares = str(asset_row.number)
                share_increase = asset_row.delta_number

            if share_increase > 0:
                shares = partly_colored_str_in_cell(
                    shares, ' + {}'.format(share_increase), 20, red_str)

            line_str = fmt.format('{:<15}'.format(node),
                                  '{:>40}'.format(equity_str),
                                  '{:>20}'.format(target_percentage_str),
                                  '{:>20}'.format(current_percentage_str),
                                  '{:>20}'.format(shares))
            if idx % 2 == 1:
                print(reverse_str(line_str))
            else:
                print(line_str)

        print(fmt.format('-' * 15, '-' * 40, '-' * 20, '-' * 20, '-' * 20))

    root = graph_common.root(graph)
    amount_left = (cli_args.amount_to_allocate -
                   graph.nodes[root]['equity_delta'])
    print('\n After purchase, ${:,.2f} will be left.'.format(amount_left))


if __name__ == '__main__':
    run()
