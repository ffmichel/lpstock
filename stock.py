import google_parser as gp
import copy
import json
import math
import collections
import operator
import sys


class Shares(object):
    def __init__(self, symbols):
        self._quotes = {symbol: gp.get_quote(symbol) for symbol in symbols}

    def get_price(self, symbol):
        return self._quotes[symbol].value

    def get_name(self, symbol):
        return self._quotes[symbol].name


class percent(object):
    def __init__(self, value):
        self._value = value

    def __repr__(self):
        return format(self._value, '.2f')


class Portfolio(object):
    def __init__(self, portfolio_dict):
        self._portfolio_dict = portfolio_dict
        self._shares = Shares(self._portfolio_dict.keys())

    @property
    def total_value(self):
        return sum(self.equity(symbol) for symbol in self._portfolio_dict)

    def equity(self, symbol):
        return self._shares.get_price(symbol) * self._portfolio_dict[symbol]

    def percentage(self, symbol):
        return percent((self.equity(symbol) * 100.) / float(self.total_value))

    @property
    def all_symbols(self):
        return self._portfolio_dict.keys()


class Category(object):
    def __init__(self, name, portfolio, symbols={}, children={}, target_percentage=None):
        self.name = name
        self._portfolio = portfolio
        self._symbols = symbols
        self._children = children
        self._target_precentage = target_percentage

    @property
    def total_value(self):
        return float(sum(self._portfolio.equity(symbol)
                         for symbol in self._symbols) +
                     sum(child.total_value
                         for child in self._children))

    @property
    def all_symbols(self):
        ret = copy.deepcopy(self._symbols)
        # TODO make this nicer
        for child in self._children:
            ret.update(child.all_symbols)
        return ret

    @property
    def all_children(self):
        return [child.name for child in self._children]

    def percentage(self, symbol):
        return percent(self._portfolio.equity(symbol) * 100. / self.total_value)

    def target(self, symbol):
        if symbol in self._symbols:
            return float(self._symbols[symbol])
        for child in self._children:
            if symbol in child.all_symbols:
                return (float(self._children[child]) * child.target(symbol) / 100.)

    def proportion_children(self, child):
        return percent(child.total_value * 100. / self.total_value)


with open('asset.json', 'r') as infile:
    asset_data = json.load(infile)

portfolio_dict = asset_data['portfolio']
allocation = asset_data['allocation']

portfolio = Portfolio(portfolio_dict)
wallet_dict = dict()
key_list = allocation.keys()


def is_leaf(allocation, key):
    section = allocation[key]
    return all(s_key not in allocation for s_key in section)


while(key_list):
    #  import ipdb; ipdb.set_trace() # BREAKPOINT
    key = key_list.pop(0)
    if is_leaf(allocation, key):
        wallet_dict[key] = Category(name=key, portfolio=portfolio,
                                    symbols=allocation[key])
    else:
        section_keys = allocation[key].keys()
        if all(k in wallet_dict for k in section_keys):
            wallet_dict[key] = Category(name=key, portfolio=portfolio,
                                        children={wallet_dict[k]: allocation[key][k]
                                                  for k in section_keys})
        else:
            key_list.append(key)





wallet = wallet_dict['Wallet']


new_influx = float(sys.argv[1])

new_value = new_influx + wallet.total_value

PDiff = collections.namedtuple('PDiff',
                               ['symbol', 'percentage_diff', 'num_to_buy'])
new_portfolio_diffs = list()
for symbol in wallet.all_symbols:
    target = wallet.target(symbol)
    value = portfolio._shares.get_price(symbol)
    num_stocks = new_value * target / (value * 100.)
    floor_num_stocks = int(math.floor(num_stocks))
    equity = floor_num_stocks * value
    percentage = (equity / new_value) * 100.
    pdiff = percent(abs(percentage - target))
    name = portfolio._shares.get_name(symbol)
    num_to_buy = floor_num_stocks - portfolio._portfolio_dict[symbol]

    new_portfolio_diffs.append(PDiff(symbol=symbol, percentage_diff=pdiff._value, num_to_buy=num_to_buy))

total_bought = sum(item.num_to_buy * portfolio._shares.get_price(item.symbol) for item in new_portfolio_diffs)
reco = dict()
for item in sorted(new_portfolio_diffs, key=operator.attrgetter('percentage_diff'), reverse=True):
    new_total = total_bought + portfolio._shares.get_price(item.symbol)
    if new_total <= new_influx:
        reco[item.symbol] = item.num_to_buy + 1
        total_bought = new_total
    else:
        reco[item.symbol] = item.num_to_buy

final_total_value = total_bought + wallet.total_value
unused = new_influx - total_bought
for symbol, num in reco.iteritems():
    name = portfolio._shares.get_name(symbol)
    price = portfolio._shares.get_price(symbol)
    cost = num * price
    percentage = percent((cost + portfolio.equity(symbol)) * 100 / new_value)
    target = wallet.target(symbol)
    msg = "{:<5} | {:<40}  | num to buy: {:<3} | cost {:<8} | final_percentage {:<6} | target {:<6}"
    print msg.format(symbol, name, num, cost, percentage, percent(target))
print('Wallet current value: {}'.format(wallet.total_value))
print('Wallet value with {} added: {}, remains {} unused'. format(new_influx, final_total_value, unused))
