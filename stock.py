import google_parser as gp
import copy
import json
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

    def stock_price(self, symbol):
        return self._shares.get_price(symbol)

    def num_stocks(self, symbol):
        return self._portfolio_dict[symbol]

    def full_name(self, symbol):
        return self._shares.get_name(symbol)


class Category(object):
    def __init__(self, name, portfolio, percentage, children=[], symbols={}):
        self.name = name
        self.portfolio = portfolio
        self.percentage = percentage
        self.symbols = symbols
        self.children = children

    def __iter__(self):
        yield self
        for child in self.children:
            for grand_child in child:
                yield grand_child

    def child(self, name):
        return next((child for child in self if child.name == name), None)

    @staticmethod
    def has_symbols(allocation, portfolio):
        return any(key in portfolio.all_symbols for key in allocation)

    @staticmethod
    def has_children(allocation, portfolio):
        return any(key not in portfolio.all_symbols for key in allocation)

    @classmethod
    def from_dict(cls, name, portfolio, percentage, all_allocations):
        symbols = {}
        allocation = all_allocations[name]
        if Category.has_symbols(allocation, portfolio):
            symbols = {
                symbol: percentage
                for symbol, percentage in allocation.items()
                if symbol in portfolio.all_symbols
            }

        children = []
        if Category.has_children(allocation, portfolio):
            children = [
                Category.from_dict(
                    name=cat_name,
                    percentage=cat_percentage,
                    portfolio=portfolio,
                    all_allocations=all_allocations)
                for cat_name, cat_percentage in allocation.items()
            ]
        cls = Category(
            name=name,
            portfolio=portfolio,
            percentage=percentage,
            children=children,
            symbols=symbols)
        return cls

    @property
    def total_value(self):
        return float(
            sum(self.portfolio.equity(symbol) for symbol in self.symbols) +
            sum(child.total_value for child in self.children))

    @property
    def all_symbols(self):
        ret = copy.deepcopy(self.symbols)
        # TODO make this nicer
        for child in self.children:
            ret.update(child.all_symbols)
        return ret

    def percentage(self, symbol):
        return self.portfolio.equity(symbol) * 100. / self.total_value

    def delta_percentage(self, symbol, delta_stocks, new_total_value):
        stock_equity = self.portfolio.equity(symbol)
        new_stock_equity = stock_equity + self.portfolio.stock_price(
            symbol) * delta_stocks
        return new_stock_equity * 100. / float(new_total_value)

    def percentage_difference(self, symbol, delta_stocks, new_total_value):
        return self.target(symbol) - self.delta_percentage(
            symbol, delta_stocks, new_total_value)

    def target(self, symbol):
        if symbol in self.symbols:
            return float(self.symbols[symbol])
        for child in self.children:
            if symbol in child.all_symbols:
                return (float(child.percentage) * child.target(symbol) / 100.)

    def proportion_child(self, child):
        return percent(child.total_value * 100. / self.total_value)

    def num_new_stocks_from_target(self, symbol, influx):
        new_amount = influx + self.total_value

        stock_target = self.target(symbol)
        amount_available_for_symbol = float(new_amount) * stock_target / 100.

        stock_value = portfolio.stock_price(symbol)
        num_stocks = amount_available_for_symbol // stock_value
        return num_stocks - self.portfolio.num_stocks(symbol)


def make_recommendation(influx, wallet):
    return {
        symbol: wallet.num_new_stocks_from_target(symbol, influx)
        for symbol in wallet.all_symbols
    }


def refine_recommendation(influx, recommendation, wallet):
    total_bought = sum(num_to_buy * wallet.portfolio.stock_price(symbol)
                       for symbol, num_to_buy in recommendation.items())
    differences = [(symbol, wallet.percentage_difference(
        symbol, num_to_buy, wallet.total_value + total_bought))
                   for symbol, num_to_buy in recommendation.items()]

    for symbol, _ in sorted(differences, key=operator.itemgetter(1)):
        new_total = total_bought + wallet.portfolio.stock_price(symbol)
        if new_total <= new_influx:
            recommendation[symbol] += 1
            total_bought = new_total
    return recommendation


def display_recommendation(recommendation, wallet, final_value):
    msg = ("{:<5} | {:<40}  | num to buy: {:<4} | cost {:<8} | "
           "final_percentage {:<6} | target {:<6}")
    for symbol, num_to_buy in recommendation.items():
        name = wallet.portfolio.full_name(symbol)
        price = wallet.portfolio.stock_price(symbol)
        cost = price * num_to_buy
        final_percentage = wallet.delta_percentage(
            symbol=symbol,
            delta_stocks=num_to_buy,
            new_total_value=final_value)
        print(msg.format(symbol, name, num_to_buy, cost, percent(final_percentage),
                         wallet.target(symbol)))


if __name__ == "__main__":
    with open('asset.json', 'r') as infile:
        asset_data = json.load(infile)

    portfolio_dict = asset_data['portfolio']
    allocation = asset_data['allocation']

    portfolio = Portfolio(portfolio_dict)
    wallet = Category.from_dict('Wallet', portfolio, 100, allocation)
    new_influx = float(sys.argv[1])

    new_value = new_influx + wallet.total_value

    recommendation = make_recommendation(new_influx, wallet)
    recommendation = refine_recommendation(new_influx, recommendation, wallet)

    final_total_value = wallet.total_value + sum(
        num_to_buy * wallet.portfolio.stock_price(symbol)
        for symbol, num_to_buy in recommendation.items())

    display_recommendation(recommendation, wallet, final_total_value)
    print('Wallet current value: {}'.format(wallet.total_value))
    print('Wallet value with {} added: {}, remains {} unused'.format(
        new_influx, final_total_value, (wallet.total_value + new_influx - final_total_value)))
