import collections
from pandas_datareader import data as web

Quote = collections.namedtuple('Quote', ['name', 'value'])


def get_quote(symbol):
    quote = web.get_quotes_robinhood(str(symbol))
    return Quote(
        name=symbol,
        value=float(quote.loc['last_trade_price'].iloc[0]))


if __name__ == '__main__':
    print(get_quote('AAPL'))
