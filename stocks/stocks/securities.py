import numpy as np
import pandas as pd

from pandas_datareader import data as web


def quote(symbol, provider='yahoo'):
    if provider == 'robinhood':
        quote_df = web.get_quotes_robinhood(str(symbol))
        return float(quote_df.loc['last_trade_price'].iloc[0])
    elif provider == 'yahoo':
        quote_df = web.get_quote_yahoo(str(symbol))
        return float(quote_df.price.iloc[0])


def quotes(symbols, provider='yahoo'):
    if provider == 'robinhood':
        quote_df = web.get_quotes_robinhood(map(str, symbols))
        return quote_df.loc['last_trade_price'].astype(float)
    elif provider == 'yahoo':
        quote_df = web.get_quote_yahoo(map(str, symbols))
        return quote_df.price.astype(float)


def asset_quotes(assets_df, asset_prices=None):
    NYSE_mask = (assets_df['exchange'] == 'NYSE').values
    prices = np.ones_like(NYSE_mask, dtype=float)
    NYSE_symbols = assets_df[NYSE_mask].symbol.tolist()
    prices[NYSE_mask] = quotes(NYSE_symbols)
    if asset_prices is None:
        asset_prices = dict()
    for key, value in asset_prices.items():
        asset_mask = (assets_df['symbol'] == key).values
        prices[asset_mask] = value
    return prices


def robinhood_history_filter(rh_table):
    table = rh_table[rh_table.Type == 'TRADES'].copy()
    table['price'] = pd.to_numeric(
        table.Price.apply(lambda x: x.replace('$ ', '')))
    table['timestamp'] = table.Timestamp.apply(lambda x: pd.Timestamp(
        x, tz='US/Eastern'))
    table = table.filter(items=['timestamp', 'Symbol', 'Qty', 'price']
                         ).sort_values('timestamp').reset_index(drop=True)
    return table
