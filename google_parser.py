import requests
import json
import collections


Quote = collections.namedtuple('Quote', ['name', 'value'])


def get_quote(symbol):
    url = 'https://finance.google.com/finance?q={}&output=json'
    rsp = requests.get(url.format(symbol))
    if rsp.status_code in (200,):
        fin_data = json.loads(rsp.content[6:-2].decode('unicode_escape'))
        return Quote(name=fin_data['name'], value=float(fin_data['l']))
    else:
        raise ValueError()


if __name__ == '__main__':
    print(get_quote('AAPL'))
