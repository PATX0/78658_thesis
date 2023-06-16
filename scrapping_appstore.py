from app_store_scraper import AppStore
from google_play_scraper import app
import pandas as pd
import numpy as np
import csv
import datetime
import deepl

#scrapping comments
country_codes = {
    'Ukraine': 'UA',
    'Nigeria': 'NG',
    'USA': 'US',
    'China': 'CN',
    'Brazil': 'BR'
}
countries = ['UA','NG','US','CN','BR']
init_date = date = datetime.datetime(2017, 5, 1)

for c in countries:
    kucoin = AppStore(country=c, app_name='kucoin', app_id = '1378956601')
    kucoin.review(how_many = 1000, after=init_date)
    kucoindf = pd.DataFrame(np.array(kucoin.reviews),columns=['review'])
    kucoindf2 = kucoindf.join(pd.DataFrame(kucoindf.pop('review').tolist()))
    print(f'kucoin_reviews_{c}.csv')
    kucoindf2.to_csv(f'kucoin_reviews_{c}.csv', index=False)

    coinbase = AppStore(country=c, app_name='coinbase', app_id = '886427730')
    coinbase.review(how_many=1000)
    coinbasedf = pd.DataFrame(np.array(coinbase.reviews),columns=['review'])
    coinbasedf2 = coinbasedf.join(pd.DataFrame(coinbasedf.pop('review').tolist()))
    coinbasedf2.to_csv(f'coinbase_reviews_{c}.csv', index=False)

    binance = AppStore(country=c, app_name='binance', app_id = '886427730')
    binance.review(how_many=1000)
    binancedf = pd.DataFrame(np.array(binance.reviews),columns=['review'])
    binancedf2 = binancedf.join(pd.DataFrame(binancedf.pop('review').tolist()))
    binancedf2.to_csv(f'binance_reviews_{c}.csv', index=False)

    # revolut = AppStore(country=c, app_name='revolut', app_id = '886427730')
    # revolut.review(how_many=1000)
    # revolutdf = pd.DataFrame(np.array(revolut.reviews),columns=['review'])
    # revolutdf2 = revolutdf.join(pd.DataFrame(revolutdf.pop('review').tolist()))
    # revolutdf2.to_csv(f'revolut_reviews_{c}.csv', index=False)