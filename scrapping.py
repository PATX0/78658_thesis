from app_store_scraper import AppStore
import pandas as pd
import numpy as np
import csv

#scrapping comments
kucoin = AppStore(country='gb', app_name='kucoin', app_id = '1378956601')
kucoin.review(how_many=1000)
kucoindf = pd.DataFrame(np.array(kucoin.reviews),columns=['review'])
kucoindf2 = kucoindf.join(pd.DataFrame(kucoindf.pop('review').tolist()))
kucoindf2.to_csv('kucoin_reviews_1000.csv', index=False)


coinbase = AppStore(country='gb', app_name='coinbase', app_id = '886427730')
coinbase.review(how_many=1000)
coinbasedf = pd.DataFrame(np.array(coinbase.reviews),columns=['review'])
coinbasedf2 = coinbasedf.join(pd.DataFrame(coinbasedf.pop('review').tolist()))
coinbasedf2.to_csv('coinbase_reviews_1000.csv', index=False)

binance = AppStore(country='gb', app_name='binance', app_id = '886427730')
binance.review(how_many=1000)
binancedf = pd.DataFrame(np.array(binance.reviews),columns=['review'])
binancedf2 = binancedf.join(pd.DataFrame(binancedf.pop('review').tolist()))
binancedf2.to_csv('binance_reviews_1000.csv', index=False)