import pandas as pd



dfA_US_binance = pd.read_csv('csvs/appstore/binance/AS_Binance_US.csv')
dfP_US_binance = pd.read_csv('csvs/playstore/binance/PS_Binance_US.csv')
dfA_US_coinbase = pd.read_csv('csvs/appstore/coinbase/AS_Coinbase_US.csv')
dfP_US_coinbase = pd.read_csv('csvs/playstore/coinbase/PS_Coinbase_US.csv')
dfA_US_kucoin = pd.read_csv('csvs/appstore/kucoin/AS_Kucoin_US.csv')
dfP_US_kucoin = pd.read_csv('csvs/playstore/kucoin/PS_Kucoin_US.csv')

def normalize_csv(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])    
    sentiment_mapping = {'very_bad': 1, 'bad': 2, 'neutral': 3, 'good': 4, 'very_good': 5}
    df['sentiment'] = df['sentiment'].map(sentiment_mapping)