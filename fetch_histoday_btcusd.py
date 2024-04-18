import requests
import pandas as pd
from datetime import datetime


#Fetch OHLCV data from the BTC/USD pair since 2017
def fetch_ohlcv_data(symbol, comparison_symbol, limit, aggregate, timestamp):
    url = 'https://min-api.cryptocompare.com/data/v2/histoday'
    params = {
        'fsym': symbol,
        'tsym': comparison_symbol,
        'limit': limit,
        'aggregate': aggregate,
        'toTs': timestamp
        #'api_key': CC_API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()['Data']['Data']
    return data

#Cleaning BTC_USD_daily_price.csv to only have timestamp, closingPrice(close), and volumeto (USD)   
def filter_BTC():
    # Load the dataset
    btc = pd.read_csv('csvs/btc_usd_ohlcv.csv')
    
    # Keep only the specified columns
    to_keep = ['time', 'volumeto', 'close']
    new = btc[to_keep]
    
    # Save the filtered dataframe to a new CSV file
    new.to_csv('csvs/BTC_daily_pricevol.csv', index=False)


def main():
    symbol = 'BTC'
    comparison_symbol = 'USD'
    aggregate = 1  # 1 day
    limit = 2000   # max limit

    # Start from the most recent day and go backwards
    end_timestamp = int(datetime.now().timestamp())

    all_data = []
    while True:
        data = fetch_ohlcv_data(symbol, comparison_symbol, limit, aggregate, end_timestamp)
        if not data:
            break
        all_data.extend(data)
        # Update the timestamp to the earliest date in the fetched data
        end_timestamp = data[0]['time']
        # Stop if the data reaches 2017 or earlier
        if datetime.fromtimestamp(end_timestamp).year <= 2017:
            break

    df = pd.DataFrame(all_data)
    df['time'] = pd.to_datetime(df['time'], unit='s')  # Convert timestamp to datetime
    df.to_csv('btc_usd_ohlcv.csv', index=False)
    #print('Data saved to btc_usd_ohlcv.csv')
    
    filter_BTC()

if __name__ == '__main__':
    main()