import openai
import csv
import requests
import time
import schedule
import datetime

#openai.api_key = "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"

import os
import sys
os.path.dirname(sys.executable)


CC_API_KEY = '825ed1c29fd024a7f04bddb228d707c78372516c3f651051e0328f7ccfcab1f1'

def get_hourly_exchange_volume(exchange):
    url = f'https://min-api.cryptocompare.com/data/exchange/histohour?api_key={CC_API_KEY}'
    params = {
        'e': exchange,
        'tsym': 'USDT',
        'limit': 1  # Retrieve data for the most recent day
        #'aggregate': 24
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'Data' in data and len(data['Data']) > 0:
        #timestampunix = data['Data'][0]['time']
        #formatted_time = datetime.datetime.utcfromtimestamp(timestampunix).strftime("%H:%M:%S")
        #print(formatted_time)
        volume = data['Data'][0]['volume']
        return volume
    else:
        return None

def write_to_csv(data):
    with open('hourly_exchange_volume.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def main():
    coinbase_volume = get_hourly_exchange_volume('Coinbase')
    binance_volume = get_hourly_exchange_volume('Binance')
    kucoin_volume = get_hourly_exchange_volume('Kucoin')
    kraken_volume = get_hourly_exchange_volume('Kraken')
    timestamp = time.strftime('%Y-%m-%d/%H:%M:%S')
    
    data = [timestamp, binance_volume, coinbase_volume, kucoin_volume, kraken_volume]
    write_to_csv(data)
    print(f"Running script at {timestamp}")

# Call the main function to retrieve daily volumes and append to CSV
#main()
schedule.every().hour.at(":30").do(main)
 #Keep the script running indefinitely
print("Waiting, running at :30 each hour...")
while True:
    schedule.run_pending()
    time.sleep(5)
