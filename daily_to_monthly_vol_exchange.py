import openai
import csv
import requests
import time
import schedule

#openai.api_key = "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"




CC_API_KEY = '825ed1c29fd024a7f04bddb228d707c78372516c3f651051e0328f7ccfcab1f1'

def get_daily_exchange_volume(exchange):
    url = f'https://min-api.cryptocompare.com/data/exchange/histoday?api_key={CC_API_KEY}'
    params = {
        'e': exchange,
        'tsym': 'USD',
        'limit': 1  # Retrieve data for the most recent day
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'Data' in data and len(data['Data']) > 0:
        volume = data['Data'][0]['volume']
        return volume
    else:
        return None

def write_to_csv(data):
    with open('daily_exchange_volume.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def main():
    coinbase_volume = get_daily_exchange_volume('Coinbase')
    binance_volume = get_daily_exchange_volume('Binance')
    kucoin_volume = get_daily_exchange_volume('Kucoin')
    kraken_volume = get_daily_exchange_volume('Kraken')
    timestamp = time.strftime('%Y-%m-%d')
    
    data = [timestamp, binance_volume, coinbase_volume, kucoin_volume, kraken_volume]
    write_to_csv(data)

# Call the main function to retrieve daily volumes and append to CSV
main()


""" schedule.every().day.at("12:00").do(main)

while True:
    schedule.run_pending()
    time.sleep(1)
 """

