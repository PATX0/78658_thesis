import csv
import requests
import time
import schedule
import datetime

CC_API_KEY = '825ed1c29fd024a7f04bddb228d707c78372516c3f651051e0328f7ccfcab1f1'
ts=1685398634 #seconds passed since 01/01/1970
formatted_time = datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
print(formatted_time)

def get_trending_coins_by_volume():
    url = 'https://min-api.cryptocompare.com/data/top/totaltoptiervolfull?api_key={CC_API_KEY}'
    params = {
        'limit': 15,
        'tsym': 'USDT'
    }
    #https://min-api.cryptocompare.com/data/top/totaltoptiervolfull?limit=10&tsym=USDT
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'Data' in data:
        filtered_data = [entry for entry in data['Data'] if 'USD' not in entry['CoinInfo']['FullName'] and 'Tether' not in entry['CoinInfo']['FullName']]
        return filtered_data
    else:
        return None





def get_sentiment(symbol):
    """inOutVar calculates the net change of in/out of the money addresses, the number of "In the Money" addresses increases is bullish signal.
    largetxsVar is is bullish when the short term trend of the number of txs > $100k is greater than the long term average.
    addressesNetGrowthVar is bullish when more addresses are being created than emptied.
    concentrationVar is based on the accumulation (bullish) or reduction (bearish) of addresses with more than 0.1% of the circulating supply."""
    #https://min-api.cryptocompare.com/data/tradingsignals/intotheblock/latest?fsym=ETH
    url = 'https://min-api.cryptocompare.com/data/tradingsignals/intotheblock/latest?api_key={CC_API_KEY}'
    params = {
        'fsym': symbol
    }
    response = requests.get(url, params=params)
    data = response.json()

    bull_count = 0
    bear_count = 0
    neutral_count = 0
    if data['Response'] == 'Success': 
        sentiment_data = data['Data']
        symbol = sentiment_data['symbol']
        #print(f"Sentiment Analysis of {symbol}:")
        #print(f"InOutVar: Sentiment - {sentiment_data['inOutVar']['sentiment']}, Score - {round(sentiment_data['inOutVar']['score'], 2)}")
        #print(f"largetxsVar: Sentiment - {sentiment_data['largetxsVar']['sentiment']}, Score - {round(sentiment_data['largetxsVar']['score'], 2)}")
        #print(f"addressesNetGrowthVar: Sentiment - {sentiment_data['addressesNetGrowth']['sentiment']}, Score - {round(sentiment_data['addressesNetGrowth']['score'], 2)}")
        #print(f"concentrationVar: Sentiment - {sentiment_data['concentrationVar']['sentiment']}, Score - {round(sentiment_data['concentrationVar']['score'], 2)}")
        if sentiment_data['inOutVar']['sentiment'] == 'bullish':
            bull_count += 1
        if sentiment_data['inOutVar']['sentiment'] == 'neutral':
            neutral_count += 1
        else: bear_count += 1
        if sentiment_data['largetxsVar']['sentiment'] == 'bullish':
            bull_count += 1
        if sentiment_data['largetxsVar']['sentiment'] == 'neutral':
            neutral_count += 1
        else: bear_count += 1
        if sentiment_data['addressesNetGrowth']['sentiment'] == 'bullish':
            bull_count += 1
        if sentiment_data['addressesNetGrowth']['sentiment'] == 'neutral':
            neutral_count += 1
        else: bear_count += 1
        if sentiment_data['concentrationVar']['sentiment'] == 'bullish':
            bull_count += 1
        if sentiment_data['concentrationVar']['sentiment'] == 'neutral':
            neutral_count += 1
        else: bear_count += 1
        if bull_count >= 3 or (bear_count == 1 and neutral_count == 1):
            print(f"Bullish on {symbol}")
        elif bear_count >= 3 or (bull_count == 1 and neutral_count == 1):
            print(f"Bearish on {symbol}")
        else:
            print(f"Neutral on {symbol}")
    else:
        print(f"Unable to retrieve sentiment data for {symbol}.")
        
    return data['Data']


def get_daily_volume(symbol, exchange):
    url = f'https://min-api.cryptocompare.com/data/exchange/symbol/histoday?fsym={symbol}&tsym=USDT&limit=10&e={exchange}&api_key={CC_API_KEY}'
    
    response = requests.get(url)
    data = response.json()
    
    daily_volume = []
    
    if 'Data' in data:
        for volume_data in data['Data']:
            from_volume = volume_data['volumeFrom']
            to_volume = volume_data['volumeTo']
            daily_volume.append({'From Volume': from_volume, 'To Volume': to_volume})
    
    return daily_volume

#for volume in get_daily_volume(Bitcoin, COINBASE):
#    print(f'{exchange} Daily Volume for {symbol}:')
#    print(f'From Volume: {volume["From Volume"]}')
#    print(f'To Volume: {volume["To Volume"]}')
#    print('-' * 50)

def main():
    tickers = []
    total_top_tier_volume = get_trending_coins_by_volume()
    if total_top_tier_volume:
        for entry in total_top_tier_volume:
            coin_name = entry['CoinInfo']['FullName']
            volume = entry['RAW']['USDT']['VOLUME24HOURTO']
            print(f"{coin_name}: {volume}")
            tickers.append(entry['CoinInfo']['Name'])
    else:
        print('Failed to retrieve total top tier volume data.')
    print('\n')

    for ticker in tickers:
        data = get_sentiment(ticker)

main()
