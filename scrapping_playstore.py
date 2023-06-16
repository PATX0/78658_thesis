from google_play_scraper import Sort, reviews_all, reviews
import csv


exchanges = ['com.coinbase.android','com.binance.dev', 'com.kubi.kucoin']

#scrapping comments
lang_codes = {
    'UA': 'uk',
    'NG': 'en',
    'US': 'en',
    'CN': 'zh',
    'BR': 'pt'
}

countries = ['UA','NG','US','CN','BR']

def scrap_coinbase(country):
    #coinbase
    coinbase = reviews_all(
            'com.coinbase.android',
            lang=lang_codes[country],
            country=country, # defaults to 'us'
            #sort=Sort.MOST_RELEVANT, # defaults to Sort.NEWEST
            #count=5000 # defaults to 100
    )
    return coinbase

def scrap_binance(country):
    #binance
    binance = reviews_all(
            'com.binance.dev',
            lang=lang_codes[country],
            country=country, # defaults to 'us'
            #sort=Sort.MOST_RELEVANT, # defaults to Sort.NEWEST
            #count=5000 # defaults to 100
    )
    return binance
     
def scrap_kucoin(country):
    #kucoin
    kucoin = reviews_all(
        'com.kubi.kucoin',
        lang=lang_codes[country],
        country=country, # defaults to 'us'
        #sort=Sort.MOST_RELEVANT, # defaults to Sort.NEWEST
        #count=5000 # defaults to 100
    )
    return kucoin
     

for c in countries:
    binance = scrap_binance(c)
    coinbase = scrap_coinbase(c)
    kucoin = scrap_kucoin(c)
#BINANCE
    filename = f'playstore_binance_reviews_{c}.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'reviews'])

        for review in binance:
            timestamp = review['at']
            content = review['content']
            writer.writerow([timestamp, content])

    print(f"CSV file '{filename}' has been created successfully.")
#COINBASE
    filename = f'playstore_coinbase_reviews_{c}.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'reviews'])

        for review in coinbase:
            timestamp = review['at']
            content = review['content']
            writer.writerow([timestamp, content])

    print(f"CSV file '{filename}' has been created successfully.")
#KUCOIN
    filename = f'playstore_kucoin_reviews_{c}.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'reviews'])

        for review in kucoin:
            timestamp = review['at']
            content = review['content']
            writer.writerow([timestamp, content])

    print(f"CSV file '{filename}' has been created successfully.")

