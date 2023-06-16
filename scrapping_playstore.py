from google_play_scraper import Sort, reviews_all, reviews
import csv


exchanges = ['com.coinbase.android','com.binance.dev', 'com.kubi.kucoin']

#scrapping comments
country_codes = {
    'Ukraine': 'UA',
    'Nigeria': 'NG',
    'USA': 'US',
    'China': 'CN',
    'Brazil': 'BR'
}
countries = ['UA','NG','US','CN','BR']

def scrap(country):
        #binance
        # binance = reviews_all(
        #     'com.binance.dev',
        #     lang='en', # defaults to 'en'
        #     country=country, # defaults to 'us'
        #     #sort=Sort.MOST_RELEVANT, # defaults to Sort.NEWEST
        #     #count=5000 # defaults to 100
        # )
        #coinbase
        coinbase = reviews_all(
            'com.coinbase.android',
            lang='en', # defaults to 'en'
            country=country, # defaults to 'us'
            #sort=Sort.MOST_RELEVANT, # defaults to Sort.NEWEST
            #count=5000 # defaults to 100
        )
        #kucoin
        # kucoin = reviews_all(
        #     'com.kubi.kucoin',
        #     lang='en', # defaults to 'en'
        #     country=country, # defaults to 'us'
        #     #sort=Sort.MOST_RELEVANT, # defaults to Sort.NEWEST
        #     #count=5000 # defaults to 100
        #  )
        return [0,coinbase,0]  #binance,coinbase,kucoin

for c in countries:
    data = scrap(c)
    #binance = data[0]
    coinbase = data[1]
    #kucoin =data[2]
#BINANCE
    # filename = f'playstore_binance_reviews_{c}.csv'
    # with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['timestamp', 'reviews'])

    #     for review in binance:
    #         timestamp = review['at']
    #         content = review['content']
    #         writer.writerow([timestamp, content])

    # print(f"CSV file '{filename}' has been created successfully.")
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
    # filename = f'playstore_kucoin_reviews_{c}.csv'
    # with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['timestamp', 'reviews'])

    #     for review in kucoin:
    #         timestamp = review['at']
    #         content = review['content']
    #         writer.writerow([timestamp, content])

    # print(f"CSV file '{filename}' has been created successfully.")

