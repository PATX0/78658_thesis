import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

dfA_BR_binance = pd.read_csv('csvs/appstore/binance/AS_Binance_BR_bert.csv')
dfP_BR_binance = pd.read_csv('csvs/playstore/binance/PS_Binance_BR_bert.csv')
dfA_BR_coinbase = pd.read_csv('csvs/appstore/coinbase/AS_Coinbase_BR_bert.csv')
dfP_BR_coinbase = pd.read_csv('csvs/playstore/coinbase/PS_Coinbase_BR_bert.csv')
dfA_BR_kucoin = pd.read_csv('csvs/appstore/kucoin/AS_Kucoin_BR_bert.csv')
dfP_BR_kucoin = pd.read_csv('csvs/playstore/kucoin/PS_Kucoin_BR_bert.csv')

dfA_US_binance = pd.read_csv('csvs/appstore/binance/AS_Binance_US.csv')
dfP_US_binance = pd.read_csv('csvs/playstore/binance/PS_Binance_US.csv')
dfA_US_coinbase = pd.read_csv('csvs/appstore/coinbase/AS_Coinbase_US.csv')
dfP_US_coinbase = pd.read_csv('csvs/playstore/coinbase/PS_Coinbase_US.csv')
dfA_US_kucoin = pd.read_csv('csvs/appstore/kucoin/AS_Kucoin_US.csv')
dfP_US_kucoin = pd.read_csv('csvs/playstore/kucoin/PS_Kucoin_US.csv')

def display_sentiment_counts(csv):
    # Calculate the count of each sentiment label in the 'sentiment' column
    sentiment_counts = csv['sentiment'].value_counts()


    # Plotting the bar chart
    colors = ['darkgreen', 'lightgreen', 'yellow', 'orange', 'red']
    # Extract x-values (sentiment labels) and y-values (counts) as lists
    x_values = sentiment_counts.index.tolist()
    y_values = sentiment_counts.values.tolist()

    # Plotting the bar chart
    plt.figure(figsize=(10,6))
    plt.bar(x_values, y_values, color=colors)
    plt.title("Sentiment Distribution - Bar Chart")
    plt.ylabel('Number of Reviews')
    plt.xlabel('Sentiment')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


def crosscheck_btc(df_sentiment):

    # Load the sentiment data
    df_sentiment['timestamp'] = pd.to_datetime(df_sentiment['timestamp'])
    df_sentiment['year_month'] = df_sentiment['timestamp'].dt.to_period('M')
    sentiment_mapping = {'very_bad': 1, 'bad': 2, 'neutral': 3, 'good': 4, 'very_good': 5}
    df_sentiment['sentiment'] = df_sentiment['sentiment'].map(sentiment_mapping)
    df_sentiment_monthly = df_sentiment.groupby('year_month')['sentiment'].mean().reset_index()

    # Load the BTC/USD data
    dfBTC = pd.read_csv('csvs/BTCUSD_daily_price.csv')
    dfBTC['timestamp'] = pd.to_datetime(dfBTC['timestamp'])
    dfBTC['year_month'] = dfBTC['timestamp'].dt.to_period('M')
    dfBTC_monthly = dfBTC.groupby('year_month')['price'].mean().reset_index()

    # Plotting
    fig, ax1 = plt.subplots(figsize=(15, 6))

    # Plotting sentiment
    ax1.plot(df_sentiment_monthly['year_month'].dt.to_timestamp(), df_sentiment_monthly['sentiment'], label='Average Sentiment', color='blue')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Average Sentiment Score', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('Average Sentiment and BTC Price Over Time')

    # Plotting BTC price on a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(dfBTC_monthly['year_month'].dt.to_timestamp(), dfBTC_monthly['price'], label='BTC Price', color='orange')
    ax2.set_ylabel('BTC Price in USD', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Additional settings
    plt.xticks(rotation=45)
    ax1.grid(True)

    plt.show()


crosscheck_btc(dfP_US_binance)

