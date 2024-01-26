import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

### APPSTORE REVIEWS HAVE A USER RATING, WE CAN COMPARE IT WITH THE SENTIMENT SCORE
### PLAYSTORE DOESNT HAVE IT

dfA_BR_binance = pd.read_csv('csvs/appstore/binance/AS_Binance_BR_bert.csv')
dfP_BR_binance = pd.read_csv('csvs/playstore/binance/PS_Binance_BR_bert.csv')
dfA_BR_coinbase = pd.read_csv('csvs/appstore/coinbase/AS_Coinbase_BR_bert.csv')
dfP_BR_coinbase = pd.read_csv('csvs/playstore/coinbase/PS_Coinbase_BR_bert.csv')
dfA_BR_kucoin = pd.read_csv('csvs/appstore/kucoin/AS_Kucoin_BR_bert.csv')
dfP_BR_kucoin = pd.read_csv('csvs/playstore/kucoin/PS_Kucoin_BR_bert.csv')

dfA_US_binance = pd.read_csv('csvs/appstore/binance/AS_Binance_US_bert.csv')
dfP_US_binance = pd.read_csv('csvs/playstore/binance/PS_Binance_US_bert.csv')
dfA_US_coinbase = pd.read_csv('csvs/appstore/coinbase/AS_Coinbase_US_bert.csv')
dfP_US_coinbase = pd.read_csv('csvs/playstore/coinbase/PS_Coinbase_US_bert.csv')
dfA_US_kucoin = pd.read_csv('csvs/appstore/kucoin/AS_Kucoin_US_bert.csv')
dfP_US_kucoin = pd.read_csv('csvs/playstore/kucoin/PS_Kucoin_US_bert.csv')

def display_sentiment_counts(csv):
    # Calculate the count of each sentiment label in the 'sentiment' column
    sentiment_counts = csv['sentiment'].value_counts()


    # Plotting the bar chart
    colors = ['darkgreen', 'lightgreen', 'yellow', 'orange', 'red']
    # Extract x-values (sentiment labels) and y-values (counts) as lists
    x_values = sentiment_counts.index.tolist()
    y_values = sentiment_counts.values.tolist()

    # Plotting the bar chart
 # Plotting the bar chart
    plt.figure(figsize=(10,6))
    bars = plt.bar(x_values, y_values, color=colors)
    plt.title("Sentiment Distribution - Bar Chart")
    plt.ylabel('Number of Reviews')
    plt.xlabel('Sentiment')
    plt.grid(axis='y')


    # Annotate the count above each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

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

def compare_score_vs_rating(df): ### ONLY WORKS WITH APPSTORE     
    # Map the sentiment to numerical scores
    sentiment_mapping = {
        'very_bad': 1,
        'bad': 2,
        'neutral': 3,
        'good': 4,
        'very_good': 5
    }
    df['sentiment_score'] = df['sentiment'].map(sentiment_mapping)
    
    # Group data by rating and calculate average sentiment score for each rating
    grouped_by_rating = df.groupby('rating')['sentiment_score'].mean()

    # Create a scatter plot with a trend line
    plt.figure(figsize=(10, 6))
    sns.regplot(x=grouped_by_rating.index, y=grouped_by_rating.values, scatter_kws={'s': 100}, color='darkblue')
    plt.title('Average Sentiment Score by User Rating')
    plt.xlabel('User Rating')
    plt.ylabel('Average Sentiment Score')
    plt.grid(True)
    plt.show()


compare_score_vs_rating(dfA_BR_binance)
display_sentiment_counts(dfP_US_binance)
crosscheck_btc(dfP_US_binance)


# Function to plot sentiment distribution
def plot_sentiment_distribution(data, title):
    plt.figure(figsize=(8, 4))
    sns.countplot(data['sentiment'])
    plt.title(f'Sentiment Distribution for {title}')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Count')
    plt.show()

# Function to plot sentiment over time
def plot_sentiment_over_time(data, title):
    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values(by='timestamp')

    # Resampling to monthly average sentiment
    monthly_sentiment = data.resample('M', on='timestamp').mean()

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_sentiment.index, monthly_sentiment['sentiment'], marker='o')
    plt.title(f'Monthly Average Sentiment Over Time for {title}')
    plt.xlabel('Time')
    plt.ylabel('Average Sentiment')
    plt.show()

# Plotting for each dataset

plot_sentiment_distribution(dfP_US_binance, "Binance")
plot_sentiment_over_time(dfP_US_binance, "Binance")

plot_sentiment_distribution(dfP_US_coinbase, "Coinbase")
plot_sentiment_over_time(dfP_US_coinbase, "Coinbase")

plot_sentiment_distribution(dfP_US_kucoin, "Kucoin")
plot_sentiment_over_time(dfP_US_kucoin, "Kucoin")