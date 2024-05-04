import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

### APPSTORE REVIEWS HAVE A USER RATING, WE CAN COMPARE IT WITH THE SENTIMENT SCORE
### PLAYSTORE DOESNT HAVE IT

#BINANCE EXCHANGE CSVS
#playstore
dbUSP = pd.read_csv('csvs/binance/PS_Binance_US_bert.csv')
dbNGP = pd.read_csv('csvs/binance/PS_Binance_NG_bert.csv')
dbUAP = pd.read_csv('csvs/binance/PS_Binance_UA_bert.csv')
dbCNP = pd.read_csv('csvs/binance/PS_Binance_CN_bert.csv')
dbBRP = pd.read_csv('csvs/binance/PS_Binance_BR_bert.csv')
#appstore
dbUSA = pd.read_csv('csvs/binance/AS_Binance_US_bert.csv')
dbNGA = pd.read_csv('csvs/binance/AS_Binance_NG_bert.csv')
dbUAA = pd.read_csv('csvs/binance/AS_Binance_UA_bert.csv')
dbCNA = pd.read_csv('csvs/binance/AS_Binance_CN_bert.csv')
dbBRA = pd.read_csv('csvs/binance/AS_Binance_BR_bert.csv')

#COINBASE
#playstore
dcUSP = pd.read_csv('csvs/coinbase/PS_Coinbase_US_bert.csv')
dcNGP = pd.read_csv('csvs/coinbase/PS_Coinbase_NG_bert.csv')
dcUAP = pd.read_csv('csvs/coinbase/PS_Coinbase_UA_bert.csv')
dcCNP = pd.read_csv('csvs/coinbase/PS_Coinbase_CN_bert.csv')
dcBRP = pd.read_csv('csvs/coinbase/PS_Coinbase_BR_bert.csv')
#appstore
dcUSA = pd.read_csv('csvs/coinbase/AS_Coinbase_US_bert.csv')
dcNGA = pd.read_csv('csvs/coinbase/AS_Coinbase_NG_bert.csv')
dcUAA = pd.read_csv('csvs/coinbase/AS_Coinbase_UA_bert.csv')
dcCNA = pd.read_csv('csvs/coinbase/AS_Coinbase_CN_bert.csv')
dcBRA = pd.read_csv('csvs/coinbase/AS_Coinbase_BR_bert.csv')

#KUCOIN
#playstore
dkUSP = pd.read_csv('csvs/kucoin/PS_Kucoin_US_bert.csv')
dkNGP = pd.read_csv('csvs/kucoin/PS_Kucoin_NG_bert.csv')
dkUAP = pd.read_csv('csvs/kucoin/PS_Kucoin_UA_bert.csv')
dkCNP = pd.read_csv('csvs/kucoin/PS_Kucoin_CN_bert.csv')
dkBRP = pd.read_csv('csvs/kucoin/PS_Kucoin_BR_bert.csv')
#appstore
dkUSA = pd.read_csv('csvs/kucoin/AS_Kucoin_US_bert.csv')
dkNGA = pd.read_csv('csvs/kucoin/AS_Kucoin_NG_bert.csv')
dkUAA = pd.read_csv('csvs/kucoin/AS_Kucoin_UA_bert.csv')
dkCNA = pd.read_csv('csvs/kucoin/AS_Kucoin_CN_bert.csv')
dkBRA = pd.read_csv('csvs/kucoin/AS_Kucoin_BR_bert.csv')

brTotal = pd.read_csv('csvs/BR/BRtotal.csv')
cnTotal = pd.read_csv('csvs/CN/CNtotal.csv')
ngTotal = pd.read_csv('csvs/NG/NGtotal.csv')
uaTotal = pd.read_csv('csvs/UA/UAtotal.csv')
usTotal = pd.read_csv('csvs/US/UStotal.csv')

binancetotal = pd.read_csv('csvs/binance/binanceTotal.csv')
coinbasetotal = pd.read_csv('csvs/coinbase/coinbaseTotal.csv')
kucointotal = pd.read_csv('csvs/kucoin/kucoinTotal.csv')

btc = pd.read_csv('csvs/BTC_daily_pricevol.csv')

palette = {'Binance': 'orange', 'Coinbase': 'blue', 'Kucoin': 'green',
            'BR': 'gold', 'CN': 'red', 'NG': 'darkgreen', 'UA': 'cyan', 'US': 'indigo' }
# Groups the average sentiment by year
def preprocess_sentiment_year(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Filter the data to include only entries after 2017-01-01 (coinbase has)
    df = df[df['timestamp'] > pd.Timestamp('2017-01-01')]
    df['year'] = df['timestamp'].dt.year
    df = df.groupby('year')['sentiment'].mean().reset_index()
    return df

# Groups the average btc price by month
def preprocess_btc_month(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year_month'] = df['timestamp'].dt.to_period('M')  # Grouping by year and month
    monthly_avg = df.groupby('year_month')['price'].mean().reset_index()
    monthly_avg['year_month'] = monthly_avg['year_month'].dt.to_timestamp()  # Converting to timestamp for plotting
    return monthly_avg

# Groups the average btc price by year
def preprocess_btc_year(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['timestamp'].dt.year  # Extracting year as an integer
    annual_avg = df.groupby('year')['price'].mean().reset_index()
    return annual_avg

# Groups the average sentiment by month
def preprocess_sentiment_month(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Filter the data to include only entries after 2017-01-01 (coinbase has)
    df = df[df['timestamp'] > pd.Timestamp('2017-01-01')]
    df['year_month'] = df['timestamp'].dt.to_period('M')  # Grouping by year and month
    monthly_avg = df.groupby('year_month')['sentiment'].mean().reset_index()
    monthly_avg['year_month'] = monthly_avg['year_month'].dt.to_timestamp()  # Converting to timestamp for plotting
    return monthly_avg

# Function to plot sentiment trends
def plot_sentiment_trends(df_dict, title):
    plt.figure(figsize=(10, 6))
    for label, df in df_dict.items():
        color = palette.get(label, 'gray')  # Default to 'gray' if label is not found in palette
        plt.plot(df['year'], df['sentiment'], label=label, marker='o', color=color)
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Average Sentiment')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_sentiment_btc_year(sentiment_data, btc_data):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plotting sentiment trends
    for label, df in sentiment_data.items():
        ax1.plot(df['year'], df['sentiment'], label=f"{label} Sentiment", marker='o', color=palette[label])

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Average Sentiment')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.set_title("Sentiment and BTC Price Trends Over Time")

    # Adjust Bitcoin plotting to match sentiment data years
    ax2 = ax1.twinx()
    ax2.plot(btc_data['year'], btc_data['price'], label='BTC Price', color='black', marker='x', linestyle='--')
    ax2.set_ylabel('BTC Average Close Price (USD)')
    ax2.legend(loc='upper right')

    plt.show()


def plot_sentiment_btc_year_month(sentiment_data, btc_data):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plotting sentiment trends
    for label, df in sentiment_data.items():
        ax1.plot(df['year_month'], df['sentiment'], label=f"{label} Sentiment", marker='o', color=palette.get(label, 'gray'))

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Average Sentiment')
    ax1.xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks to display only years
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format ticks to display only the year
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.set_title("Sentiment and BTC Price Trends Over Time")

    # Adjust Bitcoin plotting to match sentiment data timing
    ax2 = ax1.twinx()
    ax2.plot(btc_data['year_month'], btc_data['price'], label='BTC Price', color='black', marker='x', linestyle='--')
    ax2.set_ylabel('BTC Average Close Price (USD)')
    ax2.legend(loc='upper right')

    plt.show()

# Function to plot sentiment trends using heatmaps
def plot_heatmap(df_dict, x):
    # Transform dictionary of DataFrames into a single DataFrame for heatmap
    heatmap_data = pd.DataFrame({
        label: df.set_index('year')['sentiment']
        for label, df in df_dict.items()
    })
    heatmap_data = heatmap_data.T  # Transpose to make years the columns and labels the rows

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm_r", fmt=".2f",vmin = 2, vmax = 4)
    if x == 'e' :
        plt.ylabel('Exchange')
        plt.title('Average Sentiment Heatmap by Exchange')
    if x == 'c':
        plt.ylabel('Country')
        plt.title('Average Sentiment Heatmap by Country')
    plt.xlabel('Year')
    plt.show()

# Function that checks if the rating == sentiment 
def calculate_hit_rate(df, exchange_name):
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')  # Ensuring 'rating' is numeric
    df = df.dropna(subset=['rating', 'sentiment'])  # Ensure both fields are non-empty
    df['match_type'] = (df['rating'] == df['sentiment']).replace({True: 'Exact Match', False: 'Mismatch'})
    df['exchange'] = exchange_name  # Add exchange name for grouping in plots
    return df

# Function to plot the hit rate of sentiment score
def plot_hitrate_sentiment(b,c,k):
    binance = calculate_hit_rate(b, 'Binance')
    coinbase = calculate_hit_rate(c, 'Coinbase')
    kucoin = calculate_hit_rate(k, 'Kucoin')
    combined_df = pd.concat([binance, coinbase, kucoin], ignore_index=True)

    # Calculate percentages
    match_counts = combined_df.groupby(['exchange', 'match_type']).size().reset_index(name='counts')
    total_counts = combined_df.groupby('exchange').size().reset_index(name='total')
    match_counts = match_counts.merge(total_counts, on='exchange')
    match_counts['percentage'] = (match_counts['counts'] / match_counts['total']) * 100

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='exchange', y='percentage', hue='match_type', data=match_counts, palette="pastel")
    plt.title('Sentiment Score vs Rating Across Exchanges (Percentage)')
    plt.xlabel('Exchanges')
    plt.ylabel('Percentage')
    plt.legend(title='Match Type')

    # Annotate percentages on bars
    for p in plt.gca().patches:
        height = p.get_height()
        plt.gca().text(p.get_x() + p.get_width() / 2, height + 0.5, '{:1.2f}%'.format(height),
                       ha="center")

    plt.show()

#function to plot sentiment distribution count
def plot_sentiment_distribution(exchange_data):
    # Create an empty DataFrame to store all data
    combined_data = pd.DataFrame()

    # Loop through each exchange data and append to the combined DataFrame
    for exchange_name, data in exchange_data.items():
        # Assuming sentiment scores are in a column named 'sentiment'
        temp = data[['sentiment']].copy()  # Select only the 'sentiment' column
        temp['Exchange'] = exchange_name  # Add exchange name for grouping in plot
        combined_data = pd.concat([combined_data, temp], ignore_index=True)

    # Plotting the sentiment score distribution using count plot
    plt.figure(figsize=(12, 6))
    sns.countplot(x='sentiment', hue='Exchange', data=combined_data, palette=palette)
    plt.title('Sentiment Score Distribution Across Exchanges')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Count')
    plt.legend(title='Exchange')
    plt.show()

def main():
# Preprocessing each DataFrame
    # Define custom colors for each exchange
    btc_data_y = preprocess_btc_year(btc)
    btc_data_m = preprocess_btc_month(btc)   

    binancem = preprocess_sentiment_month(binancetotal)
    coinbasem = preprocess_sentiment_month(coinbasetotal)
    kucoinm = preprocess_sentiment_month(kucointotal)

    binance = preprocess_sentiment_year(binancetotal)
    coinbase = preprocess_sentiment_year(coinbasetotal)
    kucoin = preprocess_sentiment_year(kucointotal)

    br = preprocess_sentiment_year(brTotal)
    cn  = preprocess_sentiment_year(cnTotal)
    ng = preprocess_sentiment_year(ngTotal)
    ua = preprocess_sentiment_year(uaTotal)
    us = preprocess_sentiment_year(usTotal)
    # Gather the data
    exchange_data_y = {
        'Binance': binance,
        'Coinbase': coinbase,
        'Kucoin': kucoin
    }
    exchange_data_m = {
        'Binance': binancem,
        'Coinbase': coinbasem,
        'Kucoin': kucoinm
    }
    exchanges = {
        'Binance': binancetotal,
        'Coinbase': coinbasetotal,
        'Kucoin': kucointotal
    }
    country_totals = {
        'BR': br,
        'CN': cn,
        'NG': ng,
        'UA': ua,
        'US': us
    }


    


    #Plot sentiment counts for all exchanges
    plot_sentiment_distribution(exchanges)
    # Plot sentiment trends over time for exchanges
    plot_sentiment_trends(exchange_data_y, 'Average Sentiment by Exchange')
    # Plot sentiment trends over time for countries
    plot_sentiment_trends(country_totals, 'Average Sentiment by Country')

    #Plot sentiment trends by month and correlate with btc price  & volume
    plot_sentiment_btc_year_month(exchange_data_m, btc_data_m)

    #plot sentiment trend by year and btc price yearly
    plot_sentiment_btc_year(exchange_data_y, btc_data_y)
    # Plot sentiment trends over time for exchanges with heatmap
    plot_heatmap(exchange_data_y, 'e')
    # Plot sentiment trends over time for countries with heatmap
    plot_heatmap(country_totals, 'c')     

    # Plot hit rate
    plot_hitrate_sentiment(binancetotal, coinbasetotal, kucointotal)

# Execution starts here
if __name__ == '__main__':
    main()



