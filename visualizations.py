import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


brTotal = pd.read_csv('csvs/BR/BRtotal.csv')
cnTotal = pd.read_csv('csvs/CN/CNtotal.csv')
ngTotal = pd.read_csv('csvs/NG/NGtotal.csv')
uaTotal = pd.read_csv('csvs/UA/UAtotal.csv')
usTotal = pd.read_csv('csvs/US/UStotal.csv')

binancetotal = pd.read_csv('csvs/binance/binanceTotal.csv')
coinbasetotal = pd.read_csv('csvs/coinbase/coinbaseTotal.csv')
kucointotal = pd.read_csv('csvs/kucoin/kucoinTotal.csv')

btc = pd.read_csv('csvs/BTC_daily_pricevol.csv')
eth = pd.read_csv('csvs/ETH_daily_pricevol.csv')

palette = {'Binance': 'orange', 'Coinbase': 'blue', 'Kucoin': 'green',
            'BR': 'gold', 'CN': 'red', 'NG': 'darkgreen', 'UA': 'cyan', 'US': 'indigo' }


def preprocess_sentiment(df, period):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Filter the data to include only entries after 2017-01-01
    df = df[df['timestamp'] > pd.Timestamp('2017-01-01')]

    if period == 'year':
        df['year'] = df['timestamp'].dt.year
        return df.groupby('year')['sentiment'].mean().reset_index()
    elif period == 'month':
        df['year_month'] = df['timestamp'].dt.to_period('M')
        monthly_avg = df.groupby('year_month')['sentiment'].mean().reset_index()
        monthly_avg['year_month'] = monthly_avg['year_month'].dt.to_timestamp()
        return monthly_avg

def preprocess_price(df, period):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['timestamp'] > pd.Timestamp('2017-01-01')]

    if period == 'year':
        df['year'] = df['timestamp'].dt.year
        return df.groupby('year')['price'].mean().reset_index()
    elif period == 'month':
        df['year_month'] = df['timestamp'].dt.to_period('M')
        monthly_avg = df.groupby('year_month')['price'].mean().reset_index()
        monthly_avg['year_month'] = monthly_avg['year_month'].dt.to_timestamp()
        return monthly_avg

# Groups the average btc volume by month
def preprocess_volume(df,period):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['timestamp'] > pd.Timestamp('2017-01-01')]

    if period == 'year': 
        df['year'] = df['timestamp'].dt.year
        return df.groupby('year')['price'].mean().reset_index()
    elif period == 'month':
        df['year_month'] = df['timestamp'].dt.to_period('M')  # Grouping by year and month (!= .dt.month -> it would display the total avg by month from all years )
        monthly_avg = df.groupby('year_month')['volume'].mean().reset_index()
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

# Plot sentiment avg and btc price by year
def plot_sentiment_btc_year(sentiment_data, btc_data):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plotting sentiment trends
    for label, df in sentiment_data.items():
        ax1.plot(df['year'], df['sentiment'], label=f"{label} Sentiment", marker='o', color=palette[label])

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Average Sentiment')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.set_title("Sentiment and BTC Price Trends Over Time  (yearly)")

    # Adjust Bitcoin plotting to match sentiment data years
    ax2 = ax1.twinx()
    ax2.plot(btc_data['year'], btc_data['price'], label='BTC Price', color='black', marker='x', linestyle='--')
    ax2.set_ylabel('BTC Average Close Price (USD)')
    ax2.legend(loc='upper right')

    plt.show()

# Plot sentiment avg and btc price by month 
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
    ax1.set_title("Sentiment and BTC Price Trends Over Time (monthly)")

    # Adjust Bitcoin plotting to match sentiment data timing
    ax2 = ax1.twinx()
    ax2.plot(btc_data['year_month'], btc_data['price'], label='BTC Price', color='black', marker='x', linestyle='--')
    ax2.set_ylabel('BTC Average Close Price (USD)')
    ax2.legend(loc='upper right')

    plt.show()

# Plot sentiment avg and btc volume by month
def plot_sentiment_and_btc_volume(sentiment_data, btc_volume_data):
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
    ax1.set_title("Sentiment and BTC Volume Trends Over Time")

    # Adjust Bitcoin plotting to match sentiment data timing and use log scale
    ax2 = ax1.twinx()
    ax2.plot(btc_volume_data['year_month'], btc_volume_data['volume'], label='BTC Volume', color='black', marker='x', linestyle='--')
    ax2.set_yscale('log')  # Using logarithmic scale
    ax2.set_ylabel('BTC Average Trading Volume (Log Scale)')
    ax2.legend(loc='upper right')

    plt.show()

#ETH
def plot_sentiment_eth_year_month(sentiment_data, eth_data):
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
    ax1.set_title("Sentiment and eth Price Trends Over Time (monthly)")

    # Adjust Bitcoin plotting to match sentiment data timing
    ax2 = ax1.twinx()
    ax2.plot(eth_data['year_month'], eth_data['price'], label='eth Price', color='black', marker='x', linestyle='--')
    ax2.set_ylabel('eth Average Close Price (USD)')
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

# Function to plot sentiment distribution count
def plot_sentiment_frequency(exchange_data):
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
    ax = sns.countplot(x='sentiment', hue='Exchange', data=combined_data, palette=palette)
    plt.title('Sentiment Score Distribution Across Exchanges')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Count')
    plt.legend(title='Exchange')
     # Annotate the exact counts above the bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
    
    plt.show()

# Function to boxplot sentiment distro
def plot_boxplot_distribution(exchanges):
    # Combine all dataframes into one with a new 'Group' column
    combined_df = pd.DataFrame()
    for label, df in exchanges.items():
        df['Group'] = label
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Group', y='sentiment',hue='Group', data=combined_df, palette=palette)
    plt.title('Sentiment Score Distribution Across Exchanges')
    plt.xlabel('Exchanges')
    plt.ylabel('Sentiment Score')
    plt.show()

# plot time series decomposition
def preprocess_and_decompose(df, title):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['timestamp'] > pd.Timestamp('2017-01-01')]
    df['year_month'] = df['timestamp'].dt.to_period('M')  # Grouping by year and month
    df = df.groupby('year_month')['sentiment'].mean().reset_index()
    df['year_month'] = df['year_month'].dt.to_timestamp()  # Converting to timestamp for plotting

    # Ensure 'year_month' is datetime type and set as index
    df.set_index('year_month', inplace=True)
    
    # Decomposition using the index as the time series x-axis
    decomposition = seasonal_decompose(df['sentiment'], model='additive', period=12)
    fig = decomposition.plot()
    fig.suptitle(title, fontsize=16)
    
    # Enhance x-axis to display dates better
    plt.gcf().autofmt_xdate()  # Auto format x-axis dates to look better
    plt.show()

# Plot violin sentiment distro
def plot_violin_distribution(dfs, labels):
    combined_df = pd.DataFrame()
    for df, label in zip(dfs, labels):
        df['label'] = label
        combined_df = pd.concat([combined_df, df], axis=0)
    
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='label', y='sentiment', data=combined_df)
    plt.title('Sentiment Score Distribution Across Exchanges and Countries')
    plt.xlabel('Group')
    plt.ylabel('Sentiment Scores')
    plt.show()



def preprocess_2017(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['timestamp'] > pd.Timestamp('2017-01-01')]
    return df

def main():
# Preprocessing each DataFrame
    # Define custom colors for each exchange
    btc_data_y = preprocess_price(btc,'year')
    btc_data_m = preprocess_price(btc,'month')   
    btc_volume = preprocess_volume(btc,'month')
    eth_data_m = preprocess_price(eth,'month')
    eth_data_y = preprocess_price(eth, 'year')

    binancem = preprocess_sentiment(binancetotal,'month')
    coinbasem = preprocess_sentiment(coinbasetotal,'month')
    kucoinm = preprocess_sentiment(kucointotal,'month')

    binancey = preprocess_sentiment(binancetotal,'year')
    coinbasey = preprocess_sentiment(coinbasetotal,'year')
    kucoiny = preprocess_sentiment(kucointotal,'year')

    br = preprocess_sentiment(brTotal,'year')
    cn  = preprocess_sentiment(cnTotal,'year')
    ng = preprocess_sentiment(ngTotal,'year')
    ua = preprocess_sentiment(uaTotal,'year')
    us = preprocess_sentiment(usTotal,'year')
    
    br_17 = preprocess_2017(brTotal)
    cn_17 = preprocess_2017(cnTotal)
    ng_17 = preprocess_2017(ngTotal)
    ua_17 = preprocess_2017(uaTotal)
    us_17 = preprocess_2017(usTotal)

    countries_17 = {
        'BR': br_17,
        'CN': cn_17,
        'NG': ng_17,
        'UA': ua_17,
        'US': us_17
    }
    # Gather the data
    exchange_data_grouped = {
        'Binance': binancey,
        'Coinbase': coinbasey,
        'Kucoin': kucoiny
    }
    exchange_data_m = {
        'Binance': binancem,
        'Coinbase': coinbasem,
        'Kucoin': kucoinm
    }
    exchanges = {
        'Binance':  preprocess_2017(binancetotal),
        'Coinbase':  preprocess_2017(coinbasetotal),
        'Kucoin':  preprocess_2017(kucointotal)
    }
    country_totals = {
        'BR': br,
        'CN': cn,
        'NG': ng,
        'UA': ua,
        'US': us
    }

    df_countries = [brTotal, cnTotal, ngTotal, uaTotal, usTotal]
    df_exchanges = [binancetotal, coinbasetotal, kucointotal]
    labels_countries = ['BR', 'CN', 'NG', 'UA', 'US']
    labels_exchanges = ['Binance', 'Coinbase', 'Kucoin']

    #time series decomposition binance
    #preprocess_and_decompose(binancetotal, 'Binance Sentiment Decomposition')

    # time series decomposition coinbase
    #preprocess_and_decompose(coinbasetotal, 'Coinbase Sentiment Decomposition')

    # time series decomposition kucoin
    #preprocess_and_decompose(kucointotal, 'Kucoin Sentiment Decomposition')

    # boxplot sentiment 
    #plot_boxplot_distribution(exchange_data_m)

    #Plot sentiment counts for all exchanges
    
    plot_sentiment_frequency(exchanges)

    plot_sentiment_frequency(countries_17)

    # Plot sentiment trends over time for exchanges
    plot_sentiment_trends(exchange_data_grouped, 'Average Sentiment by Exchange')
    # Plot sentiment trends over time for countries
    plot_sentiment_trends(country_totals, 'Average Sentiment by Country')

    # Plot sentiment trends by month and correlate with btc price
    plot_sentiment_btc_year_month(exchange_data_m, btc_data_m)
    # plot sentiment trend by year and btc price yearly
    plot_sentiment_btc_year(exchange_data_grouped, btc_data_y)
    # Plot trends
    plot_sentiment_and_btc_volume(exchange_data_m, btc_volume)
    #ETH 
    plot_sentiment_eth_year_month(exchange_data_m, eth_data_m)

    # Plot sentiment trends over time for exchanges with heatmap
    #plot_heatmap(exchange_data_grouped, 'e')
    # Plot sentiment trends over time for countries with heatmap
    #plot_heatmap(country_totals, 'c')     

    # Plot hit rate
    plot_hitrate_sentiment(binancetotal, coinbasetotal, kucointotal)

    #plot violin exchanges
    #plot_violin_distribution(df_exchanges, labels_exchanges)

    #plot violin countries
    #plot_violin_distribution(df_countries, labels_countries)


# Execution starts here
if __name__ == '__main__':
    main()



