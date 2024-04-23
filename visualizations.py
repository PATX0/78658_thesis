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

def main():   
    #####
    # compare_score_vs_rating(dfA_US_binance)
    # #####
    # display_sentiment_counts(dfP_US_binance)
    # #######
    # crosscheck_btc(dfP_US_binance)
    # ##################################
    # plot_sentiment_distribution(dfP_US_binance, "Binance")
    # plot_sentiment_over_time(dfP_US_binance, "Binance")
    # plot_sentiment_distribution(dfP_US_coinbase, "Coinbase")
    # plot_sentiment_over_time(dfP_US_coinbase, "Coinbase")
    # plot_sentiment_distribution(dfP_US_kucoin, "Kucoin")
    # plot_sentiment_over_time(dfP_US_kucoin, "Kucoin")
    # ##########density chart - sentiment distro
    # plot_average_monthly_sentiment()
    # #same with seaborn
    # plt.figure(figsize=(15, 5))
    # plot_density_sentiment_distribution(dfP_US_binance)
    # plot_density_sentiment_distribution(dfP_US_coinbase)
    # plot_density_sentiment_distribution(dfP_US_kucoin)
    # plt.legend()
    # plt.show()
    # ########
    # visualize_sentiment_exchanges(dfP_US_binance, dfP_US_coinbase, dfP_US_kucoin)
    ##############
    avg_sentiment_and_total_per_exchange(dfP_US_binance,dfP_US_coinbase,dfP_US_kucoin)

###########
    
def display_sentiment_counts(csv):
    # Calculate the count of each sentiment label in the 'sentiment' column
    sentiment_counts = csv['sentiment'].value_counts()
    #sentiment_mapping = {'very_bad': 1, 'bad': 2, 'neutral': 3, 'good': 4, 'very_good': 5}
    #df_sentiment['sentiment'] = df_sentiment['sentiment'].map(sentiment_mapping)

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
    df_sentiment_monthly = df_sentiment.groupby('year_month')['sentiment'].mean().reset_index()

    # Load the BTC/USD data
    dfBTC = pd.read_csv('csvs/BTC_daily_pricevol.csv')
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
    
    # Group data by rating and calculate average sentiment score for each rating
    grouped_by_rating = df.groupby('rating')['sentiment'].mean()

    # Create a scatter plot with a trend line
    plt.figure(figsize=(10, 6))
    sns.regplot(x=grouped_by_rating.index, y=grouped_by_rating.values, scatter_kws={'s': 100}, color='darkblue')
    plt.title('Average Sentiment Score by User Rating')
    plt.xlabel('User Rating')
    plt.ylabel('Average Sentiment Score')
    plt.grid(True)
    plt.show()


####################################################################################################################################################################################################################################

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

####################################################################################################################################################################################################################################

#Average Monthly Sentiment: The first plot shows the average monthly sentiment for Binance (blue), Coinbase (green), and Kucoin (red). This graph provides insight into how user sentiment fluctuates over time on each platform. 
#Trends, seasonal patterns, or any significant changes in sentiment over the months can be observed.
#Sentiment Score Distribution Across Platforms: The second plot is a set of box plots comparing the distribution of sentiment scores across Binance, Coinbase, and Kucoin. 
#This visualization helps to understand the range, median, and variability of sentiment scores on each platform. 
#It's useful for identifying differences in user sentiment across these platforms, such as which platform tends to have more positive or negative reviews.
#These analyses give a clearer picture of the sentiment dynamics and can be used to draw insights about user experiences and perceptions on each cryptocurrency trading platform
####################################################################################################################################################################################################################################


def plot_density_sentiment_distribution(df):
    """
    Plot a density plot for sentiment distribution of a given dataset.
    """
    sns.kdeplot(df['sentiment'], label=df.name, fill=True)
    plt.xlabel('Sentiment Score')
    plt.ylabel('Density')
    plt.title(f'Sentiment Score Density Distribution for {df.name}')
    

#Here's the density plot visualizing the sentiment score distributions for Binance, Coinbase, and Kucoin:

#Each curve represents the density distribution of sentiment scores for one of the platforms.
#The x-axis shows the sentiment scores, while the y-axis represents the density (probability distribution).
#The area under each curve indicates the frequency of sentiment scores within that range.
#This visualization provides a smooth and continuous view of how sentiment scores are spread out for each platform, allowing you to compare the distributions directly. 
#It's particularly useful for identifying the most common sentiment scores and seeing how tightly or widely they are distributed on each platform
#########################################################################################################################################################################################################################

#####DONE WITH SEABORN

# Function to prepare data for average monthly sentiment analysis
def prepare_monthly_sentiment_data(df, platform_name):
    # 'timestamp' needs to be in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Set 'timestamp' as the index for search purposes
    df = df.set_index('timestamp')
    # Resample and calculate the mean of the 'sentiment' column only
    df_monthly = df[['sentiment']].resample('M').mean().reset_index()
    df_monthly['Platform'] = platform_name
    return df_monthly

def visualize_sentiment_exchanges(binance_df, coinbase_df, kucoin_df):
    # Color assignments
    colors = {
        'Binance': '#FFFF00',  #  Yellow 
        'Coinbase': '#002395',  # Blue
        'Kucoin': '#00AB84',  # Green
    }

    # Preparing monthly sentiment data for each platform
    binance_monthly = prepare_monthly_sentiment_data(binance_df, 'Binance')
    coinbase_monthly = prepare_monthly_sentiment_data(coinbase_df, 'Coinbase')
    kucoin_monthly = prepare_monthly_sentiment_data(kucoin_df, 'Kucoin')

    # Combining data
    combined_monthly = pd.concat([binance_monthly, coinbase_monthly, kucoin_monthly])

    # Average Monthly Sentiment Line Plot
    plt.figure(figsize=(15, 5))
    sns.lineplot(data=combined_monthly, x='timestamp', y='sentiment', hue='Platform', 
                 palette=colors, style='Platform')
    plt.title('Average Monthly Sentiment Analysis')
    plt.xlabel('Month')
    plt.ylabel('Average Sentiment')
    plt.xticks(rotation=45)
    plt.legend(title='Platform')
    plt.show()

    # Sentiment Score Density Plot
    plt.figure(figsize=(15, 5))
    sns.kdeplot(binance_df['sentiment'], color='#FE5000', label='Binance', fill=True)
    sns.kdeplot(coinbase_df['sentiment'], color='#002395', label='Coinbase', fill=True)
    sns.kdeplot(kucoin_df['sentiment'], color='#00AB84', label='Kucoin', fill=True)
    plt.title('Density Plot of Sentiment Scores Across Platforms')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Density')
    plt.legend(title='Platform')
    plt.show()
    
# Function to prepare data, grouping by year and aggregating the total count and the average sentiment score of the given dataset
def prepare_data(df, exchange_name):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Filter out dates before January 1, 2017
    df = df[df['timestamp'] >= '2017-01-01']
    df['year'] = df['timestamp'].dt.year
    return df.groupby('year').agg(total_count=('sentiment', 'size'),
                                average_sentiment=('sentiment', 'mean')).reset_index().assign(exchange=exchange_name)

def avg_sentiment_and_total_per_exchange(binance, coinbase, kucoin):
    # Prepare data
    binance_annual = prepare_data(binance, 'Binance')
    print('Binance complete.')
    coinbase_annual = prepare_data(coinbase, 'Coinbase')
    print('Coinbase complete.')
    kucoin_annual = prepare_data(kucoin, 'Kucoin')
    print('Kucoin complete.')

    # Combine all data
    combined_annual = pd.concat([binance_annual, coinbase_annual, kucoin_annual])

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Colors for different exchanges
    colors = {'Binance': 'orange', 'Coinbase': 'blue', 'Kucoin': 'green'}

    # Plot total count of reviews by year for each exchange
    for name, group in combined_annual.groupby('exchange'):
        ax1.bar(group['year'] + 0.00*list(colors.keys()).index(name) - 0.20, group['total_count'], color=colors[name], width=0.20, label=f'{name} Total Count')

    # Create a twin axis for the average sentiment
    ax2 = ax1.twinx()
    for name, group in combined_annual.groupby('exchange'):
        ax2.plot(group['year'], group['average_sentiment'], color=colors[name], marker='o', linestyle='-', label=f'{name} Avg Sentiment')
        # Annotate each point with its exact value
        for i, txt in enumerate(group['average_sentiment']):
            ax2.annotate(f'{txt:.2f}', (group['year'].iloc[i], group['average_sentiment'].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # Labels, title and legend
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Total Count of Reviews', color='blue')
    ax2.set_ylabel('Average Sentiment Score', color='black')
    plt.title('Total Count of Reviews and Average Sentiment by Year for Each Exchange')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
    plt.show()

# Execution starts here
if __name__ == '__main__':
    main()