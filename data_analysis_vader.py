import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

dfB = pd.read_csv('appstore_binance_reviews_BR.csv')
dfC = pd.read_csv('appstore_coinbase_reviews_BR.csv')
dfK = pd.read_csv('appstore_kucoin_reviews_BR.csv')

# Drop rows where reviews are missing or not strings
df_cleaned = dfB.dropna(subset=['reviews'])
df_cleaned = df_cleaned[df_cleaned['reviews'].apply(lambda x: isinstance(x, str))]
df_cleaned = df_cleaned.drop(columns=['isEdited','title','userName','developerResponse'])

# Given sentiment distribution
sentiment_counts = {
    "5": 0,
    "4": 0,
    "3": 0,
    "2": 0,
    "1": 0
}

# Initialize the VADER analyzer
analyzer = SentimentIntensityAnalyzer()

# Calculate the compound sentiment score using VADER
df_cleaned['compound'] = df_cleaned['reviews'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

# Categorize sentiments based on the compound score
def categorize_sentiment(compound):
    if compound >= 0.5:
        sentiment_counts['5'] += 1
        return "5"
    elif compound >= 0.1:
        sentiment_counts['4'] += 1
        return "4"
    elif compound > -0.1:
        sentiment_counts['3'] += 1
        return "3"
    elif compound > -0.5:
        sentiment_counts['2'] += 1
        return "2"
    else:
        sentiment_counts['1'] += 1
        return "1"

df_cleaned['sentiment'] = df_cleaned['compound'].apply(categorize_sentiment)

# Display sentiment distribution
sentiment_distribution = df_cleaned['sentiment'].value_counts()
print(sentiment_distribution)

# Save the modified DataFrame back to CSV
df_cleaned.to_csv('binance_reviews_BR_VADER.csv', index=False)




# Plotting the bar chart
colors = ['yellowgreen', 'lightgreen', 'yellow', 'orange', 'red']
plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color=colors)
plt.title("Sentiment Distribution - Bar Chart")
plt.ylabel('Number of Reviews')
plt.xlabel('Sentiment')
plt.grid(axis='y')
plt.tight_layout()
plt.show()