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
df_cleaned.to_csv('AS_Binance_BR.csv', index=False)
#df_cleaned.to_csv('AS_Coinbase_BR.csv', index=False)
#df_cleaned.to_csv('AS_Kucoin_BR.csv', index=False)



# Plotting the bar chart
colors = ['yellowgreen', 'lightgreen', 'yellow', 'orange', 'red']
plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color=colors)
plt.title("Sentiment Distribution - Bar Chart")
plt.ylabel('Number of Reviews')
plt.xlabel('Sentiment')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

"""
#picks a sample of 3 comments of each category to print
def display_examples():
    examples = {}

    for cat in sentiment_counts:
        subset = df[df['sentiment'] == cat]
        examples[cat] = subset['reviews'].sample(min(3, len(subset))).tolist()

    for cat, ex in examples.items():
        print(f"\nExamples for {cat}:")
        for e in ex:
            print(f"- {e}")
"""

"""
# Predefined lists of positive and negative words (a subset for demonstration purposes)
positive_words = set([
    "good", "great", "excellent", "awesome", "amazing", "fantastic", "positive", "love", "like", "best", 
    "wonderful", "easy", "helpful", "perfect", "happy", "nice", "well", "super", "smooth", "recommend"
])

negative_words = set([
    "bad", "worst", "poor", "terrible", "awful", "negative", "hate", "dislike", "horrible", "difficult",
    "problem", "bug", "issue", "slow", "error", "unreliable", "hard", "complicated", "miss", "fail"
])

# Function to compute sentiment based on word counts
def compute_sentiment(text):
    text = text.lower()
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    if positive_count > negative_count:
        return 'positive'
    elif positive_count < negative_count:
        return 'negative'
    else:
        return 'neutral'


def display_vis():
# Apply the sentiment computation function to each review
    df_cleaned['simple_sentiment'] = df_cleaned['reviews'].apply(compute_sentiment)

    # Get the distribution of sentiments using this method
    simple_sentiment_distribution = df_cleaned['simple_sentiment'].value_counts(normalize=True)


    # Compute polarity scores for each review using TextBlob
    df_cleaned['polarity'] = df_cleaned['reviews'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Summary statistics of polarity scores
    polarity_summary = df_cleaned['polarity'].describe()

    #VISUALIZATIONS
    # Initialize the figure for Binance US reviews visualization
    plt.figure(figsize=(15, 10))

    # Plotting the three visualizations for Binance reviews
    plt.subplot(3, 1, 1)
    sns.histplot(df_cleaned['polarity'], bins=30, kde=False, color='skyblue')
    plt.title('Histogram of Sentiment Polarity Scores (Playstore Binance US Reviews)')
    plt.xlabel('Polarity')
    plt.ylabel('Number of Reviews')

    plt.subplot(3, 1, 2)
    sns.boxplot(x=df_cleaned['polarity'], color='lightgreen')
    plt.title('Box Plot of Sentiment Polarity Scores (Playstore Binance US Reviews)')
    plt.xlabel('Polarity')

    plt.subplot(3, 1, 3)
    sns.kdeplot(df_cleaned['polarity'], shade=True, color='salmon')
    plt.title('Density Plot of Sentiment Polarity Scores (Playstore Binance US Reviews)')
    plt.xlabel('Polarity')

    # Adjust the layout
    plt.tight_layout()
    plt.show()
"""

"""VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically designed for sentiments expressed in social media. 
VADER uses a combination of a list of lexical features (words) related to sentiment, and a set of rules to handle common scenarios encountered in text data, such as intensification and punctuation emphasis.

Here are some key characteristics and features of VADER:

Lexicon-based Approach: VADER has a predefined list of lexical features (words) which have been manually labeled by humans for their polarity. The scores range from -4 (most negative) to +4 (most positive).

Handles Context: One of VADER's strengths is its ability to understand the context. For example, it can differentiate between positive, negative, and neutral sentiments in sentences like:

"The food here is good."
"The food here is not good."
"I think the food here is good."
Emphasis through Punctuation: VADER assigns sentiment scores in a way that takes into account punctuation emphasis. For example, "The food here is good!" would get a higher positive score than "The food here is good."

Emphasis through Capitalization: Words in capital letters have more emphasis. For instance, "The food here is GOOD!" would have a stronger positive sentiment than the aforementioned examples.

Handling of Conjunctions: VADER pays attention to conjunctions (e.g., "but") to shift sentiment polarity. For instance, "The food here is great, but the service is horrible."
 VADER would detect the positive sentiment in the first part of the sentence and the negative sentiment in the second part.

Emojis and Slangs: VADER recognizes emojis, slangs, and acronyms as sentiment-laden lexical features. For example, it understands that both ":)" and "lol" are positive.

Efficiency: VADER is designed to be fast and requires minimal computational resources.

Output Scores: VADER doesn't just give a binary positive/negative output. Instead, it provides a compound score that represents the overall sentiment of a statement, ranging from -1 (most negative) to 1 (most positive). 
It also provides scores for the positive, negative, and neutral sentiments separately.

Domain Agnostic: While VADER was developed with social media texts in mind, it's often found to be effective in a variety of other domains as well.

In summary, VADER is a powerful and versatile sentiment analysis tool, particularly suitable for analyzing short texts and those from social media sources. 
It's robust against common challenges in sentiment analysis such as context and emphasis. """