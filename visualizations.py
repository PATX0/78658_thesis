import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

""" sentiment_counts = {
    "very_good": 0,
    "good": 0,
    "neutral": 0,
    "bad": 0,
    "very_bad": 0
} """

dfA_BR_binance = pd.read_csv('AS_Binance_BR_bert.csv')
dfP_BR_binance = pd.read_csv('PS_Binance_BR_bert.csv')
dfA_BR_coinbase = pd.read_csv('AS_Coinbase_BR_bert.csv')
dfP_BR_coinbase = pd.read_csv('PS_Coinbase_BR_bert.csv')
dfA_BR_kucoin = pd.read_csv('AS_Kucoin_BR_bert.csv')
dfP_BR_kucoin = pd.read_csv('PS_Kucoin_BR_bert.csv')

# Calculate the count of each sentiment label in the 'sentiment' column
sentiment_counts = dfA_BR_binance['sentiment'].value_counts()

# Print the count of each sentiment
print(sentiment_counts)

""" if df['sentiment'] == 'very_bad':
    sentiment_counts['very_bad'] += 1
if df['sentiment'] == 'bad':
    sentiment_counts['bad'] += 1
if df['sentiment'] == 'very_good':
    sentiment_counts['very_good'] += 1
if df['sentiment'] == 'good':
    sentiment_counts['good'] += 1
if df['sentiment'] == 'neutral':
    sentiment_counts['neutral'] += 1 """

 # Plotting the bar chart
colors = ['yellowgreen', 'lightgreen', 'yellow', 'orange', 'red']
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