import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import matplotlib.pyplot as plt

# Load the data
dfB = pd.read_csv('playstore_binance_reviews_BR.csv')
dfC = pd.read_csv('playstore_coinbase_reviews_BR.csv')
dfK = pd.read_csv('playstore_kucoin_reviews_BR.csv')

df = dfC.dropna(subset=['reviews'])
df = df[df['reviews'].apply(lambda x: isinstance(x, str))]
if 'title' in df.columns:
    df = df.drop(columns =['isEdited','title','userName'])
if 'developerResponse' in df.columns:
    df = df.drop(columns=['developerResponse'])

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Create a sentiment analysis pipeline
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Apply sentiment analysis to the entire reviews column
df['sentiment'] = df['reviews'].apply(lambda x: nlp(x)[0]['label'])

# Map the numerical ratings to descriptive labels
sentiment_map = {
    '1 star': 'very_bad',
    '2 stars': 'bad',
    '3 stars': 'neutral',
    '4 stars': 'good',
    '5 stars': 'very_good'

}
df['sentiment'] = df['sentiment'].map(sentiment_map)

# Save the results back to a new CSV
df.to_csv('PS_Coinbase_BR_bert.csv', index=False)
