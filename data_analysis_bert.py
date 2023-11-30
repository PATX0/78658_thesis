import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import matplotlib.pyplot as plt

# Load the data
#Playstore
dfBP = pd.read_csv('csvs/playstore/playstore_binance_reviews_US.csv')
dfCP = pd.read_csv('csvs/playstore/playstore_coinbase_reviews_US.csv')
dfKP = pd.read_csv('csvs/playstore/playstore_kucoin_reviews_US.csv')
#Appstore
dfBA = pd.read_csv('csvs/appstore/appstore_binance_reviews_NG.csv')
dfCA = pd.read_csv('csvs/appstore/appstore_coinbase_reviews_NG.csv')
dfKA = pd.read_csv('csvs/appstore/appstore_kucoin_reviews_NG.csv')

#df = dfCP.dropna(subset=['reviews'])
#df = dfKP.dropna(subset=['reviews'])
#df = dfBA.dropna(subset=['reviews'])
#df = dfCA.dropna(subset=['reviews'])
#df = dfKA.dropna(subset=['reviews'])

df = dfKP.dropna(subset=['reviews'])
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

# Map the numerical ratings to descriptive labels ( 'X stars' is the format of the output given by the model)
sentiment_map = {
    '1 star': 'very_bad',
    '2 stars': 'bad',
    '3 stars': 'neutral',
    '4 stars': 'good',
    '5 stars': 'very_good'

}
df['sentiment'] = df['sentiment'].map(sentiment_map)

# Save the results back to a new CSV
#df.to_csv('PS_Binance_CN_bert.csv', index=False)
#df.to_csv('PS_Coinbase_CN_bert.csv', index=False)
df.to_csv('PS_Kucoin_US_bert.csv', index=False)