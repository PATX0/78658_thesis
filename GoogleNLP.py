import csv
from google.cloud import language_v1
from google.cloud.language_v1 import enums

# Set up the Google Cloud Natural Language API client
client = language_v1.LanguageServiceClient()

# Open the CSV file and read the Body column
with open('binance_reviews.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        text = row['Body']

        # Perform sentiment analysis on the text
        document = language_v1.Document(content=text, type_=enums.Document.Type.PLAIN_TEXT)
        sentiment = client.analyze_sentiment(document=document).document_sentiment

        # Print the sentiment score
        print("Score: {}".format(sentiment.score))
        print("Magnitude: {}".format(sentiment.magnitude))
