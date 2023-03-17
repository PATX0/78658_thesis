import csv
import os
import google.cloud.language_v1 as lang
#from google.oauth2.credentials import Credentials

# Set up the Google Cloud Natural Language API client
#creds = Credentials.from_authorized_user_file("client_secret.json")
#google_credential_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
#client = language_v1.LanguageServiceClient(credentials=creds)
client = lang.LanguageServiceClient()

# Define the sentiment categories
SENTIMENT_CATEGORIES = {
    "very negative": (-float("inf"), -0.5),
    "negative": (-0.5, -0.1),
    "neutral": (-0.1, 0.1),
    "positive": (0.1, 0.5),
    "very positive": (0.5, float("inf"))
}
# Open the CSV file and read the Body column
with open('binance_reviews.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        text = row['Body']

        # Perform sentiment analysis on the text
        document = lang.Document(content=text, type=lang.Document.Type.PLAIN_TEXT)
        sentiment = client.analyze_sentiment(document=document).document_sentiment
        print("The comment: {}".format(text))
        for category, (lower, upper) in SENTIMENT_CATEGORIES.items():
            if lower <= sentiment.score <= upper:
                print(f"Sentiment category: {category}")
                break
        # Print the sentiment score
        
        #print("Score: {}".format(sentiment.score))
        print("Magnitude: {}".format(sentiment.magnitude))
