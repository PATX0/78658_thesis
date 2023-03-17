import transformers
import pandas as pd

# Set up the Transformers API client
client = transformers.pipeline(
    "sentiment-analysis",
    model="bert-base-cased",
)

# Read the CSV file containing the text reviews
reviews_df = pd.read_csv("reviews.csv")

# Iterate over the text reviews in the DataFrame and call the BERT model for each one
for review in reviews_df["Body"]:
    result = client(review)[0]
    sentiment = result["label"]
    score = result["score"]
    print(f"The sentiment of '{review}' is {sentiment} with a score of {score}.")
