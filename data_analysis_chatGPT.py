import openai
import os
import pandas as pd
import json

# Set up your OpenAI API key
#openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = 'sk-proj-owKzt1K1ewex982jsPaDT3BlbkFJPRI01OEZzXJZCo0lLcm0'

# Read the CSV file containing the text reviews
reviews_df = pd.read_csv('csvs/appstore_presentiment/appstore_kucoin_reviews_NG.csv')

# Define the parameters for the GPT-3 API call
model_engine = "text-davinci-002"
prompt_template = "Do sentiment analysis: {}"

# Initialize a list to store the sentiment results
sentiments = []

# Iterate over the text reviews in the DataFrame and call the GPT-3 API for each one
for review in reviews_df["reviews"]:
    prompt = prompt_template.format(review)
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=12,
        n=1,
        stop=None,
        temperature=0.5,
    )
    print(json.dumps(response, indent=4))
    # Parse the response to extract sentiment
    sentiment = response['choices'][0]['text'].strip()
    sentiments.append(sentiment)
    print(f"The sentiment of '{review}' is {sentiment}")

# Add the sentiment data as a new column to the DataFrame
reviews_df['sentiment'] = sentiments

# Save the updated DataFrame to a new CSV file
reviews_df.to_csv("coinbase_reviews_chatGPT.csv", index=False)
