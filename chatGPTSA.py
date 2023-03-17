import openai
import json
import pandas as pd

# Set up your OpenAI API key
openai.api_key = "YOUR_API_KEY"

# Read the CSV file containing the text reviews
reviews_df = pd.read_csv("reviews.csv")

# Define the parameters for the GPT-3 API call
model_engine = "text-davinci-002"
prompt_template = "sentiment analysis: {}"

# Iterate over the text reviews in the DataFrame and call the GPT-3 API for each one
for review in reviews_df["Body"]:
    prompt = prompt_template.format(review)
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1,
        n=1,
        stop=None,
        temperature=0.5,
    )
    result = json.loads(response.choices[0].text)
    sentiment = result["label"]
    confidence = result["confidence"]
    print(f"The sentiment of '{review}' is {sentiment} with a confidence score of {confidence}.")
