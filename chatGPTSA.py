import openai
import json
import pandas as pd

# Set up your OpenAI API key

openai.api_key = os.environ.get("OPENAI_API_KEY")

# Read the CSV file containing the text reviews
reviews_df = pd.read_csv("coinbase_reviews.csv")

# Define the parameters for the GPT-3 API call
model_engine = "text-davinci-002"
prompt_template = "sentiment analysis: {}"

# Iterate over the text reviews in the DataFrame and call the GPT-3 API for each one
for review in reviews_df["Body"]:
    prompt = prompt_template.format(review)
    response = openai.Completion.create(
        model=model_engine,
        prompt=prompt,
        max_tokens=12,
        n=1,
        stop=None,
        temperature=0.5,
    )
    response_json = json.dumps(response)
    result = json.loads(response_json)
    sentiment = result["choices"][0]["text"].strip()
    print(f"The sentiment of '{review}' is {sentiment} ") #with a confidence score of {confidence}.
