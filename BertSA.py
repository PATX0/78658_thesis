import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the BERT model and tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load the comments from a CSV file
comments_df = pd.read_csv("binance_reviews.csv")
comments = comments_df["Body"].tolist()

# Tokenize the comments
inputs = tokenizer(comments, padding=True, truncation=True, return_tensors="pt")

# Perform inference on the comments
outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
predictions = outputs.logits.argmax(dim=-1).tolist()

# Print the sentiment predictions for each comment
for i, comment in enumerate(comments):
    print(f"{i + 1}. {comment}\n   Sentiment: {predictions[i]}")

