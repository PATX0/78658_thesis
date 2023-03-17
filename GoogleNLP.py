from google.cloud import language_v1

client = language_v1.LanguageServiceClient()

text = "I love using the Google Cloud Natural Language API!"

document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)

sentiment = client.analyze_sentiment(request={'document': document}).document_sentiment

print(f"Sentiment score: {sentiment.score}")
print(f"Sentiment magnitude: {sentiment.magnitude}")