from transformers import pipeline
from textblob import TextBlob

def analyze_sentiment(content):
    transformer_sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    transformer_sentiment = transformer_sentiment_pipeline(content)[0]

    blob = TextBlob(content)
    polarity = blob.sentiment.polarity
    nltk_sentiment = 'positive' if polarity > 0 else 'negative' if polarity < 0 else 'neutral'

    return polarity, nltk_sentiment, transformer_sentiment['label']

def detect_tone(polarity, nltk_sentiment, transformer_sentiment):
    # Example tone detection logic
    if transformer_sentiment == 'POSITIVE':
        return 'positive'
    elif transformer_sentiment == 'NEGATIVE':
        return 'negative'
    else:
        return 'neutral'
