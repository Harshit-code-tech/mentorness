from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

def analyze_sentiment(text):
    # TextBlob for basic sentiment
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    # NLTK for more detailed sentiment
    sia = SentimentIntensityAnalyzer()
    nltk_sentiment = sia.polarity_scores(text)

    # transformers for advanced sentiment
    sentiment_pipeline = pipeline("sentiment-analysis")
    transformer_sentiment = sentiment_pipeline(text)

    return polarity, nltk_sentiment, transformer_sentiment

def detect_tone(polarity, nltk_sentiment, transformer_sentiment):
    # Simplified tone detection logic
    if polarity > 0:
        tone = "positive"
    elif polarity < 0:
        tone = "negative"
    else:
        tone = "neutral"

    return tone
