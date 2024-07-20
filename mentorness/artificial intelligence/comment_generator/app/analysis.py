# analysis.py
from transformers import pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def analyze_sentiment(content):
    transformer_sentiment_pipeline = pipeline("sentiment-analysis",
                                              model="distilbert-base-uncased-finetuned-sst-2-english")
    transformer_sentiment = transformer_sentiment_pipeline(content)[0]

    blob = TextBlob(content)
    polarity = blob.sentiment.polarity
    nltk_sentiment = 'positive' if polarity > 0 else 'negative' if polarity < 0 else 'neutral'

    analyzer = SentimentIntensityAnalyzer()
    vader_sentiment = analyzer.polarity_scores(content)['compound']

    tone = detect_tone(polarity, nltk_sentiment, transformer_sentiment['label'], vader_sentiment)

    return polarity, nltk_sentiment, transformer_sentiment['label'], vader_sentiment, tone


def detect_tone(polarity, nltk_sentiment, transformer_sentiment, vader_sentiment):
    if transformer_sentiment == 'POSITIVE' and polarity > 0 and vader_sentiment > 0.5:
        return 'positive'
    elif transformer_sentiment == 'NEGATIVE' and polarity < 0 and vader_sentiment < -0.5:
        return 'negative'
    else:
        return 'neutral'
