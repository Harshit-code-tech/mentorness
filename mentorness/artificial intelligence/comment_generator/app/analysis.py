from transformers import pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def analyze_sentiment(content):
    transformer_sentiment_pipeline = pipeline("sentiment-analysis",
                                              model="distilbert-base-uncased-finetuned-sst-2-english")
    transformer_sentiment = transformer_sentiment_pipeline(content)[0]

    blob = TextBlob(content)
    polarity = blob.sentiment.polarity

    analyzer = SentimentIntensityAnalyzer()
    vader_sentiment = analyzer.polarity_scores(content)['compound']

    return polarity, transformer_sentiment['label'], vader_sentiment


def detect_tone(polarity, transformer_sentiment, vader_sentiment):
    if transformer_sentiment == 'POSITIVE' and polarity > 0 and vader_sentiment > 0.5:
        return 'positive'
    elif transformer_sentiment == 'NEGATIVE' and polarity < 0 and vader_sentiment < -0.5:
        return 'negative'
    else:
        return 'neutral'
