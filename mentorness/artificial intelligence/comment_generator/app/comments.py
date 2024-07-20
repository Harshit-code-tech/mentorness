from transformers import pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

def analyze_sentiment(content):
    transformer_sentiment_pipeline = pipeline("sentiment-analysis",
                                              model="distilbert-base-uncased-finetuned-sst-2-english")
    transformer_sentiment = transformer_sentiment_pipeline(content)[0]

    blob = TextBlob(content)
    polarity = blob.sentiment.polarity
    nltk_sentiment = 'positive' if polarity > 0 else 'negative' if polarity < 0 else 'neutral'

    analyzer = SentimentIntensityAnalyzer()
    vader_sentiment = analyzer.polarity_scores(content)['compound']

    topic = summarize_to_topic(content)

    tone = detect_tone(polarity, nltk_sentiment, transformer_sentiment['label'], vader_sentiment)

    return polarity, nltk_sentiment, transformer_sentiment['label'], vader_sentiment, tone, topic


def detect_tone(polarity, nltk_sentiment, transformer_sentiment, vader_sentiment):
    if transformer_sentiment == 'POSITIVE' and polarity > 0 and vader_sentiment > 0.5:
        return 'positive'
    elif transformer_sentiment == 'NEGATIVE' and polarity < 0 and vader_sentiment < -0.5:
        return 'negative'
    else:
        return 'neutral'

def summarize_to_topic(text):
    language = detect(text)
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    stemmer = Stemmer(language)
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)

    summary = summarizer(parser.document, 1)
    return " ".join(str(sentence) for sentence in summary).split()[0]


import random
from .analysis import analyze_sentiment


def generate_friendly_comment(content, topic):
    friendly_templates = [
        "Great read! {topic}... Keep it up!",
        "I really enjoyed this! {topic}... Fantastic work!",
        "This is so well-written! {topic}... Excellent job!"
    ]
    return random.choice(friendly_templates).format(topic=topic)


def generate_funny_comment(content, topic):
    funny_templates = [
        "LOL! {topic}... Hilarious!",
        "This cracked me up! {topic}... So funny!",
        "I couldn't stop laughing at {topic}... Great sense of humor!"
    ]
    return random.choice(funny_templates).format(topic=topic)


def generate_congratulating_comment(content, topic):
    congratulating_templates = [
        "Congratulations on this amazing piece! {topic}...",
        "Well done! {topic}... Keep up the great work!",
        "Bravo! {topic}... Impressive achievement!"
    ]
    return random.choice(congratulating_templates).format(topic=topic)


def generate_questioning_comment(content, topic):
    questioning_templates = [
        "Interesting point! Can you elaborate on {topic}?",
        "Could you explain more about {topic}...?",
        "I'd love to know more about {topic}... Could you clarify?"
    ]
    return random.choice(questioning_templates).format(topic=topic)


def generate_disagreement_comment(content, topic):
    disagreement_templates = [
        "I don't quite agree with {topic}... Here's why...",
        "Interesting perspective, but I see it differently: {topic}...",
        "I respect your opinion, but {topic}... doesn't resonate with me."
    ]
    return random.choice(disagreement_templates).format(topic=topic)


def generate_comments(content):
    try:
        polarity, nltk_sentiment, transformer_sentiment, vader_sentiment, tone, topic = analyze_sentiment(content)

        friendly = generate_friendly_comment(content, topic)
        funny = generate_funny_comment(content, topic)
        congratulating = generate_congratulating_comment(content, topic)
        questioning = generate_questioning_comment(content, topic)
        disagreement = generate_disagreement_comment(content, topic)
        return {
            "friendly": friendly,
            "funny": funny,
            "congratulating": congratulating,
            "questioning": questioning,
            "disagreement": disagreement,
            "tone": tone
        }
    except Exception as e:
        print(f"Error generating comments: {e}")
        return {
            "friendly": "",
            "funny": "",
            "congratulating": "",
            "questioning": "",
            "disagreement": "",
            "tone": "error"
        }



