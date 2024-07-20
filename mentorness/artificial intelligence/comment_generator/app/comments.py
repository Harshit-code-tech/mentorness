import random
from .analysis import analyze_sentiment
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from langdetect import detect

def summarize_text(text, sentence_count=1):
    language = detect(text)
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    stemmer = Stemmer(language)
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)

    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)

def generate_friendly_comment(content):
    friendly_templates = [
        "Great read! {snippet}... Keep it up!",
        "I really enjoyed this! {snippet}... Fantastic work!",
        "This is so well-written! {snippet}... Excellent job!"
    ]
    snippet = summarize_text(content)
    return random.choice(friendly_templates).format(snippet=snippet)

def generate_funny_comment(content):
    funny_templates = [
        "LOL! {snippet}... Hilarious!",
        "This cracked me up! {snippet}... So funny!",
        "I couldn't stop laughing at {snippet}... Great sense of humor!"
    ]
    snippet = summarize_text(content)
    return random.choice(funny_templates).format(snippet=snippet)

def generate_congratulating_comment(content):
    congratulating_templates = [
        "Congratulations on this amazing piece! {snippet}...",
        "Well done! {snippet}... Keep up the great work!",
        "Bravo! {snippet}... Impressive achievement!"
    ]
    snippet = summarize_text(content)
    return random.choice(congratulating_templates).format(snippet=snippet)

def generate_questioning_comment(content):
    questioning_templates = [
        "Interesting point! Can you elaborate on {snippet}?",
        "Could you explain more about {snippet}...?",
        "I'd love to know more about {snippet}... Could you clarify?"
    ]
    snippet = summarize_text(content)
    return random.choice(questioning_templates).format(snippet=snippet)

def generate_disagreement_comment(content):
    disagreement_templates = [
        "I don't quite agree with {snippet}... Here's why...",
        "Interesting perspective, but I see it differently: {snippet}...",
        "I respect your opinion, but {snippet}... doesn't resonate with me."
    ]
    snippet = summarize_text(content)
    return random.choice(disagreement_templates).format(snippet=snippet)

def generate_comments(content):
    try:
        polarity, nltk_sentiment, transformer_sentiment, vader_sentiment, tone = analyze_sentiment(content)

        friendly = generate_friendly_comment(content)
        funny = generate_funny_comment(content)
        congratulating = generate_congratulating_comment(content)
        questioning = generate_questioning_comment(content)
        disagreement = generate_disagreement_comment(content)

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
