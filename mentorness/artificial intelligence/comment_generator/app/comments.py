from .analysis import analyze_sentiment, detect_tone

def generate_friendly_comment(content):
    return f"Great read! {content[:50]}... Keep it up!"

def generate_funny_comment(content):
    return f"LOL! {content[:50]}... Hilarious!"

def generate_congratulating_comment(content):
    return f"Congratulations on this amazing piece! {content[:50]}..."

def generate_questioning_comment(content):
    return f"Interesting point! Can you elaborate on {content[:50]}?"

def generate_disagreement_comment(content):
    return f"I don't quite agree with {content[:50]}... Here's why..."

def generate_comments(content):
    polarity, nltk_sentiment, transformer_sentiment, vader_sentiment = analyze_sentiment(content)
    tone = detect_tone(polarity, nltk_sentiment, transformer_sentiment, vader_sentiment)

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
        "disagreement": disagreement
    }
