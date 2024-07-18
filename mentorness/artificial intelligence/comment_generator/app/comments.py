from .analysis import analyze_sentiment, detect_tone

def generate_friendly_comment(content, polarity):
    if polarity > 0:
        return f"Great read! {content[:50]}... Keep it up!"
    else:
        return f"Interesting perspective! {content[:50]}... Keep writing!"

def generate_funny_comment(content, polarity):
    if polarity > 0:
        return f"LOL! {content[:50]}... Hilarious!"
    else:
        return f"That was a good one! {content[:50]}... Made me chuckle!"

def generate_congratulating_comment(content, polarity):
    if polarity > 0:
        return f"Congratulations on this amazing piece! {content[:50]}..."
    else:
        return f"Good effort on this write-up! {content[:50]}... Keep improving!"

def generate_questioning_comment(content, polarity):
    if polarity > 0:
        return f"Interesting point! Can you elaborate on {content[:50]}?"
    else:
        return f"Curious about {content[:50]}... Can you explain further?"

def generate_disagreement_comment(content, polarity):
    if polarity < 0:
        return f"I don't quite agree with {content[:50]}... Here's why..."
    else:
        return f"Different viewpoint on {content[:50]}... What do you think?"

def generate_comments(content):
    polarity, nltk_sentiment, transformer_sentiment = analyze_sentiment(content)
    tone = detect_tone(polarity, nltk_sentiment, transformer_sentiment)

    friendly = generate_friendly_comment(content, polarity)
    funny = generate_funny_comment(content, polarity)
    congratulating = generate_congratulating_comment(content, polarity)
    questioning = generate_questioning_comment(content, polarity)
    disagreement = generate_disagreement_comment(content, polarity)

    return {
        "friendly": friendly,
        "funny": funny,
        "congratulating": congratulating,
        "questioning": questioning,
        "disagreement": disagreement
    }
