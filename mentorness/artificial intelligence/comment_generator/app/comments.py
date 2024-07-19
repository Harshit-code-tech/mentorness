import random
from .analysis import analyze_sentiment, detect_tone

def get_random_excerpt(content):
    words = content.split()
    if len(words) <= 50:
        return content
    start_idx = random.randint(0, len(words) - 50)
    excerpt = ' '.join(words[start_idx:start_idx + 50])
    return excerpt

def generate_friendly_comment(content):
    templates = [
        "Great read! {}... Keep it up!",
        "Loved this article! {}... Well done!",
        "This was very insightful. {}... Thank you!",
        "Wonderful piece! {}... I learned a lot.",
        "Fantastic! {}... Keep writing!"
    ]
    excerpt = get_random_excerpt(content)
    return random.choice(templates).format(excerpt)

def generate_funny_comment(content):
    templates = [
        "LOL! {}... That's hilarious!",
        "This cracked me up! {}... So funny!",
        "Haha! {}... Great sense of humor!",
        "What a funny read! {}... Loved it!",
        "This made me laugh! {}... Well done!"
    ]
    excerpt = get_random_excerpt(content)
    return random.choice(templates).format(excerpt)

def generate_congratulating_comment(content):
    templates = [
        "Congratulations on this amazing piece! {}...",
        "Well done! {}... Fantastic work!",
        "Great job! {}... Keep it up!",
        "Bravo! {}... This is excellent!",
        "Kudos! {}... Impressive article!"
    ]
    excerpt = get_random_excerpt(content)
    return random.choice(templates).format(excerpt)

def generate_questioning_comment(content):
    templates = [
        "Interesting point! Can you elaborate on {}?",
        "I have a question about {}... Can you explain more?",
        "Could you clarify {}? I'm curious.",
        "What do you mean by {}? Please elaborate.",
        "Can you provide more details on {}? Thanks!"
    ]
    excerpt = get_random_excerpt(content)
    return random.choice(templates).format(excerpt)

def generate_disagreement_comment(content):
    templates = [
        "I don't quite agree with {}... Here's why...",
        "Not sure I agree with {}... What do you think?",
        "I see it differently. {}... Let's discuss.",
        "Interesting perspective, but I disagree with {}...",
        "I have a different opinion on {}... Let's talk."
    ]
    excerpt = get_random_excerpt(content)
    return random.choice(templates).format(excerpt)

def generate_comments(content):
    polarity, nltk_sentiment, transformer_sentiment = analyze_sentiment(content)
    tone = detect_tone(polarity, nltk_sentiment, transformer_sentiment)

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
