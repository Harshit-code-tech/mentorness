# comments.py
import random
from .analysis import analyze_sentiment
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from langdetect import detect

def summarize_text(text, sentence_count=1):
    """
    Summarizes the input text using LSA summarization.
    """
    try:
        language = detect(text)
        parser = PlaintextParser.from_string(text, Tokenizer(language))
        stemmer = Stemmer(language)
        summarizer = LsaSummarizer(stemmer)
        summarizer.stop_words = get_stop_words(language)

        summary = summarizer(parser.document, sentence_count)
        return " ".join(str(sentence) for sentence in summary)
    except Exception as e:
        # Log the error
        print(f"Error in summarizing text: {e}")
        # Provide a fallback summary if summarization fails
        return text[:150] + '...'

def generate_friendly_comment(content, tone):
    friendly_templates = {
        "positive": [
            "Great read! {snippet}... Keep up the positive vibes!",
            "I really enjoyed this! {snippet}... Fantastic work, well done!",
            "This is so well-written! {snippet}... Excellent job, very uplifting!"
        ],
        "neutral": [
            "Interesting read. {snippet}... Thanks for sharing!",
            "Well-written piece. {snippet}... Good work!",
            "Nice job on this! {snippet}... Keep going!"
        ],
        "negative": [
            "Not my favorite. {snippet}... Maybe try a different approach?",
            "I found this a bit off. {snippet}... Could use some improvement.",
            "This didn’t resonate with me. {snippet}... Maybe refine your points?"
        ]
    }
    try:
        snippet = summarize_text(content)
        return random.choice(friendly_templates.get(tone, friendly_templates['neutral'])).format(snippet=snippet)
    except Exception as e:
        # Log the error
        print(f"Error generating friendly comment: {e}")
        return "Error generating friendly comment."

def generate_funny_comment(content, tone):
    funny_templates = {
        "positive": [
            "LOL! {snippet}... This is hilarious!",
            "You cracked me up! {snippet}... Great sense of humor!",
            "This is comedy gold! {snippet}... So funny, I’m still laughing!"
        ],
        "neutral": [
            "This is amusing. {snippet}... Quite the chuckle!",
            "Nice try at humor! {snippet}... Made me smile!",
            "Funny in parts. {snippet}... Had a good laugh!"
        ],
        "negative": [
            "Not quite funny. {snippet}... Missed the mark for me.",
            "Trying too hard? {snippet}... Didn’t quite land.",
            "I didn’t find this funny. {snippet}... Maybe less is more?"
        ]
    }
    try:
        snippet = summarize_text(content)
        return random.choice(funny_templates.get(tone, funny_templates['neutral'])).format(snippet=snippet)
    except Exception as e:
        # Log the error
        print(f"Error generating funny comment: {e}")
        return "Error generating funny comment."

def generate_congratulating_comment(content, tone):
    congratulating_templates = {
        "positive": [
            "Congratulations on this amazing piece! {snippet}...",
            "Well done! {snippet}... Keep up the great work!",
            "Bravo! {snippet}... Impressive achievement!"
        ],
        "neutral": [
            "Nice work! {snippet}... Good effort!",
            "Well done on this piece. {snippet}... Keep improving!",
            "Great attempt. {snippet}... Keep pushing forward!"
        ],
        "negative": [
            "Keep trying! {snippet}... Improvement needed.",
            "Not quite there yet. {snippet}... Better luck next time.",
            "A valiant effort. {snippet}... Needs more work."
        ]
    }
    try:
        snippet = summarize_text(content)
        return random.choice(congratulating_templates.get(tone, congratulating_templates['neutral'])).format(snippet=snippet)
    except Exception as e:
        # Log the error
        print(f"Error generating congratulating comment: {e}")
        return "Error generating congratulating comment."

def generate_questioning_comment(content, tone):
    questioning_templates = {
        "positive": [
            "Interesting point! Can you elaborate on {snippet}?",
            "I’d love to know more about {snippet}... Could you expand?",
            "This is intriguing! {snippet}... Can you provide more details?"
        ],
        "neutral": [
            "Could you clarify {snippet}...?",
            "Interesting idea. {snippet}... Can you explain further?",
            "I have a few questions about {snippet}... What do you think?"
        ],
        "negative": [
            "I’m not sure about {snippet}... Can you explain?",
            "This point seems off. {snippet}... Could you clarify?",
            "Not clear on {snippet}... What’s your reasoning?"
        ]
    }
    try:
        snippet = summarize_text(content)
        return random.choice(questioning_templates.get(tone, questioning_templates['neutral'])).format(snippet=snippet)
    except Exception as e:
        # Log the error
        print(f"Error generating questioning comment: {e}")
        return "Error generating questioning comment."

def generate_disagreement_comment(content, tone):
    disagreement_templates = {
        "positive": [
            "I see your point, but {snippet}... Here’s another perspective.",
            "Interesting view, but {snippet}... Here’s my take.",
            "I respect your opinion, but {snippet}... Let’s consider this angle."
        ],
        "neutral": [
            "I don’t quite agree with {snippet}... Here’s my view.",
            "Different perspective: {snippet}... What do you think?",
            "I see it differently. {snippet}... Here’s my opinion."
        ],
        "negative": [
            "I strongly disagree with {snippet}... Here’s why.",
            "Not convinced by {snippet}... Here’s an alternative view.",
            "I find {snippet}... problematic. Here’s my argument."
        ]
    }
    try:
        snippet = summarize_text(content)
        return random.choice(disagreement_templates.get(tone, disagreement_templates['neutral'])).format(snippet=snippet)
    except Exception as e:
        # Log the error
        print(f"Error generating disagreement comment: {e}")
        return "Error generating disagreement comment."

def generate_comments(content):
    try:
        sentiment_results = analyze_sentiment(content)

        if 'error' in sentiment_results:
            return {
                "friendly": "",
                "funny": "",
                "congratulating": "",
                "questioning": "",
                "disagreement": "",
                "tone": "error"
            }

        tone = sentiment_results['tone']
        friendly = generate_friendly_comment(content, tone)
        funny = generate_funny_comment(content, tone)
        congratulating = generate_congratulating_comment(content, tone)
        questioning = generate_questioning_comment(content, tone)
        disagreement = generate_disagreement_comment(content, tone)

        return {
            "friendly": friendly,
            "funny": funny,
            "congratulating": congratulating,
            "questioning": questioning,
            "disagreement": disagreement,
            "tone": tone
        }
    except Exception as e:
        # Log the error
        print(f"Error generating comments: {e}")
        return {
            "friendly": "",
            "funny": "",
            "congratulating": "",
            "questioning": "",
            "disagreement": "",
            "tone": "error"
        }
