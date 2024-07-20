# test_analysis.py
import unittest
from app.analysis import analyze_sentiment, detect_tone


class TestAnalysis(unittest.TestCase):
    def test_analyze_sentiment(self):
        text = "This is a great tool!"
        polarity, nltk_sentiment, transformer_sentiment = analyze_sentiment(text)
        self.assertIsNotNone(polarity)
        self.assertIsNotNone(nltk_sentiment)
        self.assertIsNotNone(transformer_sentiment)

    def test_detect_tone(self):
        text = "This is a great tool!"
        polarity, nltk_sentiment, transformer_sentiment = analyze_sentiment(text)
        tone = detect_tone(polarity, nltk_sentiment, transformer_sentiment)
        self.assertIn(tone, ["positive", "neutral", "negative"])


if __name__ == '__main__':
    unittest.main()
