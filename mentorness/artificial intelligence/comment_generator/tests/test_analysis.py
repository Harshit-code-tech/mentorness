# Test cases for the analysis module
import unittest
from app.analysis import analyze_sentiment, detect_tone


class TestAnalysis(unittest.TestCase):

    def test_analyze_sentiment(self):
        # Test valid text
        text = "This is a great tool!"
        result = analyze_sentiment(text)

        self.assertNotIn('error', result)
        self.assertIn('polarity', result)
        self.assertIn('nltk_sentiment', result)
        self.assertIn('transformer_sentiment', result)
        self.assertIn('vader_sentiment', result)
        self.assertIn('tone', result)

    def test_detect_tone(self):
        # Test valid tone detection
        text = "This is a great tool!"
        sentiment_results = analyze_sentiment(text)

        if 'error' not in sentiment_results:
            polarity = sentiment_results['polarity']
            nltk_sentiment = sentiment_results['nltk_sentiment']
            transformer_sentiment = sentiment_results['transformer_sentiment']
            tone = detect_tone(polarity, nltk_sentiment, transformer_sentiment, sentiment_results['vader_sentiment'])
            self.assertIn(tone, ["positive", "neutral", "negative"])
        else:
            self.fail("Error in sentiment analysis, cannot test tone detection.")


if __name__ == '__main__':
    unittest.main()
