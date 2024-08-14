import unittest
import sys
import os

# Adjust the path to ensure that the app module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))

from analysis import analyze_sentiment, detect_tone


class TestAnalysis(unittest.TestCase):

    def test_analyze_sentiment(self):
        """
        Test the analyze_sentiment function with valid text input.
        """
        text = "This is a great tool!"
        result = analyze_sentiment(text)

        # Ensure no error is returned
        self.assertNotIn('error', result)

        # Ensure all expected keys are present in the result
        self.assertIn('polarity', result)
        self.assertIn('nltk_sentiment', result)
        self.assertIn('transformer_sentiment', result)
        self.assertIn('vader_sentiment', result)
        self.assertIn('tone', result)

    def test_detect_tone(self):
        """
        Test the detect_tone function with valid sentiment results.
        """
        text = "This is a great tool!"
        sentiment_results = analyze_sentiment(text)

        if 'error' not in sentiment_results:
            polarity = sentiment_results['polarity']
            nltk_sentiment = sentiment_results['nltk_sentiment']
            transformer_sentiment = sentiment_results['transformer_sentiment']
            vader_sentiment = sentiment_results['vader_sentiment']
            tone = detect_tone(polarity, nltk_sentiment, transformer_sentiment, vader_sentiment)
            self.assertIn(tone, ["positive", "neutral", "negative"])
        else:
            self.fail("Error in sentiment analysis, cannot test tone detection.")

    def test_analyze_sentiment_empty_string(self):
        """
        Test how the analyze_sentiment function handles an empty string.
        """
        text = ""
        result = analyze_sentiment(text)
        self.assertEqual(result, {'error': 'Empty text provided'})

    def test_analyze_sentiment_long_text(self):
        """
        Test how the analyze_sentiment function handles a very long text.
        """
        text = "This is a " + "very long text " * 1000
        result = analyze_sentiment(text)
        # Ensure no error is returned and check for sentiment results
        self.assertNotIn('error', result)
        self.assertIn('polarity', result)
        self.assertIn('nltk_sentiment', result)
        self.assertIn('transformer_sentiment', result)
        self.assertIn('vader_sentiment', result)

    def test_analyze_sentiment_unsupported_language(self):
        """
        Test how the analyze_sentiment function handles text in an unsupported language.
        """
        text = "これは日本語のテキストです"  # Japanese text
        result = analyze_sentiment(text)
        self.assertEqual(result, {'error': 'Unsupported language for sentiment analysis'})

    def test_analyze_sentiment_special_characters(self):
        """
        Test how the analyze_sentiment function handles text with special characters.
        """
        text = "This is amazing! 😄🎉 #awesome"
        result = analyze_sentiment(text)
        # Ensure no error is returned and check for sentiment results
        self.assertNotIn('error', result)
        self.assertIn('polarity', result)
        self.assertIn('nltk_sentiment', result)
        self.assertIn('transformer_sentiment', result)
        self.assertIn('vader_sentiment', result)


if __name__ == '__main__':
    unittest.main()
