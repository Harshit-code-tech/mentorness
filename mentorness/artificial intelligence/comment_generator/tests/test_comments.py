import unittest
import sys
import os

# Add the parent directory to the PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.comments import generate_comments

class TestComments(unittest.TestCase):

    def setUp(self):
        """
        Set up test data for the test cases.
        """
        self.valid_text = "This is a great tool!"
        self.empty_text = ""
        self.long_text = " ".join(["This is a long text."] * 100)

    def test_generate_comments_valid_text(self):
        """
        Test the generate_comments function with valid text input.
        """
        comments = generate_comments(self.valid_text)
        self.assertIn('friendly', comments)
        self.assertIn('funny', comments)
        self.assertIn('congratulating', comments)
        self.assertIn('questioning', comments)
        self.assertIn('disagreement', comments)

        self.assertIsInstance(comments['friendly'], str)
        self.assertIsInstance(comments['funny'], str)
        self.assertIsInstance(comments['congratulating'], str)
        self.assertIsInstance(comments['questioning'], str)
        self.assertIsInstance(comments['disagreement'], str)
        self.assertIn(comments['tone'], ['positive', 'neutral', 'negative'])

    def test_generate_comments_empty_text(self):
        """
        Test the generate_comments function with an empty text input.
        """
        comments = generate_comments(self.empty_text)
        self.assertIn('friendly', comments)
        self.assertIn('funny', comments)
        self.assertIn('congratulating', comments)
        self.assertIn('questioning', comments)
        self.assertIn('disagreement', comments)

        self.assertTrue(comments['tone'] == 'error' or comments['tone'] in ['positive', 'neutral', 'negative'])

    def test_generate_comments_long_text(self):
        """
        Test the generate_comments function with a long text input.
        """
        comments = generate_comments(self.long_text)
        self.assertIn('friendly', comments)
        self.assertIn('funny', comments)
        self.assertIn('congratulating', comments)
        self.assertIn('questioning', comments)
        self.assertIn('disagreement', comments)

        self.assertIsInstance(comments['friendly'], str)
        self.assertIsInstance(comments['funny'], str)
        self.assertIsInstance(comments['congratulating'], str)
        self.assertIsInstance(comments['questioning'], str)
        self.assertIsInstance(comments['disagreement'], str)
        self.assertIn(comments['tone'], ['positive', 'neutral', 'negative'])

    def test_generate_comments_special_characters(self):
        """
        Test the generate_comments function with text containing special characters.
        """
        text = "!@#$%^&*()_+"
        comments = generate_comments(text)
        self.assertIn('friendly', comments)
        self.assertIn('funny', comments)
        self.assertIn('congratulating', comments)
        self.assertIn('questioning', comments)
        self.assertIn('disagreement', comments)
        self.assertIn(comments['tone'], ['positive', 'neutral', 'negative'])

    def test_generate_comments_unsupported_language(self):
        """
        Test the generate_comments function with text in an unsupported language.
        """
        text = "これは日本語のテキストです"  # Japanese text
        comments = generate_comments(text)
        self.assertIn('friendly', comments)
        self.assertIn('funny', comments)
        self.assertIn('congratulating', comments)
        self.assertIn('questioning', comments)
        self.assertIn('disagreement', comments)
        self.assertTrue(comments['tone'] == 'error' or comments['tone'] in ['positive', 'neutral', 'negative'])

if __name__ == '__main__':
    unittest.main()
