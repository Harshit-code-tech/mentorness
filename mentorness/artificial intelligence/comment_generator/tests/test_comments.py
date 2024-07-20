import unittest
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

if __name__ == '__main__':
    unittest.main()
