# test_comments.py
import unittest
from app.comments import generate_comments

class TestComments(unittest.TestCase):
    def test_generate_comments(self):
        text = "This is a great tool!"
        comments = generate_comments(text)
        self.assertIn('friendly', comments)
        self.assertIn('funny', comments)
        self.assertIn('congratulating', comments)
        self.assertIn('questioning', comments)
        self.assertIn('disagreement', comments)

if __name__ == '__main__':
    unittest.main()
