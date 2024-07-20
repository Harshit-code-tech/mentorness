
import nltk
nltk.download('stopwords')  # Ensure this is added to download the stop-words
from nltk.corpus import stopwords
stop_words = stopwords.words('dutch')  # Now this should work without error