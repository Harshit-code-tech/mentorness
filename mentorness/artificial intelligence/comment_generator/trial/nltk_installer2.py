import nltk

# Download stop-words
nltk.download('stopwords')

from nltk.corpus import stopwords

# Access stop-words for Finnish and Dutch
finnish_stop_words = stopwords.words('finnish')
dutch_stop_words = stopwords.words('dutch')

# Now you can use finnish_stop_words and dutch_stop_words in your application