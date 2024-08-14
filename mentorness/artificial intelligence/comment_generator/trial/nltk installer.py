import nltk

def download_nltk_resources():
    try:
        # Check if vader_lexicon is available, download if not
        if not nltk.data.find('sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt'):
            nltk.download('vader_lexicon')
    except LookupError:
        # Attempt to download the resource if not found
        nltk.download('vader_lexicon')
    except Exception as e:
        print(f"An error occurred while downloading NLTK resources: {e}")

def download_nltk_stopwatch():
    try:
        # Check and download vader_lexicon if not available
        if not nltk.data.find('sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt'):
            nltk.download('vader_lexicon')

        # Download Finnish stop-words
        nltk.download('stopwords')

    except LookupError as e:
        print(f"LookupError: {e}")
    except Exception as e:
        print(f"An error occurred while downloading NLTK resources: {e}")




# Call this function early in your application's startup sequence
download_nltk_resources()
# Call this function early in your application's startup sequence
download_nltk_stopwatch()