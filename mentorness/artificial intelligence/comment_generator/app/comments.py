from transformers import pipeline

def generate_comments(content):
    # Specify the model name and revision
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, revision="af0f99b")

    # Dummy function to simulate comment generation based on sentiment
    results = sentiment_pipeline(content)
    sentiment = results[0]['label']

    comments = {
        'friendly': 'This is a friendly comment.',
        'funny': 'This is a funny comment.',
        'congratulating': 'Congratulations on your achievement!',
        'questioning': 'Could you clarify this point?',
        'disagreement': 'I disagree with this statement.'
    }

    if sentiment == 'POSITIVE':
        comments['friendly'] = 'Great job! This is very positive.'
    else:
        comments['disagreement'] = 'I have some concerns about this.'

    return comments
