from transformers import pipeline

# Load sentiment analysis and text generation models
sentiment_analysis = pipeline('sentiment-analysis')
text_generator = pipeline('text-generation', model='distilgpt2')

def generate_comments(content):
    # Perform sentiment analysis
    sentiment = sentiment_analysis(content)[0]

    # Generate comments based on sentiment
    friendly_comment = text_generator(f"{content} This is really nice!")[0]['generated_text']
    funny_comment = text_generator(f"{content} Haha, that's hilarious!")[0]['generated_text']
    congratulating_comment = text_generator(f"{content} Congratulations!")[0]['generated_text']
    questioning_comment = text_generator(f"{content} Can you explain more about this?")[0]['generated_text']
    disagreement_comment = text_generator(f"{content} I don't agree with this.")[0]['generated_text']

    comments = {
        'friendly': friendly_comment,
        'funny': funny_comment,
        'congratulating': congratulating_comment,
        'questioning': questioning_comment,
        'disagreement': disagreement_comment
    }
    return comments
