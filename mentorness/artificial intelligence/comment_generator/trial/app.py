from flask import Flask, render_template, request
from transformers import pipeline
import time

app = Flask(__name__)

# Load sentiment analysis and text generation models
sentiment_analysis = pipeline('sentiment-analysis')
text_generator = pipeline('text-generation', model='distilgpt2')

@app.route('/', methods=['GET', 'POST'])
def index():
    start_time = time.time()  # Start timer
    comments = None
    if request.method == 'POST':
        content = request.form['content']
        comments = generate_comments(content)
    end_time = time.time()  # End timer
    processing_time = end_time - start_time
    return render_template('index.html', comments=comments, processing_time=processing_time)

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

if __name__ == '__main__':
    app.run(debug=True)
