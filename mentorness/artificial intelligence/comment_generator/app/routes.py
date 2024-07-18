from flask import Flask, render_template, request
import time

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    start_time = time.time()  # Start timer
    comments = None
    if request.method == 'POST':
        content = request.form['content']
        # Call your content analysis and comment generation functions here
        comments = generate_comments(content)
    end_time = time.time()  # End timer
    processing_time = end_time - start_time
    return render_template('index.html', comments=comments, processing_time=processing_time)

def generate_comments(content):
    # Dummy function to simulate comment generation
    comments = {
        'friendly': 'This is a friendly comment.',
        'funny': 'This is a funny comment.',
        'congratulating': 'Congratulations on your achievement!',
        'questioning': 'Could you clarify this point?',
        'disagreement': 'I disagree with this statement.'
    }
    return comments

if __name__ == '__main__':
    app.run(debug=True)
