from flask import Blueprint, render_template, request
import time
from .comments import generate_comments

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    start_time = time.time()  # Start timer
    comments = None
    content = ""
    if request.method == 'POST':
        content = request.form['content']
        # Call your content analysis and comment generation functions here
        comments = generate_comments(content)
    end_time = time.time()  # End timer
    processing_time = end_time - start_time
    return render_template('index.html', content=content, comments=comments, processing_time=processing_time)
