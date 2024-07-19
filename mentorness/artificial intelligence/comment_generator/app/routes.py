from flask import Blueprint, render_template, request
import time
from .comments import generate_comments

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    start_time = time.time()
    comments = None
    content = ""
    error = None
    if request.method == 'POST':
        try:
            content = request.form['content']
            if not content.strip():
                raise ValueError("No content provided")
            comments = generate_comments(content)
        except Exception as e:
            error = str(e)
    end_time = time.time()
    processing_time = end_time - start_time
    return render_template('index.html', content=content, comments=comments, processing_time=processing_time, error=error)
