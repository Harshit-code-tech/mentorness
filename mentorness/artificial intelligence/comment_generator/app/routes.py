from flask import Blueprint, render_template, request
import time
from .comments import generate_comments

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    start_time = time.time()
    comments = None
    content = ""
    processing_time = 0

    try:
        if request.method == 'POST':
            content = request.form.get('content', '').strip()
            if content:
                comments = generate_comments(content)
            else:
                comments = {"tone": "empty"}
        end_time = time.time()
        processing_time = end_time - start_time
    except Exception as e:
        print(f"Error in processing: {e}")

    return render_template('index.html', content=content, comments=comments, processing_time=processing_time)
