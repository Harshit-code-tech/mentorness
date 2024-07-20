from flask import Blueprint, render_template, request, flash
import time
from .comments import generate_comments

main = Blueprint('main', __name__)


@main.route('/', methods=['GET', 'POST'])
def index():
    start_time = time.time()
    comments = None
    content = ""
    error_message = ""

    try:
        if request.method == 'POST':
            content = request.form.get('content', '')
            if not content:
                error_message = "Content cannot be empty."
            else:
                comments = generate_comments(content)
    except Exception as e:
        error_message = str(e)

    end_time = time.time()
    processing_time = end_time - start_time
    return render_template('index.html', content=content, comments=comments, processing_time=processing_time,
                           error_message=error_message)
