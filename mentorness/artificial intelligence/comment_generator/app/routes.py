# routes.py
from flask import Blueprint, render_template, request
import time
from app.comments import generate_comments  # Updated import

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
                comments = generate_comments(content)  # Call updated function

                # Handle the case where 'tone' is 'error'
                if comments['tone'] == 'error':
                    error_message = "Error generating comments. Please try again."
                    comments = None  # Clear comments if there's an error
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"

    end_time = time.time()
    processing_time = end_time - start_time
    return render_template('index.html', content=content, comments=comments, processing_time=processing_time,
                           error_message=error_message)
