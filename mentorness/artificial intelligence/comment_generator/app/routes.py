# routes.py
from flask import Blueprint, render_template, request
import time
from app.comments import generate_comments

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
                # Generate comments based on the content
                comments = generate_comments(content)

                # Handle errors in comments generation
                if comments['tone'] == 'error':
                    error_message = "Error generating comments. Please try again."
                    comments = None  # Clear comments if there's an error
    except Exception as e:
        # Handle unexpected errors
        error_message = f"An unexpected error occurred: {str(e)}"
        comments = None

    end_time = time.time()
    processing_time = end_time - start_time
    return render_template('index.html', content=content, comments=comments, processing_time=processing_time,
                           error_message=error_message)
