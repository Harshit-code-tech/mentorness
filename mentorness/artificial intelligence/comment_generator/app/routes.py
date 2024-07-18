from flask import Blueprint, render_template, request
from .comments import generate_comments

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    comments = {}
    content = ""
    if request.method == 'POST':
        content = request.form['content']
        comments = generate_comments(content)
    return render_template('index.html', content=content, comments=comments)
