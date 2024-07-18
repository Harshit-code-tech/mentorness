import os

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)