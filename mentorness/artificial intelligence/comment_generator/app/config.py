# config.py
class Config:
    DEBUG = True
    SECRET_KEY = 'your_secret_key'
    DATABASE_URI = 'sqlite:///your_database.db'
    # Add any other settings you need
class TestingConfig:
    DEBUG = True
    TESTING = True

    SECRET = 'your_secret'
    DATABASE_URI = 'sqlite:///your_database.db'
    # Add any other settings you need
    KEY = 'your_key'


    # Add any other settings you need
