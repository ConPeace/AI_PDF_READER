import os

bind = f"0.0.0.0:{os.getenv('PORT', 8080)}"
workers = 2
