"""flask server."""


import os
import random
import requests
from flask import Flask, render_template, abort, request
from MemeEngine import MemeEngine
from QuoteEngine import Ingestor


app = Flask(__name__)
meme_engine = MemeEngine.MemeEngine('./static')


def load_resources():
    """Load all resources."""
    quote_files = [
        './_data/DogQuotes/DogQuotesTXT.txt',
        './_data/DogQuotes/DogQuotesDOCX.docx',
        './_data/DogQuotes/DogQuotesPDF.pdf',
        './_data/DogQuotes/DogQuotesCSV.csv'
    ]
    quotes = []
    for quote_file in quote_files:
        quotes.extend(Ingestor.parse(quote_file))

    images_path = "./_data/photos/dog/"
    image_files = [os.path.join(root, name)
                   for root, dirs, files in os.walk(images_path)
                   for name in files]
    return quotes, image_files


QUOTES, IMAGES = load_resources()


@app.route('/')
def meme_rand():
    """Generate a random meme."""
    image_path = random.choice(IMAGES)
    quote = random.choice(QUOTES)

    path = meme_engine.make_meme(image_path, quote.body, quote.author)
    return render_template('meme.html', path=path)


@app.route('/create', methods=['GET'])
def meme_form():
    """User input for meme information."""
    return render_template('meme_form.html')


@app.route('/create', methods=['POST'])
def meme_post():
    """Create a user-defined meme."""
    image_url = request.form['image_url']
    body = request.form['body']
    author = request.form['author']

    response = requests.get(image_url, allow_redirects=True)
    image_name = random.randint(0, 100000000)
    image_path = f'./tmp/{image_name}.jpg'

    with open(image_path, 'wb') as f:
        f.write(response.content)

    path = meme_engine.make_meme(image_path, body, author)
    return render_template('meme.html', path=path)


if __name__ == "__main__":
    app.run()
