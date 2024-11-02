"""Meme project."""


import os
import random
import argparse
from MemeEngine import MemeEngine
from QuoteEngine import Ingestor


def generate_meme(path: str = None,
                  body: str = None,
                  author: str = None) -> str:
    """Generate a meme given a path, body and author."""
    if path is None:
        images_dir = os.path.join("_data", "photos", "dog")
        images = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
        path = random.choice(images)

    if body is None:
        quote_files = ['./_data/DogQuotes/DogQuotesTXT.txt',
                       './_data/DogQuotes/DogQuotesDOCX.docx',
                       './_data/DogQuotes/DogQuotesPDF.pdf',
                       './_data/DogQuotes/DogQuotesCSV.csv']
        quotes = []
        for f in quote_files:
            quotes.extend(Ingestor.parse(f))
        quote = random.choice(quotes)
    else:
        if author is None:
            raise ValueError('Author Required if Body is Used')
        quote = QuoteModel(body, author)

    meme = MemeEngine.MemeEngine('./tmp')
    output_path = meme.make_meme(path, quote.body, quote.author)
    print("OutputPath: ", output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dog meme")
    parser.add_argument('--path', type=str,
                        default='./_data/photos/dog/xander_1.jpg',
                        help='path to an image file')
    parser.add_argument('--body', type=str, default=None, help='quote body')
    parser.add_argument('--author',
                        type=str,
                        default=None,
                        help='quote author')
    args = parser.parse_args()

    try:
        output_path = generate_meme(args.path, args.body, args.author)
        print(f"Meme created: {output_path}")
    except Exception as e:
        print(f"Error generating meme: {e}")
