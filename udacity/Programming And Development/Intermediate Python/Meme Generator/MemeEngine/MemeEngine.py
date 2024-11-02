"""MemeEngine."""


from PIL import Image, ImageDraw
import random


class MemeEngine:
    """The MemeEngine class drawing text to mages."""

    def __init__(self, output_dir: str):
        """
        Initialize the MemeEngine class.

        :param output_dir: The directory path will be saved.
        """
        self.output_dir = output_dir

    def make_meme(self,
                  img_path: str,
                  text: str,
                  author: str,
                  width: int = 500) -> str:
        """
        Generate a meme by adding text to an image.

        :param img_path: The path of the image file.
        :param text: The text to be added to the image.
        :param author: The author of the text.
        :param width: The width of the image.
        :return: The path of the generated meme image.
        """
        try:
            image = Image.open(img_path)
        except Exception as e:
            print(f"Error: Could not open file {img_path}.Exception: {str(e)}")
            return ""

        aspect_ratio = width / float(image.size[0])
        height = int(aspect_ratio * float(image.size[1]))
        image = image.resize((width, height), Image.NEAREST)
        text = text.replace("\u2019", "")
        author = author.replace("\u2019", "")
        rand_x = random.randint(0, int(width / 2))
        rand_y = random.randint(0, int(height / 2))
        draw = ImageDraw.Draw(image)
        draw.text((rand_x, rand_y), text, fill='white')
        draw.text((rand_x, (rand_y + 20)), ('   -' + author), fill='white')

        try:
            str_ran = str(random.randint(0, 1000))
            out_file = self.output_dir + '/' + str_ran + '.jpg'
            image.save(out_file, "JPEG")
        except Exception as e:
            print(f"Error: Could not save image file {out_file}.Err: {str(e)}")
            return ""

        return out_file
