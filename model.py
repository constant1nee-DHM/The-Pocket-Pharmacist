import numpy as np
from PIL import Image

def convert_to_mono(image):
    with Image.open(f"{image}.png") as im:
        im = im.resize((600, 600))
        im = im.convert('L')
        im.show()

convert_to_mono('test')

I = 36000 # input neurons
