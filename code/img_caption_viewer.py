from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import math
import random
import os


def split_long_caption(caption, max_line_length):

    lines = []
    start_line = 0
    for i in range(len(caption)):
        if i - start_line >= max_line_length:
            lines.append(caption[start_line:i])
            start_line = i
    if (i - start_line) > 0 and (i- start_line) < max_line_length:
        lines.append(caption[start_line:])
    return lines




# img -> PIL image object
# caption -> string
def createCaptionImg(img, caption):

    # dimensioni immagine
    W, H = img.size
    
    # imposto il font e lo spazio tra le righe
    font_size = 10
    interline = int(font_size / 2)

    # lunghezza max linea in "numero di lettere"
    max_line_length = math.floor(W / font_size) * 2

    # faccio lo split delle linee in modo da avere lunghezza max pari a "max_line_length"
    lines = split_long_caption(caption, max_line_length)

    H_caption = len(lines) * (font_size + interline)

    caption_img = Image.new('RGB', (W, H_caption), color = (255,255,255))
    draw = ImageDraw.Draw(caption_img)
    
    font_type = ImageFont.truetype("../arial.ttf", font_size)

    # scrivo linea per linea
    start_px_h = 0
    for i,line in enumerate(lines):
        v_offset = i * (font_size + interline)

        draw.text((0, start_px_h + v_offset),line, (0,0,0), font = font_type)

    dst = Image.new('RGB', (W, H + H_caption))
    dst.paste(caption_img, (0, 0))
    dst.paste(img, (0, H_caption))
    return dst



# legge il file con la mappa immagini - caption e restituisce le due liste separate
def get_captions_and_imgs(file_name):

    img_names = []
    captions = []

    with open(file_name, 'r') as f:
        while True:       
            line = f.readline()
            if not line:
                break
            splitted = line.split('-')
            img_names.append(splitted[0].strip())
            captions.append(''.join(splitted[1:]).strip())

    return list(zip(img_names, captions))




if __name__ == "__main__":

    data_path = "../models/coco/netG_epoch_100/"
    output_path = data_path + "/img_with_captions/"
    file_path = data_path + "captions_imgs_map.txt"
    img_captions_dict = get_captions_and_imgs(file_path)

    #random.seed(0)
    #random.shuffle(img_captions_dict)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i in range(len(img_captions_dict)):

        print("img n. {}".format(i+1))
        img_name = img_captions_dict[i][0]
        caption = img_captions_dict[i][1]
        img = Image.open(data_path + img_name)
        caption_img = createCaptionImg(img, caption)
        #caption_img.show()

        caption_img.save(output_path + img_name)

    


