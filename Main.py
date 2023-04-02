import os

from PIL import Image

from Fill import fill_mask
from Mask import clip_image
from style import generate_img, load_image

images = []

for filename in os.listdir('./Resultat_Partie_5/retoucher'):
    img = Image.open(os.path.join('./Resultat_Partie_5/retoucher', filename))
    if img is not None:
        images.append(img)

for image in images:
    # Suppression elelents indésirables selon prompt
    #Ask user for prompt
    print(image.filename)
    prompt1 = input("Entrez le prompt pour la suppression d'éléments indésirables: ")
    clip_image(image, prompt1)
    prompt2 = input("Entrez le prompt pour le remplacement d'éléments indésirables: ")
    fill_mask(image, image.filename, prompt2)

# # Stylisation photo
# generate_img(load_image("input/origin.jpg"), load_image("input/style.png"), "comicGen")
# generate_img(load_image("input/origin.jpg"), load_image("input/style.png"), "van_gohgGen")
# generate_img(load_image("input/origin.jpg"), load_image("input/style.png"), "Stylized_adGen")
