from Fill import fill_mask
from Mask import clip_image
from style import generate_img, load_image

prompt = "sponsor banner green and white in background"
prompt2 = "banner"

# Suppression elelents indésirables selon prompt
clip_image('input/porsche-911.jpg', [prompt, prompt2])
fill_mask('input/porsche-911.jpg', "output/output_" + prompt.replace(" ", "_") + ".png")

# Stylisation photo
generate_img(load_image("input/origin.jpg"), load_image("input/style.png"), "comicGen")
generate_img(load_image("input/origin.jpg"), load_image("input/style.png"), "van_gohgGen")
generate_img(load_image("input/origin.jpg"), load_image("input/style.png"), "Stylized_adGen")
