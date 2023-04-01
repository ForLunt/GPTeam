from Mask import clip_image
from Fill import fill_mask



prompt = "sponsor banner green and white in background"
prompt2 = "banner"

# Suppression elelents indésirables selon prompt
clip_image('input/porsche-911.jpg', [prompt,prompt2])
fill_mask('input/porsche-911.jpg', "output/output_"+prompt.replace(" ","_") + ".png")