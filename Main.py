from GPTeam.Mask import clip_image
from GPTeam.Trou import fill_mask




clip_image('input/porsche-911.jpg', ['sponsor banner green and white in background','banner'])
fill_mask('input/porsche-911.jpg', 'mask.png')