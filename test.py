import torch
import requests
import numpy as np
import torch.nn.functional as F
from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt


def resize_image(imageName, size_max):
    max_size = size_max  # la taille maximale pour la plus grande dimension

    # ouvrir l'image et récupérer ses dimensions
    image = Image.open(imageName)
    width, height = image.size

    # calculer la nouvelle taille de l'image tout en conservant le ratio d'aspect
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))

    # redimensionner l'image
    image = image.resize((new_width, new_height), resample=Image.BICUBIC)

    # enregistrer l'image redimensionnée
    image.save('redimensionne_'+imageName)

def clip_image(imageName):
    # load model
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.eval();

    # non-strict, because we only stored decoder weights (not CLIP weights)
    model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False);

    # load and normalize image
    input_image = Image.open(imageName)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352), antialias=True),
    ])
    img = transform(input_image).unsqueeze(0)

    prompts = ['sponsor banner green and white in background','banner']

    with torch.no_grad():
        preds = model(img.repeat(len(prompts),1,1,1), prompts)[0]

    print("Image size : ",input_image.size)

    # get resized prediction
    prediction = torch.where(preds[0][0] > 0, torch.tensor(255), torch.tensor(0))
    prediction = prediction.numpy().astype(np.uint8)
    prediction = Image.fromarray(prediction)
    prediction = prediction.resize(input_image.size)

    # visualize prediction and save to file

    for i in range(len(prompts)):
        fig, ax = plt.subplots(1, 1)
        prediction = torch.where(preds[i][0] > 0, torch.tensor(255), torch.tensor(0))
        prediction = prediction.numpy().astype(np.uint8)
        prediction = Image.fromarray(prediction)
        prediction = prediction.resize(input_image.size)
        ax.imshow(prediction, cmap='gray')
        ax.axis('off')
        fig.canvas.manager.full_screen_toggle()
        print(fig.get_size_inches())
        fig.savefig('output_{}.png'.format(prompts[i].replace(' ', '_')), bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        # resize_image('output_{}.png'.format(prompts[i].replace(' ', '_')), 512)
        # resize_image(imageName, 512)

        output_image = Image.open('output_{}.png'.format(prompts[i].replace(' ', '_')))
        resized_image = output_image.resize(input_image.size)
        resized_image.save('re_'+'output_{}.png'.format(prompts[i].replace(' ', '_')))

clip_image('porsche-911.jpg')