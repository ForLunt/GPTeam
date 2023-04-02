import os
import torch
import numpy as np
from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

def resize_image(imageName, size_max,):
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

def clip_image(imageName, prompts):
    if (imageName == ''):
        raise Exception("Image name is empty")
    elif (len(prompts) == 0):
        raise Exception("Prompts list is empty")

    # load model
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)

    model.eval()

    # non-strict, because we only stored decoder weights (not CLIP weights)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=device), strict=False);

    input_image = imageName
    # load and normalize image
    # input_image = Image.open(imageName)

    # Create the "Output" directory if it doesn't exist
    if not os.path.exists('./Resultat_Partie_5/maskage/'):
        os.makedirs('./Resultat_Partie_5/maskage/')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352), antialias=True),
    ])
    img = transform(input_image).unsqueeze(0)

    with torch.no_grad():
        preds = model(img.repeat(len(prompts),1,1,1), prompts)[0]


    # get resized prediction
    prediction = torch.where(preds[0][0] > 0, torch.tensor(255), torch.tensor(0))
    prediction = prediction.numpy().astype(np.uint8)
    prediction = Image.fromarray(prediction)
    prediction = prediction.resize(input_image.size)

    # visualize prediction and save to file

    fig, ax = plt.subplots(1, 1)
    prediction = torch.where(preds[0] > 0, torch.tensor(255), torch.tensor(0))
    prediction = prediction.numpy().astype(np.uint8)
    prediction = Image.fromarray(prediction)
    prediction = prediction.resize(input_image.size)
    ax.imshow(prediction, cmap='gray')
    ax.axis('off')
    fig.canvas.manager.full_screen_toggle()
    fig.savefig('./Resultat_Partie_5/maskage/temp_output_{}.png'.format(prompts.replace(' ', '_')), bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # close the figure window

    output_image = Image.open('./Resultat_Partie_5/maskage/temp_output_{}.png'.format(prompts.replace(' ', '_')))
    resized_image = output_image.resize(input_image.size)
    resized_image.save('./Resultat_Partie_5/maskage/output_{}.png'.format(prompts.replace(' ', '_')))

    #remove temp files
    os.remove('./Resultat_Partie_5/maskage/temp_output_{}.png'.format(prompts.replace(' ', '_')))

