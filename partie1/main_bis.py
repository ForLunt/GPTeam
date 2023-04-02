import torch
import torchvision
import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision.models.vgg import vgg16
import os
#import cv2

# Définission des classes
class_names  = ['exclure', 'ok', 'retoucher']

print("Chargement du modèle", end="")
model = vgg16()
print(".", end="")
num_ftrs = model.classifier[6].in_features
print(".", end="")
model.fc = nn.Linear(num_ftrs, 3)
print(".", end="")
model.load_state_dict(torch.load('./model_custom_weightsV2.pth'))
print("Chargement du modèle terminé")


model.eval()

print("Préparation des transformations des images")
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Recherche des images", end="")
dirname = "Partie 1"
images = {}

for class_name in class_names:
    print(".", end="")
    # On récupère toutes les images du répertoire
    images[class_name] = []
    for filename in os.listdir(os.path.join(dirname, class_name)):
        filename = os.path.join(dirname, class_name, filename)
        image = Image.open(filename)
        images[class_name].append((filename, image))

print("Recherche des images terminée")

print("Début de l'analyse des images")

found = 0
total = 0
# On parcourt toutes les classes
for class_name in class_names:
    # On parcourt toutes les images de la classe
    for filename, image in images[class_name]:
        input_image = transform(image).unsqueeze(0)

        # On passe l'image dans le modèle de détection de classe

        with torch.no_grad():
            detections = model(input_image)

        # On analyse les résultats de détection
        _, preds = torch.max(detections, 1)

        class_found = class_names[preds.tolist()[0]]
        print(f"Image {filename}, found class = {class_found}, true class = {class_name}")
        if class_found == class_name:
            found += 1
        total += 1

print(f"Résultat de l'analyse : {found}/{total} = {found/total*100}% de réussite")

