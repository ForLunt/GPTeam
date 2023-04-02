from torchvision.models.vgg import vgg16
import torch.nn as nn
import torchvision
import torch
import PIL

class_names  = ['exclure', 'ok', 'retoucher']

class Modele :
    def __init__(self) :
        print("Chargement du modèle", end="")
        self.__model = vgg16()
        print(".", end="")
        num_ftrs = self.__model.classifier[6].in_features
        print(".", end="")
        self.__model.fc = nn.Linear(num_ftrs, 3)
        print(".", end="")
        self.__model.load_state_dict(torch.load('./model_custom_weightsV2.pth'))
        print("Chargement du modèle terminé")

        self.__model.eval()

        print("Préparation des transformations des images")
        self.__transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def processImage(self, image : PIL.Image) :
        print("Début du traitement de l'image")
        input_image = self.__transform(image).unsqueeze(0)

        # On passe l'image dans le modèle de détection de classe

        print("Conversion")
        with torch.no_grad():
            detections = self.__model(input_image)

        print("Analyse des résultats")
        # On analyse les résultats de détection
        _, preds = torch.max(detections, 1)

        class_found = class_names[preds.tolist()[0]]
        return class_found