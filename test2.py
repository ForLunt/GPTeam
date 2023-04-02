import torch
import torchvision
import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision.models.vgg import vgg16
#import cv2

class_names  = ['exclure', 'ok', 'retoucher']

#model = vgg16(pretrained=True)

model = vgg16()
num_ftrs = model.classifier[6].in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, 3)

model.load_state_dict(torch.load('model_custom_weightsV2.pth'))

model.eval()
#print(model)

# Chargez l'image que vous souhaitez analyser
image = Image.open('5.jpg')

# Préparez l'image pour l'analyse
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
input_image = transform(image).unsqueeze(0)

# Passez l'image dans le modèle de détection d'objet

with torch.no_grad():
    detections = model(input_image)

# Analysez les résultats de détection
_, preds = torch.max(detections, 1)

print(class_names[preds.tolist()[0]])
# Dessinez les boîtes englobantes sur l'image d'origine
"""image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
for box, score, label in zip(boxes, scores, labels):
    if score > 0.5 and label == 1:
        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(image_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Image with cars', image_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()"""
