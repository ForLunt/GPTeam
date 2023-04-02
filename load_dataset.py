import os
# On chage les données d'entrainement depuis le dossier training_dataset et on les classifie en 3 catégories : ok, exclure et retoucher en fonction du dossier dans lequel elles se trouvent

# On récupère les images du dossier ok
imgs_ok = []
for file in os.listdir("training_dataset/ok/"):
    imgs_ok.append(file)

# On récupère les images du dossier exclure
imgs_exclure = []
for file in os.listdir("training_dataset/exclure/"):
    imgs_exclure.append(file)

# On récupère les images du dossier retoucher
imgs_retoucher = []
for file in os.listdir("training_dataset/retoucher/"):
    imgs_retoucher.append(file)
                            
# On affiche le nom des images du dossier ok
print("Images du dossier ok :")
print(imgs_ok)
print("")
# On affiche le nom des images du dossier exclure
print("Images du dossier exclure :")
print(imgs_exclure)
print("")
# On affiche le nom des images du dossier retoucher
print("Images du dossier retoucher :")
print(imgs_retoucher)
print("")

import torchvision
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
# Get a batch of training data
inputs, classes = next(iter(trainloader))
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out,title=[class_names[x] for x in classes])