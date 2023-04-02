import PIL
import os
from flask import Flask, request, send_file
from flask_cors import CORS, cross_origin
from style import generate_img
from Mask import clip_image
from Fill import fill_mask
import json
from partie1.Modele import Modele
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# créer un modèle
model = Modele()

@app.route('/upload', methods=['POST'])
@cross_origin()
def upload():
    # vide les listes
    resultats = {"ok" : [], "retoucher" : [], "exclure"  : []}

    files = request.files
    print(files)
    for file in files.values():
        print(f"Received file {file.filename}")
        # convertie l'image en un objet PIL.Image
        file_image = PIL.Image.open(file)
        resultat = model.processImage(file_image)

        print(resultat)

        if resultat == "ok":
            resultats["ok"].append(file.filename)
        elif resultat == "retoucher":
            resultats["retoucher"].append(file.filename)
        elif resultat == "exclure":
            resultats["exclure"].append(file.filename)


    return json.dumps(resultats)


@app.route('/retouche', methods=['POST'])
@cross_origin()
def retouches():
    # On commence par supprimer les fichiers précédents dans le dossier output
    for file in os.listdir('output'):
        os.remove(os.path.join('output', file))

    # On recupere un fichier image, le prompt de ce que l'on veut modifier et le prompt de ce que l'on veut ajouter
    file = request.files[0]
    prompt = request.form['prompt']
    prompt2 = request.form['prompt2']

    print(f"Received file {file.filename}")
    # convertie l'image en un objet PIL.Image
    file_image = PIL.Image.open(file)
    clip_image(file.filename, prompt) 

    # recupere le fichier image generer par clip
    clip_image_name = 'output/output_{}.png'.format(prompt.replace(' ', '_'))

    # On appelle la fonction fill_mask qui va remplacer les zone délimété par le prompt par le prompt2
    fill_mask(file.filename, clip_image_name, prompt2)

    # On renvoie le fichier image retouché
    return send_file("output/result.png", mimetype='image/png')


@app.route('/style', methods=['POST'])
@cross_origin()
def style():
    # On commence par supprimer les fichiers précédents dans le dossier output
    for file in os.listdir('output'):
        os.remove(os.path.join('output', file))

    # On recupere un fichier image et le nom du style que l'on veut appliquer
    file = request.files[0]
    style_name = request.form['prompt']

    # recupere le fichier style qui porte le nom de la variable style_name et qui est  dans la liste des fichier du dossier style
    for file_style in os.listdir('style'):
        if file_style == style_name+".png":
            style = file_style


    print(f"Received file {file.filename}")

    # convertie l'image en un objet PIL.Image
    file_image = PIL.Image.open(file)

    generate_img(file_image, style, "output/stylized_"+file.filename+".png")
    
    # On renvoie le fichier image stylisé
    return send_file("output/stylized_"+file.filename+".png", mimetype='image/png')


if __name__ == '__main__':
    app.run(port=5000)