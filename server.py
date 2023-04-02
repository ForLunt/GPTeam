import PIL
from flask import Flask, Response, request
from flask_cors import CORS, cross_origin
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

if __name__ == '__main__':
    app.run(port=5000)