import PIL
from flask import Flask, request
from flask_cors import CORS, cross_origin

from partie1.Modele import Modele
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app = Flask(__name__)

# créer un modèle
model = Modele()

Liste_ok = []
Liste_retoucher = []
Liste_exclure = []


@app.route('/upload', methods=['POST'])
@cross_origin()
def upload():
    # vide les listes
    Liste_ok.clear()
    Liste_retoucher.clear()
    Liste_exclure.clear()

    files = request.files
    print(files)
    for file in files.values():
        print(f"Received file {file.filename}")
        # convertie l'image en un objet PIL.Image
        file_image = PIL.Image.open(file)
        resultat = model.processImage(file_image)

        if resultat == "ok":
            Liste_ok.append(file.filename)
        elif resultat == "retoucher":
            Liste_retoucher.append(file.filename)
        elif resultat == "exclure":
            Liste_exclure.append(file.filename)


    return "Files uploaded successfully"

if __name__ == '__main__':
    app.run(port=5000)