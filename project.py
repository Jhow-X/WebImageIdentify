from flask import Flask, flash, request, redirect, url_for
import os
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

UPLOAD_FOLDER = './upload'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg',}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                model = load_model('keras_model.h5')
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                image = Image.open('./upload/'+file.filename)
                size = (224, 224)
                image = ImageOps.fit(image, size, Image.ANTIALIAS)
                image_array = np.asarray(image)
                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                data[0] = normalized_image_array
                prediction = model.predict(data)
                probab = []
                x = np.argmax(prediction)
                for i in prediction:
                    for j in i:
                        probab.append(j)
                if(x == 0):
                    return f'Covid com {probab[0]*100:.2f}% de certeza'
                elif(x == 1):
                    return f'Normal com {probab[1]*100:.2f}% de certeza'
                else:
                    return f'Pneumonia com {probab[2]*100:.2f}% de certeza'
    return '''
    <!doctype html>
    <title>Covid data view</title>
    <h1>Selecione o arquivo</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''