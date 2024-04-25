from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np

model = load_model('my_model.h5')
vector = list()

def reader(image_path:str)-> vector:
    img = Image.open(io.BytesIO(image_path)).convert('L')
    imgR = img.resize((28,28))
    img_array = np.array(imgR)
    img_array = img_array.astype('float32') / 255
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error":"no file"})
        try:
            image = file.read()
            imageArray = reader(image)
            predictions = model.predict(imageArray)
            predicted_class = np.argmax(predictions, axis=1)
            data = {"prediction":int(predicted_class)}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error":str(e)})
    return "OK"

if __name__ == "__main__":
    app.run(debug=True)
