from flask import Flask, request, render_template
from PIL import Image
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image


app = Flask(__name__)


#model = pickle.load(open('models/crop_disease_model.pkl', 'rb'))


def preprocess_image(img):
    img = img.resize((224, 224)) 
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0 
    return img

def predict_disease(img):
    preprocessed_img = preprocess_image(img)
    # prediction = model.predict(preprocessed_img)
    # predicted_disease = np.argmax(prediction)
    predicted_disease = "Placeholder Disease"
    return predicted_disease


#Routes
@app.route('/')
def home():
    return render_template("upload.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/future')
def future_features():
    return render_template("future_features.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' in request.files:
        file = request.files['image']
        img = Image.open(file)
        predicted_disease = predict_disease(img)
        return render_template('upload.html', predicted_disease=predicted_disease)
    else:
        return render_template("upload.html", message="No image uploaded.")


# Main
if __name__ == '__main__':
    app.run(debug=True)