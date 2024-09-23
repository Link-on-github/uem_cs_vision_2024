# from flask import Flask, render_template, request, redirect, url_for, flash
# import os
# from werkzeug.utils import secure_filename
# import torch
# from torchvision import transforms
# from PIL import Image
# import numpy as np

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads/'
# app.secret_key = 'your_secret_key'

# # Load your trained PyTorch model and map it to the CPU
# model = torch.load('/Users/arneshbanerjee/Coding/Code/python/uem_csvision/models/wheat_disease_model_further_trained1.pth', map_location=torch.device('cpu'))
# model.eval()  # Set the model to evaluation mode

# # Define the disease categories
# disease_labels = ['Apple Scab', 'Apple Black Rot', 'Apple Cedar Rust', 'Healthy']

# # Image transformations (adjust as per your model requirements)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize values can differ
# ])

# def predict_disease(image_path):
#     img = Image.open(image_path).convert('RGB')  # Ensure image is in RGB
#     img = transform(img).unsqueeze(0)  # Add batch dimension

#     with torch.no_grad():
#         prediction = model(img)
    
#     # Convert the output to probabilities and get the predicted class
#     probabilities = torch.nn.functional.softmax(prediction, dim=1)
#     predicted_class = torch.argmax(probabilities, dim=1).item()

#     return disease_labels[predicted_class]


# @app.route('/')
# def home():
#     return render_template('index.html')


# @app.route('/upload', methods=['GET', 'POST'])
# def upload_image():
#     if request.method == 'POST':
#         if 'image' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['image']
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file:
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(file_path)

#             # Predict the disease using the uploaded image
#             prediction = predict_disease(file_path)

#             return render_template('result.html', prediction=prediction, image_url=file_path)

#     return render_template('upload.html')


# @app.route('/about')
# def about():
#     return render_template('about.html')


# @app.route('/future-features')
# def future_features():
#     return render_template('future_features.html')


# if __name__ == '__main__':
#     app.run(debug=True)





























from flask import Flask, render_template, request, redirect, url_for
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
import os
import google.generativeai as genai

app = Flask(__name__)

# Load the pre-trained model
model_path = r"/Users/arneshbanerjee/Coding/Code/python/uem_csvision/models/Wheat_Disease_Detection.keras"
model = tf.keras.models.load_model(model_path)

# Configure Google Gemini API
genai.configure(api_key="AIzaSyBN1mT6DqwL3niuYTS7HuK4e_VYNvxQ7j8")

# Define disease class names
class_folders = ['Aphid', 'Black Rust', 'Blast', 'Brown Rust', 'Fusarium Head Blight', 
                 'Healthy Wheat', 'Leaf Blight', 'Leaf Rust', 'Mildew', 'Mite', 
                 'Septoria', 'Smut', 'Stem fly', 'Tan spot', 'Yellow Rust']

# Function to generate disease information from Gemini API
def response_generator(disease_name):
    model = genai.GenerativeModel('gemini-pro')
    full_prompt = f"Assume you are a helper to the farmers. Now The Wheat is probably having this disease: {disease_name}. Please suggest the farmer what to do about this in details. Help him further to say what are the concerns to take about this."
    response = model.generate_content(full_prompt)
    return response.text

@app.route('/')
def landing_page():
    return render_template('landing.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        # Get image file from form
        image = request.files['image']
        image_path = os.path.join('static', image.filename)
        image.save(image_path)
        
        # Load and preprocess image
        img = load_img(image_path, target_size=(255, 255))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Predict disease
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        disease_name = class_folders[predicted_class_index]
        
        # Get detailed disease information using Gemini
        disease_info = response_generator(disease_name)
        
        # Pass data to template to display
        return render_template('result.html', image_url=image_path, disease_name=disease_name, disease_info=disease_info)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)