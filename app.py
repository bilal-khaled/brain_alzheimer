from flask import Flask, render_template, request ,url_for, redirect
import joblib
import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.config['STATIC_URL'] = '/static'

# Load the trained models
model_ML = joblib.load('models/model_1.pkl')
model = tf.keras.models.load_model('models/model.h5')

@app.route('/')
def index():
    return render_template('start.html')

@app.route('/page')
def show_page():
    return render_template('page.html')

@app.route('/machine', methods=['GET', 'POST'])
def show_machine():
    if request.method == 'POST':
        # Get the form data
        gender = request.form['gender']
        age = int(request.form['age'])
        ses = int(request.form['ses'])
        mmse = int(request.form['mmse'])
        nwbv = float(request.form['nwbv'])


        # Convert gender to numerical representation
        gender_mapping = {'1': 0, '0': 1}  # Update gender_mapping
        gender_encoded = gender_mapping.get(gender, -1)  # Assign -1 for unknown values

        # Create a DataFrame with the input data
        data = pd.DataFrame({
            'M/F': [gender_encoded],
            'Age': [age],
            'SES': [ses],
            'MMSE': [mmse],
            'nWBV': [nwbv],
        })

        # Perform classification using the machine learning model
        prediction = model_ML.predict(data)[0]

        # Convert the prediction to a meaningful label
        if prediction == 0:
            label = 'Non-Demented'
        elif prediction == 1:
            label = 'Demented'
        else:
            label = 'Converted'

        # Render the template with the prediction result
        # Render the template with the prediction result
        return render_template('machine.html', prediction=label)

    # For GET requests, clear the prediction result
    return render_template('machine.html')

@app.route('/deep')
def show_deep():
    return render_template('deep.html')
@app.route('/page2')
def show_page2():
    return render_template('page2.html')
@app.route('/classify', methods=['POST'])
def classify():
    # Get the uploaded image file from the request
    image_file = request.files['imageUpload']
    if image_file:
        # Create the 'uploads' folder if it doesn't exist
        os.makedirs('static/uploads', exist_ok=True)

        # Save the uploaded image to the 'uploads' folder
        image_path = os.path.join('static', 'uploads', image_file.filename)
        image_file.save(image_path)
        image = Image.open(image_file)
        # convert image into RGB
        image = image.convert('RGB')

        # Resize the image to match the input size of the model
        image = image.resize((128, 128))

        # Convert the image to an array
        image_array = np.array(image)

        # Expand the dimensions of the array to make it 4-dimensional
        image_array = np.expand_dims(image_array, axis=0)

        # Normalize the pixel values between 0 and 1
        image_array = image_array / 255.0

        class_labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

        # Perform model prediction
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction)
        predicted_label = class_labels[predicted_class]

        # Pass the predicted label and image filename to classify.html
        return redirect(url_for('show_result', predicted_class=predicted_label, image_filename=image_file.filename))
    else:
        flash('Please upload an image.')
        return redirect(request.url)

@app.route('/result')
def show_result():
    predicted_class = request.args.get('predicted_class')
    image_filename = request.args.get('image_filename')
    image_path = f'uploads/{os.path.basename(image_filename)}'  # Update the path to match your folder structure
    return render_template('classify.html', predicted_class=predicted_class, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
