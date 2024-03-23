from flask import Flask, request, jsonify
import requests
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from io import BytesIO
import tensorflow as tf

app = Flask(__name__)

# Load pre-trained VGG16 model without the top classification layers
vgg_model = VGG16(weights='imagenet', include_top=False)

# Load label encoder
labels = np.load("labels.npy")
label_encoder = LabelEncoder()
encoder_data = label_encoder.fit_transform(labels)

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

@app.route('/')
def home():
    return "Hello World"

# Function to extract features using VGG16 model
def extract_features(img_url):
    try:
        response = requests.get(img_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        img = Image.open(BytesIO(response.content))
        img = img.resize((224, 224))  # Resize the image to match the input size expected by VGG16
        img_data = np.array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = vgg_model.predict(img_data)
        return features.flatten()
    except Exception as e:
        print("Error processing image:", e)
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image URL from the request
        img_url = request.json['image_url']

        # Extract features from the image
        test_features = extract_features(img_url)

        if test_features is not None:
            # Reshape the extracted features to match the input shape expected by your model
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            test_features_reshaped = test_features.reshape(input_details[0]['shape'])
            interpreter.set_tensor(input_details[0]['index'], test_features_reshaped)

            # Run the model
            interpreter.invoke()

            # Get the output tensor
            predictions = interpreter.get_tensor(output_details[0]['index'])
            predicted_class_encoded = label_encoder.inverse_transform([np.argmax(predictions)])

            # Create a JSON response
            response = {'prediction': predicted_class_encoded[0]}
        else:
            response = {'error': 'Error processing image'}

    except Exception as e:
        response = {'error': str(e)}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
