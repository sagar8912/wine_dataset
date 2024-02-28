from flask import Flask, render_template, request
from utils import load_model, predict
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = load_model('wine_model.pkl')

# Define the range for each feature
feature_ranges = {
    'alcohol': (10, 60),
    'malic_acid': (0.1, 10),
    'ash': (0.1, 10),
    'alcalinity_of_ash': (10, 100),
    'magnesium': (10, 200),
    'total_phenols': (0.1, 10),
    'flavanoids': (0.1, 10),
    'nonflavanoid_phenols': (0.01, 2),
    'proanthocyanins': (0.1, 5),
    'color_intensity': (0.1, 50),
    'hue': (0.1, 5),
    'od280/od315_of_diluted_wines': (0.1, 10),
    'proline': (100, 3000)
}

@app.route('/')
def home():
    return render_template('index.html', feature_ranges=feature_ranges)

@app.route('/predict', methods=['POST'])
def predict_result():
    features = []
    for feature_name in feature_ranges.keys():
        feature_value = request.form.get(feature_name)
        if feature_value is None:
            return "Error: Missing feature value"
        try:
            feature_value = float(feature_value)
        except ValueError:
            return "Error: Invalid input for {}".format(feature_name)
        min_val, max_val = feature_ranges[feature_name]
        if not (min_val <= feature_value <= max_val):
            return "Error: {} must be between {} and {}".format(feature_name, min_val, max_val)
        features.append(feature_value)
    
    prediction = predict(model, np.array([features]))
    # Pass the prediction result to the result.html template
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)




