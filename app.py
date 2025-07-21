from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Set input size based on scaler
INPUT_SIZE = scaler.mean_.shape[0]

# Define the model architecture exactly as used during training
model = nn.Sequential(
    nn.Linear(INPUT_SIZE, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# Load the trained weights
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set model to evaluation mode

@app.route('/')
def home():
    return "House Price Prediction API is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)  # Example: [1200, 3, 2, ...]
        
        # Preprocess: scale the features
        scaled = scaler.transform(features)
        
        # Convert to PyTorch tensor
        input_tensor = torch.tensor(scaled, dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor).item()
        
        return jsonify({"predicted_price": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
# Save the model architecture and weights
# torch.save(model.state_dict(), 'model.pth')
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)