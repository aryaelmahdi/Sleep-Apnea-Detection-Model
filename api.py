from flask import Flask, jsonify, request
from flask_cors import CORS
import model
import io
import os
import response
import tempfile

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'edf', 'csv'}
CORS(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/")
def home():
    return jsonify(message="Welcome to Sleep Well")

@app.route("/detect/cnn", methods=['POST'])
def detect_cnn():
    file = request.files["file"]
    if file and allowed_file(file.filename):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as temp_file:
            file.save(temp_file.name)
            temp_filename = temp_file.name

        try:
            model_path = "model\cnn\CNN 801010.h5"

            processed_data = model.preprocess_cnn(temp_filename) 
            prediction = model.predict_cnn(processed_data, model_path)

            return jsonify(response.ModelResponse(200, "Prediction Success", prediction).to_dict())
        finally:
            os.remove(temp_filename)
    else:
        return jsonify(response.ExtensionNotAllowed().to_dict())

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8081)
