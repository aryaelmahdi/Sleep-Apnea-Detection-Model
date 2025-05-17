from flask import Flask, jsonify, request
from flask_cors import CORS
import model
import io
import os
import response
import tempfile

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'edf', 'csv', 'txt'}
CORS(app)
models = ["CNN", "LSTM"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/")
def home():
    return jsonify(message="Welcome to Sleep Well")

@app.route("/detect", methods=['POST'])
def detect():
    file = request.files["file"]
    model_type = request.form.get("model") or models[0]
    if file and allowed_file(file.filename):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as temp_file:
            file.save(temp_file.name)
            temp_filename = temp_file.name

        try:
            fft = model_type.upper() == models[1]
            model_path = f"model/{model_type.upper()}.h5"
            if model_path not in models:
                return jsonify(response.BadRequest(error="Model Does Not Exists"))

            processed_data = model.preprocess(temp_filename, fft) 
            prediction = model.predict(processed_data, model_path)

            return jsonify(response.ModelResponse(200, "Prediction Success", prediction).to_dict())
        finally:
            os.remove(temp_filename)
    else:
        return jsonify(response.BadRequest(error="File Not Supported").to_dict())

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8081)
