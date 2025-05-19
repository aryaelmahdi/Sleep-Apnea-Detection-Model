from flask import Flask, jsonify, request
from flask_cors import CORS
import model
import io
import os
import response
import tempfile
import extraction
import platform

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'edf', 'csv', 'txt'}
CORS(app)
models = ["CNN", "LSTM"]

def set_memory_limit(max_mb=4096):
    if platform.system() != "Windows":
        import resource

        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (max_mb * 1024 * 1024, hard))
        print(f"Memory limit set to {max_mb} MB")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/")
def home():
    return jsonify(message="Welcome to Sleep Well")

@app.route("/detect", methods=['POST'])
def detect():
    file = request.files.get("file")
    annotation = request.files.get("annotation")
    model_type = request.form.get("model")
    model_type = model_type.upper() if model_type else models[0]
    if file and allowed_file(file.filename):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as temp_file:
            file.save(temp_file.name)
            temp_filename = temp_file.name

        try:
            fft = model_type.upper() == models[1]
            model_path = f"model/{model_type.upper()}.h5"
            if model_type not in models:
                return jsonify(response.BadRequest(error="Model Does Not Exists"))
            
            annotated = annotation != None
            if annotated and annotation and allowed_file(annotation.filename):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_anno:
                        annotation.save(temp_anno.name)
                        annotation_filename = temp_anno.name
                try:
                    segments = extraction.find_s2_notation(annotation_filename)

                    apnea = 0 
                    total = 0
                    for start_time, end_time in segments:
                        processed_data, data = model.preprocess_with_trim(temp_filename, fft, start_time, end_time)
                        prediction = model.predict(processed_data, model_path)
                        if "apnea" in prediction:
                            apnea += 1
                            total += 1
                        else:
                            total += 1
                        os.remove(data)
                        
                    score = 0
                    if total and apnea != 0:
                        score = (apnea/total)*100
                    result = f"Apnea Score: {score:.2f}%"
                    return jsonify(response.ModelResponse(200, "Prediction Success", result).to_dict())
                finally:
                    try:
                        os.remove(annotation_filename)
                    except PermissionError as e:
                        print(f"Error Deleteing File: {e}")
            else:
                processed_data = model.preprocess(temp_filename, fft) 
                prediction = model.predict(processed_data, model_path)

                return jsonify(response.ModelResponse(200, "Prediction Success", prediction).to_dict())
        finally:
            os.remove(temp_filename)
    else:
        return jsonify(response.BadRequest(error="File Not Supported").to_dict())

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8081)
    set_memory_limit()