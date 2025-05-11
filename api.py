from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'edf', 'csv'}
CORS(app)

@app.route("/")
def home():
    return jsonify(message="Welcome to Sleep Well")

if __name__ == "__main__":
    app.run(debug=True)
