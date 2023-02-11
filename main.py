from flask import Flask, jsonify
import os

app = Flask(__name__)


@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})


# Route for seeing a data
@app.route('/data')
def get_time():
    # Returning an api for showing in  reactjs
    return {
        'Name':"geek", 
        "Age":"22",
        "Date":"x", 
        "programming":"python"
        }

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
