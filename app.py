from flask import Flask
import os

app = Flask(__name__, instance_relative_config=True)

@app.route('/predict-price')
def predictPrice():
    return "1", 200

if __name__ == '__main__':
    os.environ['FLASK_ENV'] = 'development'
    app.run(debug=True)
