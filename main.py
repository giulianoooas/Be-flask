from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('predict-price')
def predictPrice():
    return 1

if __name__ == '__main__':
    app.run()
