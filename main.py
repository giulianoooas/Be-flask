from flask import Flask

app = Flask(__name__)

@app.route('/predict-price')
def predictPrice():
    return 1

if __name__ == '__main__':
    app.run()
