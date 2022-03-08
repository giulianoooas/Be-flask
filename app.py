from flask import Flask, request
import os
from mlModel import NeuralNetworkModel

# labels : 0 -> bad, 1 = good

app = Flask(__name__, instance_relative_config=True)

@app.route('/comment-status')
def predictPrice():
    comment = request.json['comment']
    return str(mlModel.predict(comment)), 200

if __name__ == '__main__':
    os.environ['FLASK_ENV'] = 'development'
    mlModel = NeuralNetworkModel()
    app.run(debug=True)
