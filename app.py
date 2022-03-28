from flask import Flask, request
from flask_cors import CORS
import os
from mlModel import MlModel

# labels : 0 -> bad, 1 = good

app = Flask(__name__, instance_relative_config=True)
CORS(app)

@app.route('/comment-status')
def predictPrice():
    comment = request.args['comment']
    return str(mlModel.predict(comment)), 200

if __name__ == '__main__':
    os.environ['FLASK_ENV'] = 'development'
    mlModel = MlModel()
    app.run(debug=True)
