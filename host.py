import tensorflow as tf
from tensorflow import keras
from flask import Flask, jsonify, request
import numpy as np

app = Flask(__name__)

def preprocessing(imgarr):
    return np.asarray(imgarr).reshape([-1,32,32,3])

@app.route("/", methods = ['GET'])
def helloWorld():
    return "Hello World!"

@app.route("/evaluate", methods = ['POST'])
def predict():
    # TODO: change archtuture so that model is not loaded on evey predict call
    #this currenly is very slow
    model = tf.keras.models.load_model('model/mymodel.h5')
    payload = request.get_json()
    x = model.predict(preprocessing(payload['data']))
    print(x)
    return jsonify({'data' : x.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0")
