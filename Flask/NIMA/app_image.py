from email.headerregistry import ContentTypeHeader
from urllib import response
import numpy as np
import os
import io
from flask import Flask, request, make_response, send_file
from PIL import Image, ImageFile
from io import BytesIO
from time import time
import tensorflow as tf
import importlib
#import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)
nima = None


def load_model(base_model_name, weights_file):
    # build model and load weights
    global nima, model
    nima = Nima(base_model_name, n_classes=5, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)
    X = np.empty((1, 224, 224, 3))
    nima.nima_model.predict(X)


@app.route('/sample', methods=['POST'])
def segment():
    start1 = time()
    img = Image.open(BytesIO(request.get_data().split(b'\r\n--')[1].split(b'\r\n\r\n')[1])).convert('RGB')
    
    with io.BytesIO() as output:
        img.save(output,format="JPEG")
        contents = output.getvalue() #バイナリ取得
        #print(contents)#表示
        img2 = Image.open(io.BytesIO(contents)) #バイナリから画像に変換
        img2.save('photo.jpg')
    
    img_load_dims=(224, 224)
    X = np.empty((1, *img_load_dims, 3))
    img = load_image('photo.jpg', img_load_dims)
    if img is not None:
        X[0, ] = img

    X = nima.preprocessing_function()(X)
    predictions = nima.nima_model.predict(X)
    score = mean_score(predictions)
    print('score:', score)
    if score < 3.5:
        response = '0'
    else:
        response = '1'

    time_diff = time() - start1
    print('time_diff:', time_diff)
    return response


def load_image(img_file, target_size):
    return np.asarray(tf.keras.preprocessing.image.load_img(img_file, target_size=target_size))


def mean_score(score_dist):
    score_dist = np.array(score_dist) / np.array(score_dist).sum()
    return (score_dist*np.arange(1, 6)).sum()

# def predict(model, data_generator):
#     return model.predict(data_generator, workers=1, use_multiprocessing=False)

def earth_movers_distance(y_true, y_pred):
    cdf_true = K.cumsum(y_true, axis=-1)
    cdf_pred = K.cumsum(y_pred, axis=-1)
    emd = K.sqrt(K.mean(K.square(cdf_true - cdf_pred), axis=-1))
    return K.mean(emd)


class Nima:
    def __init__(self, base_model_name, n_classes=10, learning_rate=0.001, dropout_rate=0, loss=earth_movers_distance,
                 decay=0, weights='imagenet'):
        self.n_classes = n_classes
        print('n_classes:{0}'.format(self.n_classes))
        self.base_model_name = base_model_name
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.loss = loss
        self.decay = decay
        self.weights = weights
        self._get_base_module()

    def _get_base_module(self):
        # import Keras base model module
        if self.base_model_name == 'InceptionV3':
            self.base_module = importlib.import_module('tensorflow.keras.applications.inception_v3')
        elif self.base_model_name == 'InceptionResNetV2':
            self.base_module = importlib.import_module('tensorflow.keras.applications.inception_resnet_v2')
        else:
            self.base_module = importlib.import_module('tensorflow.keras.applications.'+self.base_model_name.lower())

    def build(self):
        # get base model class
        print('n_classes2:{0}'.format(self.n_classes))
        BaseCnn = getattr(self.base_module, self.base_model_name)

        # load pre-trained model
        self.base_model = BaseCnn(input_shape=(224, 224, 3), weights=self.weights, include_top=False, pooling='avg')

        # add dropout and dense layer
        print('n_classes: {0}'.format(self.n_classes))
        x = Dropout(self.dropout_rate)(self.base_model.output)
        x = Dense(units=self.n_classes, activation='softmax')(x)

        self.nima_model = Model(self.base_model.inputs, x)

    def compile(self):
        self.nima_model.compile(optimizer=Adam(lr=self.learning_rate, decay=self.decay), loss=self.loss)

    def preprocessing_function(self):
        return self.base_module.preprocess_input


if __name__ == "__main__":
    weights_file = os.path.join(app.static_folder, 'weights_mobilenet_35_0.218.hdf5')
    load_model("MobileNet", weights_file)
    print(" * Flask starting server...")
    app.run(host='0.0.0.0', port=5001)

