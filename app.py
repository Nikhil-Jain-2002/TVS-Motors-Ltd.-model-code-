from glob import glob
import sys #can be used to perform sys.exit()
import cv2
import numpy as np
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import yaml
import logging
import tensorflow as tf
import pandas as pd
from facenet.src import facenet
from facenet.src.align import detect_face
from logging.handlers import TimedRotatingFileHandler
import logging.config
from util import create_Dir, crop_image_by_bbox, load_config, create_network_face_detection, load_image_align_data, load_log_config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # being tried to avoid unnecessary/warning prints of Tensorfliow
tf.get_logger().setLevel('INFO') # being tried to avoid unnecessary/warning prints of Tensorfliow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #being tried to avoid unnecessary/warning prints of Tensorfliow

#Loading log_config file
log_config = load_log_config("logging.yaml")
#Defining logger
logger = logging.getLogger(__name__)
#Loading config file
config = load_config("config.yaml")


from serve import get_model_api


# define the app
app = Flask(__name__)
CORS(app) # needed for cross-domain requests, allow everything by default


# logging for heroku
if 'DYNO' in os.environ:
    app.logger.addHandler(logging.StreamHandler(sys.stdout))
    app.logger.setLevel(logging.INFO)


# load the model
model_api = get_model_api()


# API route
@app.route('/api', methods=['POST'])
def api():
    """API function
    All model-specific logic to be defined in the get_model_api()
    function
    """
    input_data = request.json
    app.logger.info("api_input: " + str(input_data))
    output_data = model_api(input_data)
    app.logger.info("api_output: " + str(output_data))
    response = jsonify(output_data)
    return response


@app.route('/')
def index():
    return "Index API"

# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally.
    app.run(host='0.0.0.0', debug=True)
