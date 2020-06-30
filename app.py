from glob import glob
import sys #can be used to perform sys.exit()
import cv2
import numpy as np
import os
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory, flash
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
from face_localize_feature_extract import main_func
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # being tried to avoid unnecessary/warning prints of Tensorfliow
tf.get_logger().setLevel('INFO') # being tried to avoid unnecessary/warning prints of Tensorfliow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #being tried to avoid unnecessary/warning prints of Tensorfliow

#Loading log_config file
log_config = load_log_config("logging.yaml")
#Defining logger
logger = logging.getLogger(__name__)
#Loading config file
config = load_config("config.yaml")


# define the app
app = Flask(__name__)
CORS(app) # needed for cross-domain requests, allow everything by default
face_feature_req = True

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/featureExtraction", methods=["GET"])
def featureExtraction():
    face_extract = request.args.get('face_extract')
    if face_extract == "yes":
        face_feature_req = True
    elif face_extract == "no":
        face_feature_req = False
    else:
        bad_request_given()
    output_path = main_func(face_feature_req)
    #output_path = output_path_folder + '\\Localized_Images'
    shutil.rmtree("C:\\Users\\jainn\\Desktop\\Git\\Face_Localize_Feature_Extract\\uploads")
    #shutil.rmtree("C:\\Users\\jainn\\Desktop\\Git\\Face_Localize_Feature_Extract\\output")
    return render_template("client.html")
    #return send_from_directory(output_path, filename, as_attachment=True)  


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        create_Dir("C:\\Users\\jainn\\Desktop\\Git\\Face_Localize_Feature_Extract\\", "uploads")
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, "uploads")
        #filename = secure_filename(f.filename)
        f.save(file_path, secure_filename(f.filename))
    return render_template("complete.html") 

# HTTP Errors handlers
@app.errorhandler(400)
def bad_request_given():
    return """
    Please check the input (Y/N or y/n)
    """, 400


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
    #This is used when running locally.
    app.run(debug=True)




