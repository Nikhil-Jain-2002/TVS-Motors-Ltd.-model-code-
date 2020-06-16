from glob import glob
import sys #can be used to perform sys.exit()
import cv2
import numpy as np
import os
import yaml
import logging
import tensorflow as tf
import pandas as pd
from facenet.src import facenet
from facenet.src.align import detect_face
from logging.handlers import RotatingFileHandler
from util import create_Dir, crop_image_by_bbox, load_config, create_network_face_detection, load_image_align_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # being tried to avoid unnecessary/warning prints of Tensorfliow
tf.get_logger().setLevel('INFO') # being tried to avoid unnecessary/warning prints of Tensorfliow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #being tried to avoid unnecessary/warning prints of Tensorfliow

#Creating log file and applying basciConfig
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s ', datefmt='%m/%d/%Y %I:%M:%S %p')
file_handler = logging.FileHandler('facenet_logfile.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
handler = RotatingFileHandler('facenet_logfile.log', maxBytes = 100000, backupCount = 5)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.addHandler(handler)

config = load_config("config.yaml")
logger.debug("Config file loaded.")

if __name__ == '__main__':

    FACE_FEATURE_REQUIRED = config["FACE_FEATURE_REQUIRED"] #should be set by the user -- True/False. True/1 means Face Localization + Feature Extraction and False/0 means only Face Localization is performed
    margin = config["margin"] #add to config -- developer
    image_size = config["image_size"] #add to config -- developer --image size used to resize faces which will be passed to Facenet for face feature extraction
    BBox_Thresh = config["BBox_Thresh"] #add to config -- developer
    image_paths = config["image_paths"] #Input path
    dest_path = config["dest_path"] #Output Folder
    dest_path = create_Dir(dest_path) #create output DIR
    logger.debug("Output directory created")
    img_dest_path = create_Dir(dest_path,'Localized_Images') #create DIR to store localized images within output/Localized_Images
    discard_folder_path = create_Dir(dest_path,'Discarded_Images') #create DIR to store discarded images

    if FACE_FEATURE_REQUIRED:
        model_path = config["model_path"] #add to config --model_path: "Required for face extraction alone"
        csv_name = config["csv_name"] #Output CSV file name
        csv_dest_path = create_Dir(dest_path,'csv_output') #Create csv folder within output folder
        csv_dest_file_path = os.path.join(csv_dest_path,csv_name)

    # To perform face localize
    pnet, rnet, onet  = create_network_face_detection(gpu_memory_fraction=1.0)
    logger.info("Face localization is in process...")
    train_images, image_paths = load_image_align_data(img_dest_path,image_paths,image_size, margin, pnet, rnet, onet, discarded_folder_path = discard_folder_path, bbox_thresh = BBox_Thresh)
    logger.info("Face Localization executed successfully.")
    # To perform Facial feature extraction
    if FACE_FEATURE_REQUIRED:
        logger.info("Face Feature Extraction is in process ")
        with tf.Graph().as_default():
            with tf.Session() as sess:
                facenet.load_model(model_path)
                images_placeholder = sess.graph.get_tensor_by_name("input:0")
                embeddings = sess.graph.get_tensor_by_name("embeddings:0")
                phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
                feed_dict = {images_placeholder: train_images, phase_train_placeholder: False} #currently passing entire images as input to the model..pass it in batches and keep the batch_size at config param -- default it to 32
                train_embs = sess.run(embeddings, feed_dict = feed_dict)
                train_imgs_dict = dict(zip(image_paths, train_embs))
                df_train = pd.DataFrame.from_dict(train_imgs_dict, orient='index')
                logger.info('Face Embedded: No. of images: ' + str(len(image_paths)) + ' within ' + str(len(train_images)) + ' Localized Images')
                df_train.to_csv(csv_dest_file_path) #output CSV files -- {img_names,features}
                logger.info("Face Feature Extraction executed successfully.")
logger.info("Path of output folder is: " + dest_path)
