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
from logging.handlers import TimedRotatingFileHandler
import logging.config
import shutil
from util import create_Dir, crop_image_by_bbox, load_config, create_network_face_detection, load_image_align_data, load_log_config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # being tried to avoid unnecessary/warning prints of Tensorfliow
tf.get_logger().setLevel('INFO') # being tried to avoid unnecessary/warning prints of Tensorfliow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #being tried to avoid unnecessary/warning prints of Tensorfliow

#Loading log_config file
log_config = load_log_config("logging.yaml")
#Defining logger
logger = logging.getLogger(__name__)
logger.debug("The program has been started...please wait.")
logger.debug("Log config file loaded successfully.")

#Loading config file
config = load_config("config.yaml")
logger.debug("Config file loaded successfully.")

def main_func(face_feature_req):
    FACE_FEATURE_REQUIRED = face_feature_req #should be set by the user -- True/False. True/1 means Face Localization + Feature Extraction and False/0 means only Face Localization is performed
    batch_size = config["batch_size"] #batch_size = 32, user param in config
    margin = config["margin"]         #add to config -- developer
    image_size = config["image_size"] #add to config -- developer --image size used to resize faces which will be passed to Facenet for face feature extraction
    BBox_Thresh = config["BBox_Thresh"] #add to config -- developer
    image_paths = config["image_paths"] #Input path
    dest_path = config["dest_path"] #Output Folder
    dest_path = create_Dir(dest_path) #create output DIR
    logger.debug("Output directory created.")
    img_dest_path = create_Dir(dest_path,'Localized_Images') #create DIR to store localized images within output/Localized_Images
    discard_folder_path = create_Dir(dest_path,'Discarded_Images') #create DIR to store discarded images

    if FACE_FEATURE_REQUIRED:
        model_path = config["model_path"] #add to config --model_path: "Required for face extraction alone"
        csv_name = config["csv_name"] #Output CSV file name
        csv_dest_path = create_Dir(dest_path,'csv_output') #Create csv folder within output folder
        csv_dest_file_path = os.path.join(csv_dest_path,csv_name)

    # To perform face localize
    logger.info("Face localization is in process...")
    pnet, rnet, onet  = create_network_face_detection(gpu_memory_fraction=1.0)
    train_images, image_paths = load_image_align_data(img_dest_path, image_paths, image_size, margin, pnet, rnet, onet, discarded_folder_path = discard_folder_path, bbox_thresh = BBox_Thresh)
    logger.info("Face Localization executed successfully.")

    # To perform Facial feature extraction
    if FACE_FEATURE_REQUIRED:
        logger.info("Face Feature Extraction is in process...")
        temp_tr_images, temp_image_paths = [], []  # temp vars required for batch process
        list_image_paths, list_train_embs = [], []  # to collate into a single list post batch process
        with tf.Graph().as_default():
            with tf.Session() as sess:
                facenet.load_model(model_path)
                images_placeholder = sess.graph.get_tensor_by_name("input:0")
                embeddings = sess.graph.get_tensor_by_name("embeddings:0")
                phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
                bt_sz = batch_size
                logger.debug("Face Feature Extraction model's batch size is set to " + str(batch_size))
                for i in range(0, len(train_images), bt_sz):
                    temp_tr_images = train_images[i : i+bt_sz]
                    temp_image_paths = image_paths[i : i+bt_sz]
                    feed_dict = {images_placeholder: temp_tr_images, phase_train_placeholder: False}
                    logging.debug('len(temp_tr_images): ' + str(len(temp_tr_images)))
                    train_embs = sess.run(embeddings, feed_dict=feed_dict)
                    list_train_embs.extend(train_embs)
                    list_image_paths.extend(temp_image_paths)
                embs_dict = dict(zip(list_image_paths, list_train_embs))
                df_train = pd.DataFrame.from_dict(embs_dict, orient='index')
                logger.debug('Face Embedded: No. of images: ' + str(len(image_paths)) + ' within ' + str(len(train_images)) + ' Localized Images')
                df_train.to_csv(csv_dest_file_path)  # output CSV files -- {img_names,features}
                logger.info("Face Feature Extraction executed successfully.")

    logger.info("Path of output folder is: " + dest_path)
    return dest_path

if __name__ == '__main__':
    main_func()
    