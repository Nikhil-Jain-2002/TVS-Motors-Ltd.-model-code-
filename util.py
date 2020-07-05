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
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # being tried to avoid unnecessary/warning prints of Tensorfliow
tf.get_logger().setLevel('INFO') # being tried to avoid unnecessary/warning prints of Tensorfliow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # being tried to avoid unnecessary/warning prints of Tensorfliow

CONFIG_PATH = os.getcwd()
# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config
config = load_config("config.yaml")

#Function to load log config file
def load_log_config(config_fname):
   with open(os.path.join(CONFIG_PATH, config_fname), 'r') as f:
      log_config = yaml.safe_load(f.read())
   return logging.config.dictConfig(log_config)
log_config = load_log_config("logging.yaml")
#Defining Logger
logger = logging.getLogger(__name__)

# Method to perform cropping the images using bounding box info.
def crop_image_by_bbox(image, bbox, img_size, margin):
   bb = np.zeros(4, dtype=np.int32)
   bb[0] = np.maximum(bbox[0] - margin / 2, 0)
   bb[1] = np.maximum(bbox[1] - margin / 2, 0)
   bb[2] = np.minimum(bbox[2] + margin / 2, img_size[1])
   bb[3] = np.minimum(bbox[3] + margin / 2, img_size[0])
   cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
   return cropped, bb

#Create directory
def create_Dir(folder_path, folder_name=''):
   if folder_name is not None:
      folder_path = os.path.join(folder_path, folder_name)
   if not os.path.exists(folder_path):
      os.makedirs(folder_path)
   return folder_path

# Create model to perform localization -- MTCNN
def create_network_face_detection(gpu_memory_fraction):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet

#Face Localization
def load_image_align_data(dest_path, image_paths, image_size, margin, pnet, rnet, onet, discarded_folder_path='',
                          bbox_thresh=0.95):
   minsize = config["minsize"]  # minimum size of face
   threshold = config["threshold"]  # three steps's threshold
   factor = config["factor"]  # scale factor
   image_list, image_names = [], []
   discared_image_cnt = 0
   img_files = glob(image_paths + '*.png')  # Incase glob doesn't work in Windows environment replace it with 'os' library module. #readme
   img_files.extend(glob(image_paths + '*.jpg'))
   img_files = sorted(img_files)
   logging.info('Total images read are: ' + str(len(img_files)))
   if len(img_files) == 0:
      logger.error("Total images read are 0. Please input images. ")
      sys.exit()
   for files in img_files:
      img = cv2.imread(files, 1)
      if img is not None:
         image_list.append(img)
   image_files = img_files.copy()
   recog_image_paths, img_list = [], []
   fnames_list = []
   cropped_big_face = dest_path + '_cropped_face\\'  # Dir to store cropped faces
   cropped_big_face = create_Dir(cropped_big_face)

   for img_path, x in zip(image_files, range(len(image_list))):
      fname = os.path.basename(img_path)
      dest_fname = os.path.join(dest_path, fname)
      img_size = np.asarray(image_list[x].shape)[0:2]
      img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # chk img.shape and log filename if its empty
      bounding_boxes, _ = detect_face.detect_face(
         image_list[x], minsize, pnet, rnet, onet, threshold, factor)
      nrof_samples = len(bounding_boxes)
      r_cnt = 0
      img_temp = image_list[x].copy()
      while (nrof_samples == 0 and r_cnt < 3):
         image_list[x] = cv2.rotate(image_list[x], cv2.ROTATE_90_CLOCKWISE)
         bounding_boxes, _ = detect_face.detect_face(
            image_list[x], minsize, pnet, rnet, onet, threshold, factor)
         nrof_samples = len(bounding_boxes)
         r_cnt += 1
      if nrof_samples > 0:
         if r_cnt == 0:
            # cv2.imwrite(os.path.join(dest_path,fname),img)
            pass
         # perform image rotation of degrees: [90,180,270] iff faces aren't recognized
         elif r_cnt == 1:
            rot_angle = cv2.ROTATE_90_CLOCKWISE
         elif r_cnt == 2:
            rot_angle = cv2.ROTATE_180
         elif r_cnt == 3:
            rot_angle = cv2.ROTATE_90_COUNTERCLOCKWISE

         if r_cnt > 0:
            image_list[x] = cv2.rotate(img_temp, rot_angle)
         else:
            image_list[x] = img_temp
         big_area = -1;
         big_face_no = -1  # param used for finding the bigger face within the image
         img_size = np.asarray(image_list[x].shape)[0:2]
         for i in range(nrof_samples):
            if bounding_boxes[i][4] > bbox_thresh:
               img_name = fname  # img_path
               det = np.squeeze(bounding_boxes[i, 0:4])
               cropped, bb = crop_image_by_bbox(image_list[x], det, img_size, margin)
               x1, y1, x2, y2 = bb
               area_ratio = (x2 - x1) * (y2 - y1) / (np.prod(img_size))
               if area_ratio > big_area:
                  big_area = area_ratio
                  big_face_no = i

               # cv2.rectangle(image_list[x], (x1, y1), (x2, y2), (0,0,255), 3) #comment -- to remove drawing bounding box on all faces detected

         # cv2.imwrite(dest_fname,image_list[x]) #comment -- to remove drawing bounding box on all faces detected
         if big_face_no < 0:
            continue
         else:  # indirectly checks bounding_boxes[i][4] > 0.95
            det = np.squeeze(bounding_boxes[big_face_no, 0:4])
            confd = str(round(bounding_boxes[big_face_no][4], 3))
            logger.debug('confidence score of ' + img_name + " is: " + confd)  # print in log: confidence score of big face detected and localized.
            cropped, bb = crop_image_by_bbox(image_list[x], det, img_size, margin)
            cv2.imwrite(os.path.join(cropped_big_face, img_name), cropped)
            x1, y1, x2, y2 = bb
            cv2.rectangle(image_list[x], (x1, y1), (x2, y2), (0, 0, 255), 3)  # draw bounding box only on big face
            cv2.imwrite(dest_fname, image_list[x])
            aligned = cv2.resize(
               cropped, (image_size, image_size), cv2.INTER_LINEAR)
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
            fnames_list.append(img_name)
      else:
         discared_image_cnt += 1
         discard_fname = os.path.join(discarded_folder_path, fname)
         cv2.imwrite(discard_fname, img_temp)
         logger.debug('Total number of Discarded images are:' + str(discared_image_cnt))

   if len(img_list) > 0:
      images = np.stack(img_list)
      logger.debug('Total number of Localized images:' + str(len(images)))  # No. of images been able to be localized
      return images, fnames_list
   else:
      logger.info("No faces recognized, please input images.")
      sys.exit()
      return None
      
"""def get_file(filename):  # pragma: no cover
    try:
        src = os.path.join(root_dir(), filename)
        # Figure out how flask returns static files
        # Tried:
        # - render_template
        # - send_file
        # This should not be so non-obvious
        return open(src).read()
    except IOError as exc:
        return str(exc)
def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))    """    