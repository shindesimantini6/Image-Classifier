import logging
import os
from datetime import datetime
import cv2
from keras.applications.mobilenet import MobileNet, decode_predictions, preprocess_input
import numpy as np

def write_image(out, frame):
    """
    writes frame from the webcam as png file to disk. datetime is used as filename.
    """
    if not os.path.exists(out):
        os.makedirs(out)
    now = datetime.now() 
    dt_string = now.strftime("%H-%M-%S-%f") 
    filename = f'{out}/{dt_string}.png'
    logging.info(f'write image {filename}')
    cv2.imwrite(filename, frame)


def key_action():
    # https://www.ascii-code.com/
    k = cv2.waitKey(1)
    if k == 113: # q button
        return 'q'
    if k == 32: # space bar
        return 'space'
    if k == 112: # p key
        return 'p'
    return None


def init_cam(width, height):
    """
    setups and creates a connection to the webcam
    """

    logging.info('start web cam')
    cap = cv2.VideoCapture(1)

    # Check success
    if not cap.isOpened():
        raise ConnectionError("Could not open video device")
    
    # Set properties. Each returns === True on success (i.e. correct resolution)
    assert cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    assert cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def add_text(text, frame):
    # Put some rectangular box on the image
    # cv2.putText()
    return NotImplementedError


from tensorflow.keras.models import load_model

def predict_frame(image, model_path):
   classes = ['him_teacup',
 'empty_train_ss',
 'Bottle_gk',
 'bodycream_train_ss',
 'headphone_rbk',
 'plant_rbk',
 'empty_crista_train',
 'empty_gk',
 'helge_empty',
 'crista_bottle_train',
 'bottle_rbk',
 'chris_empty',
 'bottle_mw_train',
 'moritz_bottle',
 'mar_yoda_train',
 'biketool_mm',
 'bottle_train_ss',
 'chris_bottle_train',
 'helge_scarf_train',
 'naz_bottle_train',
 'fan_rbk',
 'apple_crista_train',
 'helge_bottle_train',
 'lighter_mm',
 'naz_glasses_train',
 'helge_mouse_train']
   
   # reverse color channels
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

   # reshape image to (1, 224, 224, 3)
   image = np.expand_dims(image,axis=0) 

   # apply pre-processing
   image_preprocess = preprocess_input(image)
   model = load_model(model_path)
   pred = model.predict(image_preprocess)
   stats = dict(zip(classes, pred[0]))
   prediction = max(stats, key=stats.get)
   print(prediction)
   number = stats.get(prediction)
   print(number)
   return prediction, number


