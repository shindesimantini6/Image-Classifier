import cv2
from keras.applications.mobilenet import MobileNet, decode_predictions, preprocess_input
import numpy as np
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
   x = np.array(image)
   X = np.array([x]) 

   # apply pre-processing
   image_preprocess = preprocess_input(image)
   model = load_model(model_path)
   pred = model.predict(image_preprocess)
   stats = dict(zip(classes, pred[0]))
   stats.get('naz_glasses_train')
   prediction = max(stats, key=stats.get)
   return prediction


