#pip install -q -U "tensorflow-gpu==2.0.0b1"
#pip install -q -U tensorflow_hub


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import logging
import argparse
import sys
import json
from PIL import Image

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser ()
parser.add_argument ('--image_dir', default='./test_images/hard-leaved_pocket_orchid.jpg', help = 'Path to image.', type = str)
parser.add_argument('--model', help='Trained Model.', type=str)
parser.add_argument ('--top_k', default = 5, help = 'Top K most likely classes.', type = int)
parser.add_argument ('--classes' , default = 'label_map.json', help = 'Mapping of categories to real names.', type = str)
commands = parser.parse_args()
image_path = commands.image_dir
model_path = commands.model
top_k = commands.top_k
classes = commands.classes

image_size = 224



# Create the process_image function
with open(classes, 'r') as f:
    class_names = json.load(f)

#Load Model
model = tf.keras.models.load_model(
  model_path, 
  custom_objects={'KerasLayer': hub.KerasLayer})

def process_image(img):
    image = np.squeeze(img)
    image = tf.image.resize(image, (image_size, image_size))/255.0
    return image


#Prediction

def predict(image_path, model, top_k = 5):
    im = Image.open(image_path)
    image = np.asarray(im)
    image = process_image(image)
    
    prediction = model.predict(np.expand_dims(image, 0))

    dataframe = pd.DataFrame(prediction[0]).reset_index().rename(columns = {'index': 'class_code', 
                                                            0: 'prob'})\
                        .sort_values(by='prob', ascending = False).head(top_k).reset_index(drop=True)
    dataframe.loc[:,'class_code'] = dataframe.class_code + 1
    dataframe.loc[:,'class_name'] = np.nan

    for value in dataframe['class_code'].values:
        class_name = class_names[str(value)]
        dataframe.loc[:,'class_name'] = np.where(dataframe.class_code==value, class_name, dataframe.class_name)
    
    class_list = []
    prob_list = []
    
    for class_name, prob in dataframe[['class_name','prob']].values:
        class_list.append(class_name)
        prob_list.append(prob)
           
   
    print("Best Classes: {}".format(class_list))
    print("Probs: {}".format(prob_list))
    return dataframe, image



if __name__ == "__main__":
    predict(image_path,model,top_k)
