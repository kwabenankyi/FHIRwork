# data processing, CSV & image file I/O
import os
import sys
# Suppress output
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')
import re
import pandas as pd
from keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import Deep learning Libraries --> preprocessing, modeling & Evaluation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, Adamax
from keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# Generate data paths with labels
train_data_dir = '/Users/jerrywu/Desktop/FHIR/chest_xray/train'
filepaths = []
labels = []

folds = os.listdir(train_data_dir)
# print(folds)

for fold in folds:
    foldpath = os.path.join(train_data_dir, fold)
    filelist = os.listdir(foldpath)
    for file in filelist:
        fpath = os.path.join(foldpath, file)
        
        filepaths.append(fpath)
        labels.append(fold)

# Concatenate data paths with labels into one dataframe
Fseries = pd.Series(filepaths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')

train_df = pd.concat([Fseries, Lseries], axis= 1)

# crobed image size
batch_size = 16
img_size = (224, 224)

tr_gen = ImageDataGenerator()
ts_gen = ImageDataGenerator()
val_gen= ImageDataGenerator()

train_gen = tr_gen.flow_from_dataframe(train_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= True, batch_size= batch_size)
g_dict = train_gen.class_indices
classes = list(g_dict.keys())

#loading the pre_trained model
loaded_model = tf.keras.models.load_model('/Users/jerrywu/Desktop/FHIR/pneumonia.h5',compile = False)
loaded_model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])



import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
def open_image():
    file_path = filedialog.askopenfilename(title="Open Image File", filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")])
    if file_path:
        p_prediction(file_path)
        #display_image(file_path)

def p_prediction(image_path):
    image = load_img(image_path)

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Make predictions
    predictions = loaded_model.predict(img_array)
    class_labels = classes
    score = tf.nn.softmax(predictions[0])
    result = class_labels[tf.argmax(score)]

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    if result == "PNEUMONIA":
        print("Contact your GP immediately, because you might have symptoms of Pneumonia!")
    elif result == "NORMAL":
        print("You are chest seems normal!")

"""def display_image(file_path):
    image = Image.open(file_path)
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.photo = photo
    status_label.config(text=f"Image loaded: {file_path}")
"""

def main():
    root = tk.Tk()
    root.title("Pneumonia Detection App")
    text_widget = tk.Text(root, wrap=tk.WORD, height=15, width=35)
    open_button = tk.Button(root, text="Open Image", command=open_image)
    open_button.pack(padx=20, pady=10)
    image_label = tk.Label(root)
    image_label.pack(padx=20, pady=20)
    status_label = tk.Label(root, text="", padx=20, pady=10)
    status_label.pack()
    root.mainloop()

main()


