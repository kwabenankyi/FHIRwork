# data processing, CSV & image file I/O
import os
import sys
# Suppress output
#sys.stdout = open(os.devnull, 'w')
#sys.stderr = open(os.devnull, 'w')
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
from PIL import Image, ImageTk

import tkinter.font as tkFont
warnings.filterwarnings('ignore')

import main as appointment_main


# Generate data paths with labels
this_filepath = os.path.dirname(__file__)
print(this_filepath)
train_data_dir = this_filepath+'\\train'
print(train_data_dir)
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
loaded_model = tf.keras.models.load_model(this_filepath+'\\pneumonia.h5',compile = False)
loaded_model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])






import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk



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


   #sys.stdout = sys.__stdout__
   #sys.stderr = sys.__stderr__
   return result

"""def display_image(file_path):
   image = Image.open(file_path)
   photo = ImageTk.PhotoImage(image)
   image_label.config(image=photo)
   image_label.photo = photo
   status_label.config(text=f"Image loaded: {file_path}")
"""





class App:
    def display_image(self,file_path):
        image = Image.open(file_path)
        image = image.resize((255, 255))
        photo = ImageTk.PhotoImage(image)
        self.GLabel_549.config(image=photo)
        self.GLabel_549.photo = photo
        #status_label.config(text=f"Image loaded: {file_path}")
    
    def open_image(self):
        file_path = filedialog.askopenfilename(title="Open Image File", filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")])
        if file_path:
            self.GLineEdit_681.insert(tk.END,file_path)
            #self.GLineEdit_681["text"] = file_path
            result = p_prediction(file_path)
            self.display_image(file_path)
            if result == "PNEUMONIA":
                self.GLabel_254["text"] = "Based on the result of the test, it appears that the patient has pneumonia.However, there is still 30.7% chance in which the patient was falsely tested for Pneumonia, click on the booking feature offered by FHIR to book the earliest appointment."
                #self.GLabel_254["text"] = "Contact your GP immediately, because you might have symptoms of Pneumonia!"
            elif result == "NORMAL":
                self.GLabel_254["text"] = "Based on the result of the test, it appears that you DON’T have pneumonia. However, if there are any other concerns, please click on the booking feature offered by FHIR to book the earliest appointment with your nearest hospital."
            self.display_appt_button()
    
    def display_appt_button(self):
        GButton_112=tk.Button(self.root)
        GButton_112["bg"] = "#e3eeee"
        ft = tkFont.Font(family='Times',size=18)
        GButton_112["font"] = ft
        GButton_112["fg"] = "#000000"
        GButton_112["justify"] = "center"
        GButton_112["text"] = "Book a Appointment"
        GButton_112["relief"] = "raised"
        GButton_112.place(x=320,y=326,width=237,height=57)
        GButton_112["command"] = self.GButton_112_command
    def display_appt_label(self):
        label=tk.Label(self.root)
        label["bg"] = "#ffffff"
        ft = tkFont.Font(family='Times',size=14)
        label["font"] = ft
        label["fg"] = "#333333"
        label["justify"] = "left"
        label["text"] = "appointment booked successfully!"
        label.place(x=280,y=390,width=320,height=28)

    def __init__(self, root):
        self.root = root
        #setting title
        self.root.title("")
        #setting window size
        width=600
        height=430
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.root.geometry(alignstr)
        self.root.resizable(width=False, height=False)

        GLabel_385=tk.Label(self.root)
        GLabel_385["bg"] = "#ffffff"
        GLabel_385["cursor"] = "mouse"
        ft = tkFont.Font(family='Times',size=10)
        GLabel_385["font"] = ft
        GLabel_385["fg"] = "#333333"
        GLabel_385["justify"] = "center"
        GLabel_385["text"] = ""
        GLabel_385.place(x=0,y=10,width=800,height=800)

        GButton_881=tk.Button(self.root, text="Open Image", command=self.open_image)
        GButton_881["bg"] = "#e3eeee"
        ft = tkFont.Font(family='Times',size=18)
        GButton_881["font"] = ft
        GButton_881["fg"] = "#000000"
        GButton_881["justify"] = "center"
        GButton_881["text"] = "Upload Image"
        GButton_881["relief"] = "raised"
        GButton_881.place(x=30,y=80,width=236,height=56)
        #GButton_881["command"] = self.GButton_881_command

        self.GLabel_254=tk.Label(self.root)
        self.GLabel_254["bg"] = "#ffffff"
        ft = tkFont.Font(family='Times',size=14)
        self.GLabel_254["font"] = ft
        self.GLabel_254["fg"] = "#333333"
        self.GLabel_254["justify"] = "left"
        self.GLabel_254["text"] = ""
        self.GLabel_254["wraplength"] = 320
        self.GLabel_254.place(x=280,y=160,width=320,height=140)


        GLabel_88=tk.Label(self.root)
        GLabel_88["bg"] = "#e3eeee"
        ft = tkFont.Font(family='Times',size=28)
        GLabel_88["font"] = ft
        GLabel_88["fg"] = "#333333"
        GLabel_88["justify"] = "center"
        GLabel_88["text"] = "Pneumonia Detection"
        GLabel_88.place(x=0,y=0,width=599,height=71)

        self.GLabel_549=tk.Label(self.root)
        self.GLabel_549["bg"] = "#9fb7b8"
        ft = tkFont.Font(family='Times',size=10)
        self.GLabel_549["font"] = ft
        self.GLabel_549["fg"] = "#333333"
        self.GLabel_549["justify"] = "center"
        self.GLabel_549["text"] = ""
        self.GLabel_549.pack(side = "bottom", fill = "both", expand = "yes")
        self.GLabel_549.place(x=10,y=160,width=255,height=255)

        self.GLineEdit_681=tk.Entry(root)
        self.GLineEdit_681["borderwidth"] = "1px"
        ft = tkFont.Font(family='Times',size=10)
        self.GLineEdit_681["font"] = ft
        self.GLineEdit_681["fg"] = "#333333"
        self.GLineEdit_681["justify"] = "center"
        self.GLineEdit_681["text"] = ""
        self.GLineEdit_681.place(x=280,y=80,width=319,height=56)


    def GButton_881_command(self):
        print("command")


    def GButton_112_command(self):
        appointment_main.main()
        self.display_appt_label()

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

main()


