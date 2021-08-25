# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 00:21:03 2021

@author: Tommy
"""

# import module
import os
import cv2
from pdf2image import convert_from_path
import glob
import numpy as np
import matplotlib.pyplot as plt

#%% Define the paths with the data source
# Store Pdf with convert_from_path function
pdf_path = os.path.join("data","sample2.pdf")
 


#%% Functions
def convert_pdf_to_images(pdf_path):
    head, tail = os.path.split(pdf_path)
    doc_name = tail.split(".")[0]
    destination_folder = os.path.join("output",head)
    images = convert_from_path(pdf_path,
                               poppler_path=r"C:\Program Files\poppler-0.68.0\bin")
    for index, image in enumerate(images):
        output_file_name = f'{doc_name}_page{index}.png'
        dst_path = os.path.join(destination_folder, output_file_name)
        try:
            image.save(dst_path)
        except:
            os.mkdir(destination_folder)
            image.save(dst_path)
            



# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# List all files in folder
head, tail = os.path.split(pdf_path)
doc_img_list = glob.glob(os.path.join("output",head,"*.png"))
print(glob.glob(os.path.join("output",head,"*.png")))

view_frames_with_face = True
list_of_doc_pages_with_faces = []
for index, img in enumerate(doc_img_list):
    # Read the frame
    image = cv2.imread(img)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors =9)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(rgb_img, (x, y), (x+w, y+h), (255, 0, 0), 5)
    if len(faces) > 0:
        if view_frames_with_face == True:
            plt.imshow(rgb_img)
            plt.show()
        list_of_doc_pages_with_faces.append(index+1)
        
        


import face_detection
print(face_detection.available_detectors)
detector = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
detections = detector.detect(image[:, :, ::-1])

view_frames_with_face = True
list_of_doc_pages_with_faces = []
for index, img in enumerate(doc_img_list):
    # Read the frame
    image = cv2.imread(img)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors =9)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(rgb_img, (x, y), (x+w, y+h), (255, 0, 0), 5)
    if len(faces) > 0:
        if view_frames_with_face == True:
            plt.imshow(rgb_img)
            plt.show()
        list_of_doc_pages_with_faces.append(index+1)
        