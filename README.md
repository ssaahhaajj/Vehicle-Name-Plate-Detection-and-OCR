# Vehicle-Name-Plate-Detection-and-OCR
###### (Code may have contain bugs)

FOR COMPLETE DETAILS REFER TO THIS LINK:
https://docs.google.com/document/d/1g59nSdaVmuzD7d7yV5HkR6v3xfKdxnw5m-grqW7Lkto/edit?usp=sharing


The Model can be broken down into 2 parts:

## 1. Licence Plate detection and segmentation of characters


The first part of Licence Plate Detection is completed using YOLOv3 with darknet backbone. I used the dataset provided by TCS i.e. 237 images of cars and some bikes at different angles.

For providing the bounding boxes and annotations I have used LabelImg which is also available in github. Steps involved in bulding dataset:


   - first have all the images in a path
   
   - then open Labelmg, draw the bounding box and save the annotation of classes in it. In my case no of classes is 1
   
   - save in the YOLO format
   
   - Run the program LINK (https://gist.github.com/ssaahhaajj/5d9e53a30bec46fdcc76cc604e890bab) in the path of folder having training and testing images


The dataset is available at https://www.floydhub.com/sahajjain/datasets/licenceplate

Model Weights : https://drive.google.com/file/d/1qAof2POtYAInDYI7XyS4O4m5V4ByMq-b/view?usp=sharing


 For the segmentation part I have simply used the OpenCV Library to convert into different forms and my Closed Component Analysis to segment the characters is hard-coded.


## 2. Character Recognition


And at last for the Character prediction, I have used a pre-trained model on the english alphabets in different fonts and the numbers which just predicts the character present on the Licence Plate.


 For the improvement in the accuracy of the model I checked the presence of “IND” substring in the string formed and removed it.

##### **Apache Licence**
