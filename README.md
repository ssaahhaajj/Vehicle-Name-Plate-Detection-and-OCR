# Vehicle-Name-Plate-Detection-and-OCR

The Model can be broken down into 2 parts:
1. Licence Plate detection and segmentation of characters
2. Character Recognition

The first part of Licence Plate Detection is completed using YOLOv3 with darknet backbone. I used the dataset provided by TCS i.e. 237 images of cars and some bikes at different angle

For providing the bounding boxes and annotations I have used LabelImg which is also available in github. Steps involved in bulding dataset:
   -> first have all the images in a path
   -> then open Labelmg, draw the bounding box and save the annotation of classes in it. In my case no of classes is 1 i.e. numplate
   -> save in the YOLO format
   -> Run the program LINK (https://gist.github.com/ssaahhaajj/5d9e53a30bec46fdcc76cc604e890bab) in the path of folder having training and testing images

The dataset is available at https://www.floydhub.com/sahajjain/datasets/licenceplate



