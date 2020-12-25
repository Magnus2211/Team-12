import numpy as np
import argparse
import cv2
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

def yolo(image1):
    config_file="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    frozen_model="frozen_inference_graph.pb"
    model=cv2.dnn_DetectionModel(frozen_model,config_file)
    Labels=[]
    file_name="labels.txt"
    with open(file_name,"rt") as fpt:
        Labels= fpt.read().rstrip("\n").split("\n") #δημιουργείται η λίστα Labels με τα ονόματα των αντικειμένων που θα αναγνωρίζονται

    model.setInputSize(320,320)
    model.setInputScale(1.0/127.5)
    model.setInputMean((127.5,127.5,127.5))
    model.setInputSwapRB(True)
    img=cv2.imread(image1) #"διβάζει" την εικόνα
    ClassIndex,confidence,bbox=model.detect(img,confThreshold=0.5)
    for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
        cv2.rectangle(img,boxes,color=(0,255,0),thickness=2)
        cv2.putText(img,Labels[ClassInd-1].upper(),(boxes[0]+10,boxes[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tkimage = ImageTk.PhotoImage(image)
    result.config(image=tkimage)
    result.image = tkimage


def detect_objects(image1):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default=image1,
        help="path to input image")
    ap.add_argument("-p", "--prototxt", default="MobileNetSSD_deploy.prototxt.txt",
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", default="MobileNetSSD_deploy.caffemodel",
        help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load our serialized model from disk
    print("[INFO] loading model...")
    model = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    # (note: normalization is done via the authors of the MobileNet SSD
    # implementation)
    image = cv2.imread(args["image"])
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    model.setInput(blob)
    detections = model.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # show the output image
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    tkimage = ImageTk.PhotoImage(image)
    result.config(image=tkimage)
    result.image = tkimage

def search_image():                                                              
    image1 = filedialog.askopenfilename()
    if image1:
        if r.get()==1:
            yolo(image1)
        else:detect_objects(image1)

root = tk.Tk()                                                                   
root.geometry('1200x900-100-100')
root.resizable(True, True)
root.title('YOLO')
w = tk.Label(root, text = "IMAGE-DETECTION-YOLO", font = "Arial 36", bg ='lightgray', width = 900)
w.pack()
r=IntVar()
r.set("1")
Radiobutton(root,text="yolo",variable=r,value=1).pack()
Radiobutton(root,text="allo",variable=r,value=2).pack()
button = tk.Button(root, text = "CHOOSE", font = "Arial 36", command = search_image)
button.pack()
# label to show the result
result = tk.Label(root)
result.pack()
root.mainloop()
