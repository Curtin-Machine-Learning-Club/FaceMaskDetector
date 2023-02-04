#!/usr/bin/env pipenv-shebang

##
# Author : Milan Marocchi
# Purpose : To run a mask detector through video
# NOTE : Inspiration take from https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/
##
print("[INFO] Loading libraries")
import sys
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from threading import Thread
from threading import Lock
import tensorflow as tf
import numpy as np
import argparse
import imutils
import time
import cv2
import os

"""
Runs the default callback
"""
def defaultCallBack():
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        return True

    return False

"""
Draws and displays the frame
"""
def drawAndDisplay(frame, locs, preds):

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    return frame

class VideoController():
    
    def __init__(self):
        print("[INFO] Setting up camera")
        self.done = False
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 640)
        self.cam.set(4, 480)
        self.cameraThread = None
        self.frame = None
    
    def start(self):
        self.cameraThread = Thread(target=self.runCamera)
        self.cameraThread.start()
        return self

    def runCamera(self):
        while not self.done:
            _, frame_ = self.cam.read()
            self.frame = frame_

    def stop(self):
        self.done = True
        self.cameraThread.join() 

    def getCamera(self):
        return self.cam

    def getFrame(self):
        frame = self.frame

        return frame
    
    def release(self):
        self.cam.release()
    

class MaskDetector():

    def __init__(self, args, videoController):
        self.done = False
        self.args = args
        self.videoController = videoController
        self.maskNet = None
        self.output = None
        self.faceNet = None
        self.detectThread = None

    def initialise(self):
        print("[INFO] loading face detector model...")
        prototxtPath = os.path.sep.join([self.args["face"], "deploy.prototxt"])
        weightsPath = os.path.sep.join([self.args["face"],
                                        "res10_300x300_ssd_iter_140000.caffemodel"])
        self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        # load the face mask detector model from disk
        print("[INFO] loading face mask detector model...")

        self.maskNet = self.maskNet = load_model(self.args["model"])

        return self
        

    def start(self):

        self.detectThread = Thread(target=self.runDetector)
        self.detectThread.start()
        return self

    def stop(self):
        
        self.done = True
        self.detectThread.join()

    """
    Detects the face and then predicts

    param frame : The image to use
    returns : The location of the faces and associated predictions
    """
    def detectAndPredictMask(self, frame):
        
        (height, width) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300),
                                      (104.0, 177.0, 123.0))

        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()

        faces = []
        locs = []
        preds = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i , 2]

            if confidence > self.args["confidence"]:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX , startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(width - 1, endX), min(height - 1, endY))

                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

            if len(faces) > 0:
                faces = np.array(faces, dtype="float32")
                preds = self.maskNet.predict(faces, batch_size=32)

            return (locs, preds)

    """
    Runs a detector iteration
    """
    def _runDetectorIteration(self):

        frame = self.videoController.getFrame() 
        locs, preds = self.detectAndPredictMask(frame)
        self.output = [locs, preds]

    """
    Runs the detector
    """
    def runDetector(self):

        while not self.done:
            self._runDetectorIteration()

    """
    Returns the mask net predictions
    
    param faces : The faces to use for prediction
    returns : The prediction on wearing mask or not wearing mask
    """
    def _getMaskNetPredictions(self, faces):
        input_details = self.maskNet.get_input_details()
        output_details = self.maskNet.get_output_details()

        # Test model on random input data.
        input_shape = input_details[0]['shape']
        input_data = np.array(faces, dtype=np.float32)
        self.maskNet.set_tensor(input_details[0]['index'], input_data)

        self.maskNet.invoke()

        preds = self.maskNet.get_tensor(output_details[0]['index'])

        return preds
    
    """
    Returns the output of prediction (preds, locs) where preds are the
    predictions and locs are the locations of the face.
    """
    def getOutput(self):
        return self.output
    

"""
Main method for running the mask detecting video functionality
"""
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str,
        default="face_detector",
        help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
        default="mask_detector.model",
        help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    videoController = VideoController().start()
    time.sleep(10)
    maskDetector = MaskDetector(args, videoController).initialise().start()
        
    done = False
    while not done:
        output = maskDetector.getOutput()
        if not (output == None):
            frames = drawAndDisplay(videoController.getFrame(), output[0], output[1])
            cv2.imshow('DETECT', frames)
        if defaultCallBack():
            done = True
        if done:
            videoController.stop()
            maskDetector.stop()
            exit()
    
    cv2.destroyAllWindows
    videoController.release()

if __name__ == "__main__":
    main()
