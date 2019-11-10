import cv2
import numpy as np
def getImageFromVideo(path):
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    while success:
        yield image      
        success, image = vidcap.read()

'''
for i in getImageFromVideo("C:/Users/skunk/Downloads/FMSD26.mov"):
    cv2.imshow("image", i)
    cv2.waitKey(0)
'''
