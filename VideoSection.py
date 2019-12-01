import cv2
import numpy as np
import random

keyPoints = []

def getImageFromVideo(path):
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    while success:
        yield image      
        success, image = vidcap.read()

# Gray scale?
# FaceCoords is format ((x1,y1), (x2,y2))
def placeImage(coords, scene, img, scale=.5, debug=True, FaceCoords=None):
    # Need to  make sure it doesn't go out of bounds.
    x1, y1 = coords
    img3 = scene.copy()

    img = cv2.resize(img, (int(img.shape[0]*scale), int(img.shape[1]*scale)))
    xCenter = int(img.shape[1]/2)
    yCenter = int(img.shape[0]/2)
    for i in range(-xCenter, xCenter):
        for j in range(-yCenter, yCenter):
            # Too lazy to write bound check for potential index out of bounds so i'll use this hack.
            try:
                r = img[j + yCenter, i + xCenter][0]
                g = img[j + yCenter, i + xCenter][1]
                b = img[j + yCenter, i + xCenter][2]
                inABox = False
                if r != 0 or g != 0 or b != 0:
                    for f in FaceCoords:
                        if (not f or \
                        ((f[0][0] <= i + x1 and f[1][0] >= i + x1) and\
                        (f[0][1] <= j + y1 and f[1][1] >= j + y1))):
                            inABox = True
                    
                    if not inABox:
                        img3[j + y1, i + x1] = img[j + yCenter, i + xCenter]
            except:
                pass
    return img3

def augmentFlowers(scene, numFlowers=5, minDistance=100, debug=False, FaceCoords=None, prevFrame=None):
    orb = cv2.ORB_create()
    kp1, desc1 = orb.detectAndCompute(scene, None)
    if (len(kp1) == 0):
        return scene

    if debug:
        img2 = cv2.drawKeypoints(scene, kp1, None, color=(255,0,0))
        cv2.imshow("", img2)
        cv2.waitKey(delay=20000)
    global keyPoints
    usedKp1 = keyPoints
    if len(usedKp1) == 0:
        usedKp1 = [[kp1[0], random.uniform(.1,.2)]]
        for i in range(1, len(kp1)):
            usable = True
            for kp, _ in usedKp1:
                usedKpX = kp.pt[0]
                usedKpY = kp.pt[1]
                
                potentialKpX = kp1[i].pt[0]
                potentialKpY = kp1[i].pt[1]
                dist = ((usedKpX - potentialKpX)**2 + (usedKpY - potentialKpY)**2)**(float(1)/2)
                # closer to 0 means higher up i guess
                if dist < minDistance:
                    usable = False
                    break

            if usable:
                usedKp1.append([kp1[i], random.uniform(0.1,.2)])
        keyPoints = usedKp1
    

    augmentedScene = scene.copy()
    
    delList = []
    for i in range(0, min(numFlowers, len(usedKp1))):
        x, y = int(usedKp1[i][0].pt[0]), int(usedKp1[i][0].pt[1])
        scale = usedKp1[i][1]
        keyPoints[i][1] *= 1.05
        if keyPoints[i][1] > .4:
            delList.append(i)
        # TODO: Not sure if it's suppose to be (y, x) or (x, y)
        augmentedScene = placeImage((y, x), augmentedScene, flower, FaceCoords=FaceCoords, scale=scale)
    tmp = []
    for i in range(0, len(keyPoints)):
        if i not in delList:
            tmp.append(keyPoints[i])
    keyPoints = tmp
    return augmentedScene

# Test placeImage
img1 = cv2.imread("C:/Users/skunk/Desktop/49897907_271940710169544_307813064789458944_n.png")
img2 = cv2.imread("C:/Users/skunk/Desktop/images.png")
flower = cv2.imread("C:/Users/skunk/Desktop/csc420proj/CSC420/flower.png")

#augmentedScene = augmentFlowers(img1)
#cv2.imshow("", augmentedScene)
# General Flow
testing = False
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter("C:/Users/skunk/Desktop/csc420proj/CSC420/output.avi", fourcc, 10, (640, 480), True)

if testing:
    for frame in getImageFromVideo("C:/Users/skunk/Desktop/csc420proj/CSC420/TestVideo.mp4"):
        augmentedScene = augmentFlowers(frame)
        out.write(augmentedScene)
    out.release()

face_cascade = cv2.CascadeClassifier("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml")
count = 0

cap = cv2.VideoCapture(0)

#for frame in getImageFromVideo("C:/Users/skunk/Desktop/csc420proj/CSC420/TestVideo.mp4"):
while(True):
    ret, frame = cap.read()
    print(frame.shape)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    remapped = []
    for (x,y,w,h) in faces:
        remapped.append(((x, y), (x + w, y+ h)))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 3)
    frame = augmentFlowers(frame, FaceCoords=remapped)
    cv2.imshow("",frame)
    cv2.waitKey(delay=1)
    out.write(frame)
    if cv2.waitKey(33) == ord('a'):
        break


out.release()
