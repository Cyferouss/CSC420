import cv2
import numpy as np
import random
import os
import sys
import SquidVision

# Particle System Global var.
keyPoints = []

def getImageFromVideo(path):
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    while success:
        yield image      
        success, image = vidcap.read()

# FaceCoords is format ((x1,y1), (x2,y2))
def placeImage(coords, scene, img, scale=.5, debug=True, FaceCoords=None):
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
                    # Flowers should not be drawn on faces.
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
    # If all flowers have been removed from particle system make new ones.
    if len(usedKp1) == 0:
        usedKp1 = [[kp1[0], random.uniform(.1,.2), random.randint(0,2)]]
        for i in range(1, len(kp1)):
            usable = True
            for kp, _, _ in usedKp1:
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
                usedKp1.append([kp1[i], random.uniform(0.1,.2), random.randint(0,2)])
        keyPoints = usedKp1

    augmentedScene = scene.copy()
    delList = []
    # Place flowers
    for i in range(0, min(numFlowers, len(usedKp1))):
        x, y = int(usedKp1[i][0].pt[0]), int(usedKp1[i][0].pt[1])
        scale = usedKp1[i][1]
        keyPoints[i][1] *= 1.05
        if keyPoints[i][1] > .4:
            delList.append(i)        
        augmentedScene = placeImage((y, x), augmentedScene, usedFlower[keyPoints[i][2]], FaceCoords=FaceCoords, scale=scale)
    tmp = []
    # Remove large flowers.
    for i in range(0, len(keyPoints)):
        if i not in delList:
            tmp.append(keyPoints[i])
    keyPoints = tmp
    return augmentedScene


###### LOAD RESOURCES ######
### IMAGE LOCATIONS ###
flower = cv2.imread(os.path.join(os.getcwd(), "CSC420/flower.png"))
flowerGreen = cv2.imread(os.path.join(os.getcwd(), "CSC420/flowerGreen.png"))
flowerRed = cv2.imread(os.path.join(os.getcwd(), "CSC420/flowerRed.png"))

if flower is None or flowerGreen is None or flowerRed is None:
    print("Failed to load a flower resource!!")

### GLOBAL VARIABLES ###
usedFlower = [flower, flowerGreen, flowerRed]
INPUT_VIDEO = ""
OUTPUT_LOCATION = ""
webCamMode = False
face_cascade = cv2.CascadeClassifier("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml")
###### END LOAD RESOURCES ######

### DRIVER CODE ###

if len(sys.argv) > 1:
    INPUT_VIDEO = sys.argv[1]
    OUTPUT_LOCATION = sys.argv[2]
    webCamMode = "true" == sys.argv[3].lower()

if not webCamMode:
    tmp = next(getImageFromVideo(INPUT_VIDEO))
    width = tmp.shape[1]
    height = tmp.shape[0]

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(OUTPUT_LOCATION, fourcc, 30, (width, height), True)

    for frame in getImageFromVideo(INPUT_VIDEO):
        # Tint blue it's underwater
        frame[:,:,0] = 180

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
        out.write(frame)
    out.release()
else:
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()
        if frame is None:
            print("WEBCAM RETURNED NONE CHECK WEBCAM")
        # Tint blue
        frame[:,:,0] = 120

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)
        #-- Detect faces
        faces = face_cascade.detectMultiScale(frame_gray)
        
        #Load CNN and Weights
        SquidNet = SquidVision.sourced_cnn()
        SquidNet.load(os.path.join(os.getcwd(), "CSC420/squidnet_weights.h5"))
        
        remapped = []
        for (x,y,w,h) in faces:
            remapped.append(((x, y), (x + w, y+ h)))

        for (x,y,w,h) in faces:
            #cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 3)

            # THIS PORTION IS THE FACE!
            #cv2.imshow("", frame[y:y+h, x:x+w, :])
            
            #Segmented Slice
            segmented_face = frame[y:y+h, x:x+w, :]
            #Preprocess Slice for CNN
            processed_segment = SquidVision.process_frame(segmented_face)
            #Predict coordinates from CNN
            predicted_points = SquidNet.predict(processed_segment)
            
            #Scale Datapoints to original size
            scaled_data_points = SquidVision.scale_coordinates(predicted_points, (96,96), (np.size(segmented_face, 0), np.size(segmented_face, 1)))
            
            #Instantiate nose path
            nose_path = os.path.join(os.getcwd(), "CSC420/SquidNose.png")
            
            #Augment Face
            augmented_face = SquidVision.draw_nose(segmented_face, scaled_data_points, rgb=True)

            #Replace Frame
            if np.shape(frame[y:y+h, x:x+w, :]) == np.shape(augmented_face):
                frame[y:y+h, x:x+w, :] = augmented_face
            else:
                print("I can't believe you done this.")
            
            # ADD NOSE
            cv2.waitKey(delay=1)

        #frame = augmentFlowers(frame, FaceCoords=remapped)
        #cv2.imshow("",frame)
        #cv2.waitKey(delay=1)
        if cv2.waitKey(1) == ord('a'):
            break

