import cv2
import numpy as np

def getImageFromVideo(path):
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    while success:
        yield image      
        success, image = vidcap.read()

# Gray scale?
def placeImage(coords, scene, img, scale=.5, debug=True):
    # Need to  make sure it doesn't go out of bounds.
    x1, y1 = coords
    img3 = scene.copy()
    xCenter = int(img.shape[1]/2)
    yCenter = int(img.shape[0]/2)
    for i in range(-xCenter, xCenter):
        for j in range(-yCenter, yCenter):
            # Too lazy to write bound check for potential index out of bounds so i'll use this hack.
            try:
                x = img[j + yCenter, i + xCenter][0]
                y = img[j + yCenter, i + xCenter][1]
                z = img[j + yCenter, i + xCenter][2]
                if x != 0 or y != 0 or z != 0:
                    img3[j + y1, i + x1] = img[j + yCenter, i + xCenter]
            except:
                pass
    return img3

def augmentFlowers(scene, numFlowers=20, minDistance=300, debug=False):
    orb = cv2.ORB_create()
    kp1, desc1 = orb.detectAndCompute(scene, None)

    if debug:
        img2 = cv2.drawKeypoints(scene, kp1, None, color=(255,0,0))
        cv2.imshow("", img2)
        cv2.waitKey(delay=20000)

    usedKp1 = [kp1[0]]
    for i in range(1, len(kp1)):
        usable = True
        for kp in usedKp1:
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
            usedKp1.append(kp1[i])

    augmentedScene = scene.copy()
    for i in range(0, min(numFlowers, len(usedKp1))):
        x, y = int(usedKp1[i].pt[0]), int(usedKp1[i].pt[1])

        # TODO: Not sure if it's suppose to be (y, x) or (x, y)
        augmentedScene = placeImage((y, x), augmentedScene, flower)
    return augmentedScene

# Test placeImage
img1 = cv2.imread("C:/Users/skunk/Desktop/49897907_271940710169544_307813064789458944_n.png")
img2 = cv2.imread("C:/Users/skunk/Desktop/images.png")
flower = cv2.imread("C:/Users/skunk/Desktop/csc420proj/CSC420/flower.png")

# General Flow
augmentedScene = augmentFlowers(img1)
cv2.imshow("", augmentedScene)
cv2.waitKey(delay=20000)