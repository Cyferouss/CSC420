import cv2
import numpy as np
def getImageFromVideo(path):
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    while success:
        yield image      
        success, image = vidcap.read()

# gray scale?
def placeImage(coords, scene, img, scale=.5, debug=True):
    # Need to  make sure it doesn't go out of bounds.
    x,y = coords
    img3 = scene.copy()
    img = cv2.resize(img, (int(img.shape[0]*scale), int(img.shape[1]*scale)))
    img3[x:x+img.shape[0], y:y+img.shape[1], :] = img[:,:,:]

    if debug:
        cv2.imshow("", img3)
    return img3

def augmentFlowers(scene, minHeight=.4, numFlowers=4, minDistance=100):
    orb = cv2.ORB_create()
    kp1, desc1 = orb.detectAndCompute(scene, None)

    # Only take keypoints which are above a certain y value
    augmentScene = None
    filteredKp1 = []
    for currPoint in kp1:
        y = currPoint.pt[1]
        if y < scene.shape[0]*minHeight:
            filteredKp1.append(currPoint)
    
    usedKp1 = [filteredKp1[0]]
    for i in range(1, len(filteredKp1)):
        usable = True
        for kp in usedKp1:
            usedKpX = kp.pt[0]
            usedKpY = kp.pt[1]
            
            potentialKpX = filteredKp1[i].pt[0]
            potentialKpY = filteredKp1[i].pt[1]
            dist = ((usedKpX - potentialKpX)**2 + (usedKpY - potentialKpY)**2)**(float(1)/2)
            # closer to 0 means higher up i guess
            if dist < minDistance:
                usable = False
                break
        if usable:
            usedKp1.append(filteredKp1[i])
    augmentedScene = scene.copy()
    for i in range(0, numFlowers):
        x, y = int(usedKp1[i].pt[0]), int(usedKp1[i].pt[1])
        print((x, y))
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