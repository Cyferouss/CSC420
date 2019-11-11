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
# gray scale?
def placeImage(coords, big, nose):
    # Need to  make sure it doesn't go out of bounds.
    x,y = coords
    img3 = big.copy()

    img3[x:x+nose.shape[0], y:y+nose.shape[1], :] = nose[:,:,:]
    cv2.imshow("", img3)

# Test placeImage
img1 = cv2.imread("C:/Users/skunk/Desktop/49897907_271940710169544_307813064789458944_n.png")
img2 = cv2.imread("C:/Users/skunk/Desktop/images.png")

for i in range(0, 500):
    placeImage((i, i), img1, img2)
    cv2.waitKey(delay=20)


# General Flow
'''
out = cv2.VideoWriter(...)
for i in getImageFromVideo("C:/Users/skunk/Downloads/FMSD26.mov"):
    # Find nose
    # model prediction on i
    
    # Process Nose
    
    # Write frame
    # out.write(frame)
'''