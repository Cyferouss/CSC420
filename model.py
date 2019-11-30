import cv2
import VideoSection

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

out = cv2.VideoWriter("C:/Users/skunk/Desktop/csc420proj/CSC420/output.avi", fourcc, 60, (400, 300), True)
for frame in VideoSection.getImageFromVideo("C:/Users/skunk/Desktop/csc420proj/CSC420/TestVideo.mp4"):
    cv2.imshow('img', frame)
    cv2.waitKey(delay="20")

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frameGray, 1.1, 4)

    for (x,y,w,h) in faces:
        cv2.rectange(frame, (x,y), (x+w,y+h),(255,0,0), 3)
    cv2.imshow('img', frame)
    cv2.waitKey(delay="100000")
    out.write(frame)
out.release()