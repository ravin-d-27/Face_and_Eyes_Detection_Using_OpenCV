#importing the Open CV module

import cv2

# Loading the cascades

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# defining a function which will do face detection

def detect(gray, frame):
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # image, scale factor, neighbour zone
    # we will get good result with these numbers which are 1.3, 5
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_colour = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1,3)
        
        for (x2,y2,w2,h2) in eyes:
            cv2.rectangle(roi_colour, (x2,y2), (x2+w2,y2+h2), (0,0,255),2)
            
    return frame

# Applying face recognition using webcam

video_capture = cv2.VideoCapture(0)
while True:
    _, frame =  video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow("Footage",canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
        
    