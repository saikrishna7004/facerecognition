from time import sleep
import time
import cv2
import numpy as np
import face_recognition
import os
import json

# Define the path for training images for OpenCV face recognition Project


scale = 0.25    
box_multiplier = 1/scale


# Define a videocapture object
cap = cv2.VideoCapture(0)

# Images and names
classNames = []
path = 'faces'

# Function for Find the encoded data of the input image
# Reading the training images and classes and storing into the corresponding lists
for img in os.listdir(path):
    classNames.append(os.path.splitext(img)[0])

# Find encodings of training images

encodes = open('faces.dat', 'rb')
knownEncodes = np.load(encodes)
print('Encodings Loaded Successfully')

while True:
    success, img = cap.read()  # Reading Each frame

   # Resize the frame
    Current_image = cv2.resize(img,(0,0),None,scale,scale)
    Current_image = cv2.cvtColor(Current_image, cv2.COLOR_BGR2RGB)

    # Find the face location and encodings for the current frame
    
    face_locations = face_recognition.face_locations(Current_image,  model='cnn')
    face_encodes = face_recognition.face_encodings(Current_image,face_locations)
    for encodeFace,faceLocation in zip(face_encodes,face_locations):
        # matches = face_recognition.compare_faces(knownEncodes,encodeFace, tolerance=0.5)
        matches = face_recognition.compare_faces(knownEncodes,encodeFace)
        faceDis = face_recognition.face_distance(knownEncodes,encodeFace)
        matchIndex = np.argmin(faceDis)


        # If match found then get the class name for the corresponding match

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

        else:
            name = 'Unknown'
        print(name)
        y1,x2,y2,x1=faceLocation
        y1,x2,y2,x1=int(y1*box_multiplier),int(x2*box_multiplier),int(y2*box_multiplier), int(x1*box_multiplier)

        # Draw rectangle around detected face

        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-20),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),2)
    cv2.imshow("window_name", img)
    # sleep(5000)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#closing all open windows 
cap.release()
cv2.destroyAllWindows()