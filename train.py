import face_recognition, os, cv2
import numpy as np
from tqdm import tqdm

images = []
classNames = []
path = 'faces'

# Function for Find the encoded data of the input image
# Reading the training images and classes and storing into the corresponding lists
for img in os.listdir(path):
    image = cv2.imread(f'{path}/{img}')
    images.append(image)
    classNames.append(os.path.splitext(img)[0])

print(classNames)
def encodeImages(images):
    encodeList = []
    for i in tqdm(range(len(images)), desc="Encoding ", ascii=False, ncols=75, colour='green', unit=' Files'):
        img = images[i]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        # print(encode)
        encodeList.append(encode)
    faceData = open('faces.dat', 'wb')
    np.save(faceData, encodeList)
    faceData = np.load('faces.dat')
    # print(faceData)
    return encodeList

encodeImages(images)
print("Encoding Completed")