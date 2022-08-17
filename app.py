from flask import Flask, render_template, request, redirect, jsonify
import os, cv2, numpy, face_recognition, json, io
from PIL import Image
import base64
from dotenv import load_dotenv
import pymongo

app = Flask(__name__, template_folder="templates")


load_dotenv()
print("[ Env ]\t\t Loaded Environment Variables")

myclient = pymongo.MongoClient(os.environ['MONGO_URI'])
print("[ MongoDB ]\t Connected to MongoDB Atlas")

db = myclient['facerec']
students = db['facerec']

scale = 0.3
box_multiplier = 1/scale
classNames = []
path = 'faces'
for img in os.listdir(path):
    classNames.append(os.path.splitext(img)[0])
encodes = open('faces.dat', 'rb')
knownEncodes = numpy.load(encodes)
print('[ Encode ]\t Encodings Loaded Successfully')

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/verify", methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        print("POST")
        # print(request.json)
        file = request.json['image']
        print(file[:30])
        base64_decoded = base64.b64decode(file.split(",")[1])
        image_numpy = numpy.frombuffer(base64_decoded, numpy.uint8)
        print(image_numpy)
        img = cv2.imdecode(image_numpy, cv2.IMREAD_COLOR)
        print(img.shape)
        Current_image = cv2.resize(img,(0,0),None,scale,scale)
        print(Current_image.shape)
        face_locations = face_recognition.face_locations(Current_image,  model='cnn')
        face_encodes = face_recognition.face_encodings(Current_image,face_locations)
        name = "Please align face properly in frame"
        for encodeFace,faceLocation in zip(face_encodes,face_locations):
            matches = face_recognition.compare_faces(knownEncodes,encodeFace, tolerance=0.5)
            # matches = face_recognition.compare_faces(knownEncodes,encodeFace)
            faceDis = face_recognition.face_distance(knownEncodes,encodeFace)
            matchIndex = numpy.argmin(faceDis)


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
        # cv2.imwrite("file.jpg", img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img.astype("uint8"))
        rawBytes = io.BytesIO()
        img.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())
        s = students.find_one({'JNTUH Roll No': name})
        name = str(s) if s else name
        return jsonify({'name': name, 'image': str(img_base64).replace('=', '')[2:-1]})
    else:
        print("GET")
    return "True"

if __name__ == "__main__":
    app.run(debug=True)