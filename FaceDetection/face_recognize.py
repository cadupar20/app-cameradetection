# facerec.py
import cv2, sys, numpy, os

size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

# Part 1: Create fisherRecognizer
print('Recognizing Face Please Be in sufficient Light Conditions...')
# Create a list of images and a list of corresponding names
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1
(width, height) = (1024, 768)

# Create a Numpy array from the two lists above
(images, lables) = [numpy.array(lis) for lis in [images, lables]]


print(dir (cv2.face))

# OpenCV trains a model from the images
# NOTE FOR OpenCV2: remove '.face'
#recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer = cv2.face.EigenFaceRecognizer_create()
recognizer.train(images, lables)

# Part 2: Use fisherRecognizer on camera stream
face_cascade = cv2.CascadeClassifier(haar_file)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    (_, im) = cap.read()
    #Convert the frame to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))#, flags=cv2.CASCADE_SCALE_IMAGE)

    #Recognize and label the faces
    for (x, y, w, h) in detected_faces:
        #cv2.rectangle(im, (x, y),(x + w,y + h),(255, 0, 0), 2)
        
        #Convert the frame to grayscale
        face = gray[y:y + h, x:x + w]
        #Resize the face so that it can be used in the model
        face_resize = cv2.resize(face, (width, height))
        
        # Try to recognize the face
        #prediction = recognizer.predict(face_resize)
        prediction = recognizer.predict(face_resize)
        print('Predition: ', prediction)

        if int(prediction[1]) > 500:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(im,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,0.9,(0, 255, 0),2)
        else:
            cv2.putText(im,'Desconocido - %.0f' % (prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0),2)
        
        #Display the frame with face recognition
        cv2.imshow('Recognize Face', im)
        
        key = cv2.waitKey(1)
        if key == 27:
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#Release the camera and destroy all the windows
cap.release()
cv2.destroyAllWindows()