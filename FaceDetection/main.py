import cv2 # pip install opencv-python, matplotlib, ultralytics
import os
import numpy as np

# Face detector
# Este XML debe ser descargado de https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to capture images and store in dataset folder
def capture_images():
    # Create a directory to store the captured images
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    # Open the camera
    cap = cv2.VideoCapture(0)

    # Set the image counter
    count = 0

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the faces and save the images
        for (x, y, w, h) in faces: # (x, y, width, height)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save Images the captured face images in the dataset folder as .JPG
            cv2.imwrite(f'dataset/user{count}.jpg', gray[y:y + h, x:x + w])

            count += 1

        # Display the frame with face detection
        cv2.imshow('Capture Faces', frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Break the loop after capturing a certain number of images
        if count >= 100:
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()


# Function to train the face recognition model
def train_model():
    # Create an empty list to store the face samples and their corresponding labels
    faces = []
    labels = []

    # Load the images from the dataset folder
    for file_name in os.listdir('dataset'):
        if file_name.endswith('.jpg'):
            # Extract the label from the file name (assuming file name format: "userX.jpg")
            label = int(file_name.split('.')[0][4:])
            print("Label:", label)

            # Read the image
            image = cv2.imread(os.path.join('dataset', file_name))
            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #In Color: cv2.COLOR_BGR2RGB

            # Detect Objects in the image in the grayscale image
            # face es una variable que guarda información de las coordenadas de la cara detectada en sus 4 extremos
            # x, y, weidth, height
            face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            #Print face values detected
            print("Face values:", face )

            # Append the face sample and label to the lists
            # Fix added on 08-10-2024 for Ariel Dupar
            for (x, y, w, h) in face:
                faces.append(gray[y:y+h, x:x+w])
                labels.append(label)

    # Train the face recognition model using the faces and labels
    recognizer.train(faces, np.array(labels))


# Function to recognize faces
def recognize_faces():
    # Open the camera
    cap = cv2.VideoCapture(0) #For use mp4 replace for -> cap = cv2.VideoCapture('filename.mp4')

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect Objects in the image in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Recognize and label the faces
        for (x, y, w, h) in faces:
            # Recognize the face using the trained model
            label, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Print the recognized face label in the console
            if label == 0:
                print("Unknown Face")
            else:
                print("Recognized Face:", label)

            # Only display text on the recognized face if the confidence level is higher than 50
            if label > 50:
                # Display the recognized label and confidence level
                cv2.putText(frame, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame with face recognition
        cv2.imshow('Recognize Faces', frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()


# Main function
def main():
    # Capture face images and store in dataset folder
    capture_images()

    # Train the face recognition model
    train_model()

    # Recognize faces using the trained model
    recognize_faces()


if __name__ == '__main__':
    main()
