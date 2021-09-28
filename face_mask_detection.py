### Face mask detection ###
"""
    Description :
                    Face mask detection implementation using CNN model with keras, where the model trained on collection of images of people wearing mask and other not,
                    so the model can classify every image if the face wearing mask or not,
                    first we use opencv haarcascade classifier to detect face and then run the CNN model to classify if this face has mask or not,
                    and finally draw bounding box around face and output class "With Mask" or "Without Mask"
"""

# Importing modules
import sys
import cv2
import numpy as np
from keras.models import load_model

# Load the model last checkpoint (I choose the latest one and delete other versions) 
model = load_model("./Checkpoints/Model_08")

# Load the haarcascafe classifier
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Output classes labels
labels_dict = {0:'Without Mask', 1:'With Mask'}
color_dict  = {0:(0,0,255), 1:(0,255,0)}

# Detect mask on an image
def detectOnImage(image_path):
    """Function to detect face mask within an image"""
    
    # Load the image
    img = cv2.imread(image_path)

    # Convert image to gray scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces withi the image
    faces = classifier.detectMultiScale(img_gray, 1.3, 5)
    
    # Loop over faces and draw rectangles around each face
    for f in faces:
        
        # Get face dimensions 
        (x, y, w, h) = f
        face_img = img[y:y+h, x:x+w]
        
        # Resize face image
        resized = cv2.resize(face_img, (150,150))
        
        # Normalize face image
        normalized = resized/255.0
        
        # Reshape face image
        reshaped = np.reshape(normalized, (1,150,150,3))
        reshaped = np.vstack([reshaped])
        
        # Get model detection result and its label (1: With Mask, 0: Without Mask)
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        
        # Draw bounding box and output class 
        cv2.rectangle(img, (x,y), (x+w,y+h), color_dict[label], 2)
        cv2.rectangle(img, (x,y-40), (x+w,y), color_dict[label], -1)
        if label == 1:
            cv2.putText(img, labels_dict[label], (x+40, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        else:
            cv2.putText(img, labels_dict[label], (x+13, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
    # Show the image
    cv2.imshow('Face Mask Detection', img)
    cv2.waitKey(0)

# Detect mask on video or cam
def detectOnVid(video_path=False, cam=False):
    """Function to detect face mask within a video or webcam"""
    
    # Load the video or cam
    if cam:
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(video_path)
        
    # Loop over video or cam frames
    while True:

        # Get video frames
        success, frame = video.read()
    
        # If there's no frames stop
        if not success:
            break
        
        # Convert frame to gray scale
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces withi the image
        faces = classifier.detectMultiScale(img_gray, 1.3, 5)
        
        # Loop over faces and draw rectangles around each face
        for f in faces:

            # Get face dimensions 
            (x, y, w, h) = f
            face_img = frame[y:y+h, x:x+w]
            
            # Resize face image
            resized = cv2.resize(face_img, (150,150))
            
            # Normalize face image
            normalized = resized/255.0
            
            # Reshape face image
            reshaped = np.reshape(normalized, (1,150,150,3))
            reshaped = np.vstack([reshaped])
            
            # Get model detection result and its label (1: With Mask, 0: Without Mask)
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            
            # Draw bounding box and output class 
            cv2.rectangle(frame, (x,y), (x+w,y+h), color_dict[label], 2)
            cv2.rectangle(frame, (x,y-40), (x+w,y), color_dict[label], -1)
            cv2.putText(frame, labels_dict[label], (x+50, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
        # Show the image
        cv2.imshow('Face Mask Detection', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    # Stop video or cam
    video.release()
    cv2.destroyAllWindows()


# Get flag from cmd (img, vid or cam)
flag = sys.argv[1]   

# Run the function
# NOTE : sys.argv[2] is the path of image or video passed on cmd
if flag == "img":
    detectOnImage(f"{sys.argv[2]}")
elif flag == "vid":
    detectOnVid(f"{sys.argv[2]}")
else:
    detectOnVid(cam=True)


