import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

"""
    Takes an images as input and tries to recognize the faces in the image
    First it uses the pre-trained Caffe DL model to detect the ROI of the faces
    Then it uses the pre-trained torch DL model to extract the features - face embeddings
    Finaly, it uses the SVM model to predict the names in the image

    python3 recognize_image.py --detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7 \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle \

"""



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")  # we will use this model to detect where in the images the face ROIs are
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

"""
    Load stuff
"""

# load our serialized face detector from disk
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath) # pre-trained Caffe DL model to detect where in the image the faces are

# load our serialized face embedding model from disk
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"]) # pre-trained torch DL model to calculate our 128-d face embeddings

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read()) # SVM classifier
le = pickle.loads(open(args["le"], "rb").read())


# load the image, resize it to have a width of 600 pixels (while
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

# construct a blob from the image
imageBlob = cv2.dnn.blobFromImage(
	cv2.resize(image, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)


# apply OpenCV's deep learning-based face detector to localize the faces
detector.setInput(imageBlob)
detections = detector.forward()



# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the prediction
    confidence = detections[0, 0, i, 2]

    # filter out weak detections
    if confidence > args["confidence"]:
        # compute the (x, y)-coordinates of the bounding box for the face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # extract the face ROI
        face = image[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]
        # ensure the face width and height are sufficiently large
        if fW < 20 or fH < 20:
        	continue

        # construct a blob for the face ROI
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
        	(0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vec = embedder.forward() # get the 128d face embeddings
        # perform classification to recognize the face
        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        # draw the bounding box of the face along with the associated
        # probability
        text = "{}: {:.2f}%".format(name, proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
        	(0, 0, 255), 2)
        cv2.putText(image, text, (startX, y),
        	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

cv2.imshow("Image", image) # show the ouput image
cv2.waitKey(0)