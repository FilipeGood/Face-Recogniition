# link => https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
"""
    This script extracts the 128 d features of the face (face embeddings) and saves them

    python3 extract_embeddings.py --dataset dataset \
	--embeddings output/embeddings.pickle \
	--detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7
"""


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--embeddings", required=True,
	help="path to output serialized db of facial embeddings") # path to the output embeddings file. The script will compute face embeddings which we'll serialize to disk
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model") # Path to the OpenCV deep learning Torch embedding model. This model will allow us to extract a 128-D facial embedding vector.
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())



# load our serialized face detector from disk
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
# Caffe based DL face detector to localize faces in images
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
# This model is Torch-based and is responsible for extracting facial embeddings via deep learning feature extraction
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])


# Get the images
imagePaths = list(paths.list_images(args["dataset"])) # contains path to each images in the dataset

knownEmbeddings = []
knownNames = []
total_n_faces_processed = 0

# This loop will be responsible for extracting embeddings from faces found in each image
for (i, imagePath) in enumerate(imagePaths):
    # imagePath => 'dataset/adrian/00004.jpg' => name = 'andrian
    name = imagePath.split(os.path.sep)[-2] # os.path.sep => os path separator "/" (linux) or "\" (windows)

    image = cv2.imread(imagePath) # read images
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2] # grab the image dimensions

    #Construct a BLOB (Binary Large OBject) from the image
    imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# Apply OpenCV's deep learning-based face detector
    detector.setInput(imageBlob)
    detections = detector.forward()


    if len(detections) > 0:
        print('Detection for image: ', detections)

        # we're making the assumption that each image has only ONE
		# face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:
            # Compute the (x,y) coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI (Region of Interest) and grab the ROI dimensions
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if (fW < 20 or fH < 20):
                continue

            # Construct blob for the face ROI
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			# Pass the blob thourgh our face embedding model to obtain the 128-d quantification of the face
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            print('Feature vector: ', vec)
			# add the name of the person + corresponding face embedding to their respective lists
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total_n_faces_processed += 1


# dump the facial embeddings + names to disk
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()
