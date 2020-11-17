from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle


"""
    This script gets the face embeddings and trains a SVM in order to predict the faces based on the features of the face

    python3 train_model.py --embeddings output/embeddings.pickle \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle
"""

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings") # path to the serialized embeddings
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces") # output model that recognizes faces
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder") # path to the label encoder
args = vars(ap.parse_args())



# load face embeddings
data = pickle.loads(open(args["embeddings"], "rb").read())

# encode the labes
le = LabelEncoder()
labels = le.fit_transform(data['names'])

# train SVM model
recognizer = SVC(C=1.0, kernel='linear', probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()
