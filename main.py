##@package image_selector
#Documentation for selecting images using focus measure operators and machine learning. 
#
#More details.

import numpy as np
import cv2
import os

import pickle

DB_PATH = "dataset/"

## Documentation for a class.
#
#  More details.
class Img_class:

	## The constructor.
	def __init__(self, img, pred):
		self.img = img
		self.pred = pred

	## Documentation for a method result.
	#  @param self The object pointer.
	def result(self):
		print(self.pred)

## ModifiedLaplacian is the function that quantifies the quality of images focus.
#  This measure is based on an alternative definition of the Laplacian one, and it can be used as a blur measure operator included on derivative-based operators.
#  @param img The binary gray scale image
def ModifiedLaplacian(img):

	k = np.array([[-1], [2], [-1]])
	k_t = k.conj().T

	Lx = cv2.filter2D(img, cv2.CV_64F, k, borderType=cv2.BORDER_REPLICATE)
	Lx = cv2.filter2D(img, cv2.CV_64F, k_t, borderType=cv2.BORDER_REPLICATE)

	q = np.abs(Lx) + np.abs(Lx)

	q[np.isnan(q)] = np.min(q)

	return q
## Extract standard statistics from maps.
#
#  Standard statistics of the maps were computed to be used as features in supervised learning algorithms.
def extract_features(q):

	stddev = np.std(q)
	variance = np.var(q)
	weight = np.average(q)

	features = np.array([stddev, variance, weight])
	## @var features 
	#  Array with 3 different features

	return features

## Main function
#
#  More details
def main():

	classifier_f = open("model/MLAP_KNN_model.pickle", "rb")
	model = pickle.load(classifier_f)
	classifier_f.close()


	list_dir = sorted(os.listdir(DB_PATH))

	for img_name in list_dir:



		img = cv2.imread(DB_PATH + img_name)

		im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		im_norm = cv2.normalize(im_gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

		q = ModifiedLaplacian(im_norm)

		features = extract_features(q)

		prediction = model.predict([features])
		print("###1##")
		print(prediction)

		obj = Img_class(img, prediction)

		print("###2##")
		print(obj.pred)

		cv2.imshow("Image", img)

		if cv2.waitKey(0) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

	return 0

if __name__ == "__main__":
	main()