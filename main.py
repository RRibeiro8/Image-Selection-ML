import numpy as np
import cv2
import os

import pickle

DB_PATH = "dataset/"

def ModifiedLaplacian(img):

	k = np.array([[-1], [2], [-1]])
	k_t = k.conj().T

	Lx = cv2.filter2D(img, cv2.CV_64F, k, borderType=cv2.BORDER_REPLICATE)
	Lx = cv2.filter2D(img, cv2.CV_64F, k_t, borderType=cv2.BORDER_REPLICATE)

	q = np.abs(Lx) + np.abs(Lx)

	q[np.isnan(q)] = np.min(q)

	return q

def extract_features(q):

	stddev = np.std(q)
	variance = np.var(q)
	weight = np.average(q)

	features = np.array([stddev, variance, weight])

	return features

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

		print(prediction)

		cv2.imshow("Image", img)

		if cv2.waitKey(0) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

	return 0

if __name__ == "__main__":
	main()