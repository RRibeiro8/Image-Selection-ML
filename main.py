import numpy as np
import cv2
import os

DB_PATH = "dataset/"

def main():

	list_dir = sorted(os.listdir(DB_PATH))

	for img_name in list_dir:

		img = cv2.imread(DB_PATH + img_name)

		cv2.imshow("Image", img)

		if cv2.waitKey(0) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

	return 0

if __name__ == "__main__":
	main()