# import the necessary packages
from features import quantify_image
import argparse
import pickle
import cv2
# load the anomaly detection model
print("[INFO] loading anomaly detection model...")
model = pickle.loads(open(r'model/model.pb', "rb").read())
# load the input image, convert it to the HSV color space, and
# quantify the image in the *same manner* as we did during training
image = cv2.imread(r'Data/faulty_images/b service...jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
features = quantify_image(hsv, bins=(3, 3, 3))
# use the anomaly detector model and extracted features to determine
# if the example image is an anomaly or not
preds = model.predict([features])[0]
label = "High Temperature" if preds == -1 else "Normal Temperature"
color = (0, 0, 255) if preds == -1 else (0, 255, 0)
# draw the predicted label text on the original image
cv2.putText(image, label, (10,  25), cv2.FONT_HERSHEY_SIMPLEX,
	0.7, color, 2)
# display the image
cv2.imshow("Output", image)
cv2.waitKey(0)