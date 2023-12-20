from features import load_dataset
from sklearn.ensemble import IsolationForest
import argparse
import pickle

# load and quantify our image dataset
print("[INFO] preparing dataset...")
data = load_dataset(r'Data/non_faulty_images',bins=(3, 3, 3))
# train the anomaly detection model
print("[INFO] fitting anomaly detection model...")
model = IsolationForest(n_estimators=100, contamination=0.01,
    random_state=42)
model.fit(data)

# serialize the anomaly detection model to disk
f = open(r'model/model.pb', "wb")
f.write(pickle.dumps(model))
f.close()
