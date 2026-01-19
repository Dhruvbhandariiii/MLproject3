import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

IMG_SIZE = 64

def load_images(folder):
    data = []
    labels = []

    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        
        if file.startswith('cat'):
            label = 0
        elif file.startswith('dog'):
            label = 1
        else:
            continue

        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.flatten()  # Convert image to 1D vector

        data.append(img)
        labels.append(label)

    return np.array(data), np.array(labels)

X, y = load_images(r"D:\ml\task3\dogs-vs-cats\svm\train6photos")

print("Total images:", X.shape)

X = X / 255.0

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

svm = SVC(kernel='linear', C=1)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

def predict_test_images(folder, model):
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.flatten() / 255.0
        img = img.reshape(1, -1)

        prediction = model.predict(img)

        if prediction == 0:
            print(file, "→ Cat 🐱")
        else:
            print(file, "→ Dog 🐶")

predict_test_images(r"D:\ml\task3\dogs-vs-cats\svm\test6photos", svm)
