import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def extract_sift_features(images):
    sift = cv2.SIFT_create()
    keypoints_list = []
    descriptors_list = []
    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
    return keypoints_list, descriptors_list

def build_vocabulary(descriptors_list, vocabulary_size):
    descriptors_concat = np.concatenate(descriptors_list)
    kmeans = KMeans(n_clusters=vocabulary_size)
    kmeans.fit(descriptors_concat)
    return kmeans

def extract_features(image, vocabulary):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None:
        return np.zeros(vocabulary.cluster_centers_.shape[0])
    else:
        labels = vocabulary.predict(descriptors)
        features = np.bincount(labels, minlength=vocabulary.cluster_centers_.shape[0])
        return features

def main():
    categories = ['spoon', 'fork', 'mixer', 'chopstick', 'rice-spoon']
    vocabulary_size = 50
    all_images = []
    all_labels = []

    # Load images and extract SIFT features
    for category in categories:
        folder = os.path.join('question6_photos', category)
        images = load_images_from_folder(folder)
        all_images.extend(images)
        all_labels.extend([category] * len(images))

    keypoints_list, descriptors_list = extract_sift_features(all_images)

    # Build vocabulary
    kmeans = build_vocabulary(descriptors_list, vocabulary_size)

    # Extract features for each image
    features = []
    for img in all_images:
        features.append(extract_features(img, kmeans))
    features = np.array(features)

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train SVM classifier
    svm = SVC(kernel='linear')
    svm.fit(features_scaled, all_labels)

    # Evaluate performance
    predicted_labels = svm.predict(features_scaled)
    accuracy = np.mean(predicted_labels == all_labels)
    print("Accuracy on training set:", accuracy)

    # Visualize recognition results
    for i in range(len(all_images)):
        img = cv2.cvtColor(all_images[i], cv2.COLOR_GRAY2BGR)
        actual_label = all_labels[i]
        predicted_label = predicted_labels[i]
        text = f"Actual: {actual_label}, Predicted: {predicted_label}"
        cv2.imshow(text, img)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
