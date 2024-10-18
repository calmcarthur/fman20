from sklearn import svm
import numpy as np
import cv2
from skimage.feature import hog
from skimage import morphology
from skimage.measure import label, regionprops
import os
import time
from benchmarking.benchmark_assignment3 import benchmark_assignment3

FEATURE_LENGTH = 7357  # Ensure this is consistent

def im2segment(im):
    threshold = 40  # Threshold value
    thresholded_im = np.where(im > threshold, 1, 0)
    cleaned_image = morphology.remove_small_objects(thresholded_im.astype(bool), min_size=5)
    labels = label(cleaned_image, connectivity=2)
    regions = regionprops(labels)

    segments = []
    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        segment = np.zeros_like(im)
        segment[min_row:max_row, min_col:max_col] = thresholded_im[min_row:max_row, min_col:max_col] * \
                                                   (labels[min_row:max_row, min_col:max_col] == region.label)
        segments.append((min_col, segment))

    segments.sort(key=lambda x: x[0])
    segments = [segment for _, segment in segments]

    return segments

def segment2feature(segment):
    rows, columns = np.nonzero(segment)
    center_of_mass_x = np.mean(columns)
    center_of_mass_y = np.mean(rows)
    shift_x = (segment.shape[1] // 2) - center_of_mass_x
    shift_y = (segment.shape[0] // 2) - center_of_mass_y
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    translated = cv2.warpAffine(segment, M, (segment.shape[1], segment.shape[0]))

    features = []

    width = np.sum(np.any(translated, axis=0))
    features.append(width)

    height = translated.shape[0]
    top_heaviness = np.sum(translated[:height // 2]) / np.sum(translated)
    features.append(top_heaviness)

    right_heaviness = np.sum(translated[:, translated.shape[1] // 2:]) / np.sum(translated)
    features.append(right_heaviness)

    inverted_segment = cv2.bitwise_not(translated)
    num_holes, _ = cv2.connectedComponents(inverted_segment)
    features.append(num_holes)

    vertical_symmetry = np.sum(translated == np.flip(translated, axis=0)) / translated.size
    features.append(vertical_symmetry)

    horizontal_symmetry = np.sum(translated == np.flip(translated, axis=1)) / translated.size
    features.append(horizontal_symmetry)

    moments = cv2.HuMoments(cv2.moments(translated)).flatten()
    features.extend(moments)

    fd, _ = hog(translated, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    features.extend(fd)

    features = np.array(features)
    if len(features) > FEATURE_LENGTH:
        features = features[:FEATURE_LENGTH]
    else:
        features = np.pad(features, (0, FEATURE_LENGTH - len(features)), 'constant')

    normalized_features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)

    return normalized_features.reshape(-1, 1)

def class_train(X, Y):
    clf = svm.SVC(kernel='linear')  # Using a linear kernel, you can also try 'rbf'
    clf.fit(X, Y)
    return clf

def feature2class(x, clf):
    return clf.predict(x.T)

if __name__ == "__main__":

    start_time = time.time()

    # Loading training data
    segment_train = np.load('ocrsegmentdata.npy')
    Y_train = np.load('ocrsegmentgt.npy').flatten()

    # Transforming training data into features
    X_train = []
    for i in range(len(segment_train)):
        features = segment2feature(segment_train[i])
        X_train.append(features.flatten())
    X_train = np.array(X_train)

    # Training the classifier (SVM)
    clf = class_train(X_train, Y_train)

    # Run benchmark on all datasets
    datasets = ['short1', 'short2', 'home1', 'home2', 'home3']

    mode = 0  # debug modes: 0 with no plots, 1 with some plots

    for ds in datasets:
        datadir = os.path.join('datasets', ds)
        hitrate, confmat, allres, alljs, alljfg, allX, allY = benchmark_assignment3(im2segment, segment2feature, feature2class, clf, datadir, mode)
        print(ds + f', Hitrate = {hitrate*100:0.5}%')

    end_time = time.time()
    print(end_time - start_time)
