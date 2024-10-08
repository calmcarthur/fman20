import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Training!
class_1_train = np.array([0.4003, 0.3985, 0.3998, 0.3997])
class_2_train = np.array([0.2554, 0.3139, 0.2627, 0.3802])
class_3_train = np.array([0.5632, 0.7687, 0.0524, 0.7586])

# Combine both
X_train = np.concatenate([class_1_train, class_2_train, class_3_train]).reshape(-1, 1)
y_train = np.array([1]*4 + [2]*4 + [3]*4)  # Labels for Class 1, Class 2, Class 3

# Testing!
class_1_test = np.array([0.4015, 0.3995, 0.3991])
class_2_test = np.array([0.3247, 0.3360, 0.2974])
class_3_test = np.array([0.4443, 0.5505, 0.6469])

# Combine test and labels (this is overkill for this)
X_test = np.concatenate([class_1_test, class_2_test, class_3_test]).reshape(-1, 1)
y_test = np.array([1]*3 + [2]*3 + [3]*3)  # Labels for test data

# Train KNN classifier (with n=1, this is just absolute difference)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Predict on test data
predictions = knn.predict(X_test)

# Calculate number of correctly classified measurements
correct_classifications = np.sum(predictions == y_test)
print(correct_classifications)
