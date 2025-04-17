import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(random_state=42, solver='liblinear', multi_class='ovr')
model.fit(X_train, y_train)

# Save the model to a file
filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))
print(f"Model saved to {filename}")