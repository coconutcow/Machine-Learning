#Author: Sanket Sabharwal

# Here the required libraries are imported
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

# Importing the dataset
dataset = pd.read_csv('DataSet3.csv')
X = dataset.iloc[:, 0:85].values
Y = dataset.iloc[:, 85].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 1)

# Feature Scalinga as required by Neural Networks
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialising MLP
mlp = MLPClassifier(hidden_layer_sizes=(85,85,85),max_iter=5000)

#Fit the MLP
mlp.fit(X_train,y_train)
y_pred = mlp.predict(X_test)
 
# Making the Confusion Matrix and Printing the accuracy report
cm = confusion_matrix(y_test, y_pred)

acc = int(accuracy_score(y_test, y_pred)*100)
print(classification_report(y_test,y_pred))
print("Accuracy Using MLP Classifier: " + str(acc)+'%\n')