#Author: Sanket Sabharwal

# Here the required libraries are imported
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Here we import the dataset and prepare the matrix for further processing
dataset = pd.read_csv('DataSet1.csv')
X = dataset.iloc[:, 0:10].values
y = dataset.iloc[:, 10].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred3 = classifier.predict(X_test)

# Making the Confusion Matrix
cm3 = confusion_matrix(y_test, y_pred3)


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred2 = classifier.predict(X_test)

# Making the Confusion Matrix
cm2 = confusion_matrix(y_test, y_pred2)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred4 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm4 = confusion_matrix(y_test, y_pred4)


#Print the accuracy percentages
acc2 = int(accuracy_score(y_test, y_pred2)*100)
acc3 = int(accuracy_score(y_test, y_pred3)*100)
acc4 = int(accuracy_score(y_test, y_pred4)*100)

print(classification_report(y_test,y_pred4))
print("Accuracy Using KNN: " + str(acc4)+'%\n')
print(classification_report(y_test,y_pred3))
print("Accuracy Using Decision Tree: " + str(acc3)+'%\n') 
print(classification_report(y_test,y_pred2))
print("Accuracy Using Random Forest: " + str(acc2)+'%\n') 