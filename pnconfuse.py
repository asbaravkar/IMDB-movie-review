import numpy as np
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('IMDB Dataset.csv', delimiter = ',')
dataset['sentiment'] = dataset['sentiment'].map({'positive': 1, 'negative': 0})


# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup


#To remove HTML tags
corpus = []
print('---------Start----------')
for i in range(0, 2000):
    review = BeautifulSoup( dataset['review'][i], "lxml").text
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
print('-----------------next1--------------------')


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(corpus).toarray()
dataset
y = dataset.iloc[0:2000, 1].values
print('----------------next2--------------------')


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
print('--------------------next3----------------------')


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion martrix for Naive Bayes is :- \n")
print(cm)
skplt.metrics.plot_confusion_matrix(y_test, y_pred, figsize = (15, 15), text_fontsize = 'large', title = 'Gaussian Naive Bayes')
plt.show()


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0)
classifier1.fit(X_train, y_train)


# Predicting the Test set results
y_predL = classifier1.predict(X_test)


# Making the Confusion Matrix
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_predL)
print("Confusion martrix for Logistic regression is :- \n")
print(cm1)
skplt.metrics.plot_confusion_matrix(y_test, y_predL, figsize = (15, 15), text_fontsize = 'large', title = 'Logistic Regression')
plt.show()


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier2.fit(X_train, y_train)


# Predicting the Test set results
y_predD = classifier2.predict(X_test)

# Making the Confusion Matrix
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_predD)
print("Confusion martrix for Decision Tree is :- \n")
print(cm2)
skplt.metrics.plot_confusion_matrix(y_test, y_predD, figsize = (15, 15), text_fontsize = 'large', title = 'Decision Tree')
plt.show()


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier3.fit(X_train, y_train)


# Predicting the Test set results
y_predR = classifier3.predict(X_test)


# Making the Confusion Matrix
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, y_predR)
print("Confusion martrix for Random Forest is :- \n")
print(cm3)
skplt.metrics.plot_confusion_matrix(y_test, y_predR, figsize = (15, 15), text_fontsize = 'large', title = 'Random Forest')
plt.show()


# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier4 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier4.fit(X_train, y_train)


# Predicting the Test set results
y_predK = classifier4.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
import matplotlib.pyplot as plt
cm4 = confusion_matrix(y_test, y_predK)
print("Confusion martrix for KNN is :- \n")
print(cm4)
skplt.metrics.plot_confusion_matrix( y_test, y_predK, figsize = (15, 15), text_fontsize = 'large', title = 'K Nearest Neighbors')
plt.show()



#importing visualization libraries
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn

#plotting calibration curve

rf = RandomForestClassifier()
lr = LogisticRegression()
nb = GaussianNB()
kn = KNeighborsClassifier()
dt = DecisionTreeClassifier()

rf_probas = rf.fit(X_train, y_train).predict_proba(X_test)
lr_probas = lr.fit(X_train, y_train).predict_proba(X_test)
nb_probas = nb.fit(X_train, y_train).predict_proba(X_test)
kn_probas = kn.fit(X_train, y_train).predict_proba(X_test)
dt_probas = kn.fit(X_train, y_train).predict_proba(X_test)

probas_list = [rf_probas, lr_probas, nb_probas, kn_probas, dt_probas]
clf_names = ['Random Forest', 'Logistic Regression', 'Gaussian Naive Bayes', 'K Nearest Neighbors', 'Decision Tree']
skplt.metrics.plot_calibration_curve(y_test, probas_list, clf_names)
plt.show()
