import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import *
from sklearn.naive_bayes import GaussianNB
from preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

# preprocess('data/train.csv')
data = pd.read_csv('data/preprocessedData.csv')
x = data.drop('Type', axis=1)
y = data['Type']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
"""
for n in x.columns:
    plt.title(n)
    plt.scatter(x[n],y,color = 'red', marker = '+')
    plt.show()


qualityTree = tree.DecisionTreeClassifier()
qualityTree.fit(x_train, y_train)
plt.barh(x.columns, qualityTree.feature_importances_)
plt.show()
ACC_train_rf = qualityTree.score(x_train, y_train)
ACC_test_rf = qualityTree.score(x_test, y_test)
print(ACC_train_rf)
print(ACC_test_rf)

dot_data = StringIO()
export_graphviz(qualityTree, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=x.columns, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())
"""
"""
model_nn = MLPClassifier()
model_nn.fit(x_train,y_train)
ACC_train_nn = model_nn.score(x_train,y_train)
ACC_test_nn = model_nn.score(x_test,y_test)
print(ACC_train_nn)
print(ACC_test_nn)
"""
model_nb = GaussianNB()
model_nb.fit(x_train,y_train)
ACC_train_nb = model_nb.score(x_train,y_train)
ACC_test_nb = model_nb.score(x_test,y_test)
print(ACC_train_nb)
print(ACC_test_nb)
y_pred = model_nb.predict(x_test)
y_pred_prob = model_nb.predict_proba(x_test)
print('test-dataset confusion matrix:\n', confusion_matrix(y_test,y_pred))
print("Sensitivity (recall) score: ", recall_score(y_test,y_pred))
print("precision score: ", precision_score(y_test,y_pred))
print("f1 score: ", f1_score(y_test,y_pred))
print("accuracy score: ", accuracy_score(y_test,y_pred))
print("ROC AUC: {}".format(roc_auc_score(y_test, y_pred_prob[:,1])))
