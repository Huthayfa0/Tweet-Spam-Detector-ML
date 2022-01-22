import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import *
from sklearn.naive_bayes import BernoulliNB
from preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

plt.close('all')

# preprocess('data/train.csv')
data = pd.read_csv('data/preprocessedData.csv')
x = data.drop('Type', axis=1)
y = data['Type']
h = SelectKBest(chi2, k=5)
new_x = h.fit_transform(x, y)
x_train, x_test, y_train, y_test = train_test_split(new_x, y, test_size=0.2)


def test(model, title):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)
    print("Sensitivity (recall) score: ", recall_score(y_test, y_pred))
    print("precision score: ", precision_score(y_test, y_pred))
    print("f1 score: ", f1_score(y_test, y_pred))
    print("accuracy score: ", accuracy_score(y_test, y_pred))
    print("ROC AUC: {}".format(roc_auc_score(y_test, y_pred_prob[:, 1])))

    RocCurveDisplay.from_estimator(model, x_test, y_test)
    plt.title(title)
    plt.show()
    DetCurveDisplay.from_estimator(model, x_test, y_test)
    plt.title(title)
    plt.show()
    ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
    plt.title(title)
    plt.show()
    PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
    plt.title(title)
    plt.show()


"""
for n in x.columns:
    if n == '0':
        break
    plt.title(n)
    plt.scatter(x[n], y, color='red', marker='+')
    plt.show()
"""

qualityTree = tree.DecisionTreeClassifier()
test(qualityTree, 'Decision Tree classifier')
i = 0
plt.barh(h.get_feature_names_out(), qualityTree.feature_importances_)
plt.title('Decision Tree classifier')
plt.show()
dot_data = StringIO()
export_graphviz(qualityTree, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=h.get_feature_names_out(), class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Tree.png')
Image(graph.create_png())
model_nn = MLPClassifier()
# test(model_nn, 'Multi-layer Perceptron classifier')

model_nb = BernoulliNB()
# test(model_nb, 'Bernoulli Naive bayes')
