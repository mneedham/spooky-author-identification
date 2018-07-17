import datetime

from neo4j.v1 import GraphDatabase, basic_auth
from sklearn import datasets
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB

from util.model_persistence import ModelPersistence

model_persistence = ModelPersistence(GraphDatabase.driver("bolt://localhost", auth=basic_auth("neo4j", "neo")))

# iris = datasets.load_iris()
#
# # GaussianNB
#
# gnb = GaussianNB()
# gnb.fit(iris.data, iris.target)
#
# model_name = "{0}-{1}".format(gnb.__class__.__name__, int(datetime.datetime.timestamp(datetime.datetime.now())))
# model_persistence.save(gnb, model_name, ["classes_", "sigma_", "theta_", "class_prior_"])
#
# y_pred = gnb.predict(iris.data)
# print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0], (iris.target != y_pred).sum()))
#
# new_gnb = model_persistence.load(model_name)
#
# y_pred = new_gnb.predict(iris.data)
# print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0], (iris.target != y_pred).sum()))
#
# # Lasso
#
# reg = linear_model.Lasso(alpha=0.1)
# reg.fit([[0, 0], [1, 1]], [0, 1])
#
# model_name = "{0}-{1}".format(reg.__class__.__name__, int(datetime.datetime.timestamp(datetime.datetime.now())))
# model_persistence.save(reg, model_name, ["coef_", "n_iter_", "intercept_"])
#
# print(reg.predict([[1, 1]]))
#
# new_reg = model_persistence.load(model_name)
# print(new_reg.predict([[1, 1]]))

# Bayesian Ridge Regression

X = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
Y = [0., 1., 2., 3.]
reg = linear_model.BayesianRidge()
reg.fit(X, Y)

model_name = "{0}-{1}".format(reg.__class__.__name__, int(datetime.datetime.timestamp(datetime.datetime.now())))
model_persistence.save(reg, model_name, ["coef_", "intercept_"])

print(reg.predict([[1, 0.]]))

new_reg = model_persistence.load(model_name)
print(new_reg.predict([[1, 0.]]))