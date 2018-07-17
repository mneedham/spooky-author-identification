import importlib

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model


def save_model(model, fields):
    saved_model = {
        "module": model.__module__,
        "class": model.__class__.__name__,
        "fields": {}
    }
    attrs = saved_model["fields"]
    for field in fields:
        attrs[field] = getattr(model, field)
    return saved_model


def load_model(model):
    ModelClass = getattr(importlib.import_module(model["module"]), model["class"])
    instance = ModelClass()
    for attr in model["fields"]:
        setattr(instance, attr, model["fields"][attr])
    return instance


iris = datasets.load_iris()

# GaussianNB

gnb = GaussianNB()
gnb.fit(iris.data, iris.target)

saved_model = save_model(gnb, ["classes_", "sigma_", "theta_", "class_prior_"])

y_pred = gnb.predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0], (iris.target != y_pred).sum()))

new_gnb = load_model(saved_model)

y_pred = new_gnb.predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0], (iris.target != y_pred).sum()))

# Lasso

reg = linear_model.Lasso(alpha = 0.1)
reg.fit([[0, 0], [1, 1]], [0, 1])

saved_model = save_model(reg, ["coef_", "n_iter_", "intercept_"])

print(reg.predict([[1, 1]]))

new_reg = load_model(saved_model)

print(new_reg.predict([[1, 1]]))
