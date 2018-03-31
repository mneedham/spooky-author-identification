import importlib

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB


def save_model(model):
    fields = ["classes_", "sigma_", "theta_", "class_prior_"]
    saved_model = {
        "module": model.__module__,
        "class": model.__class__.__name__,
        "fields": {}
    }
    attrs = saved_model["fields"]
    for field in fields:
        attrs[field] = getattr(model, field)
    return saved_model


iris = datasets.load_iris()

gnb = GaussianNB()
gnb.fit(iris.data, iris.target)

saved_model = save_model(gnb)

y_pred = gnb.predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0], (iris.target != y_pred).sum()))

# reload model

ModelClass = getattr(importlib.import_module(saved_model["module"]), saved_model["class"])
new_gnb = ModelClass()

for attr in saved_model["fields"]:
    setattr(new_gnb, attr, saved_model["fields"][attr])

y_pred = new_gnb.predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0], (iris.target != y_pred).sum()))
