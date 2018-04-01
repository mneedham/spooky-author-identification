import datetime
import importlib

import numpy as np
from neo4j.v1 import GraphDatabase, basic_auth
from sklearn import datasets
from sklearn import linear_model


# save the binary representation and then load it back using reshape
#  np.frombuffer(raw_value.tobytes()).reshape(raw_value.shape)
from sklearn.naive_bayes import GaussianNB


class ModelPersistence:
    def __init__(self, driver):
        self.driver = driver

    save_model_query = """\
    MERGE (model:Model {name: {modelName} })
    SET model.module = {module}, model.class = {class}
    WITH model
    UNWIND keys({fields}) AS f
    CREATE (field:Field)
    SET field.key = f, 
        field.value = {fields}[f].value,
        field.type = {fields}[f].type,
        field.dataType = {fields}[f].dataType,
        field.shape = {fields}[f].shape

    MERGE (model)-[:HAS_FIELD]->(field)
    """

    def save(self, model, name, fields):
        params = {
            "modelName": name,
            "module": model.__module__,
            "class": model.__class__.__name__,
            "fields": {}
        }
        attrs = params["fields"]
        for field in fields:
            raw_value = getattr(model, field)
            coerced_value = raw_value

            if isinstance(raw_value, np.ndarray):
                coerced_value = raw_value.tobytes()

            data_type = "{0}.{1}".format(type(raw_value).__module__, type(raw_value).__name__)

            attrs[field] = {
                "value": coerced_value,
                "type": data_type
            }

            if data_type == "numpy.ndarray":
                attrs[field]["dataType"] = raw_value.dtype.name
                attrs[field]["shape"] = list(raw_value.shape)

        with self.driver.session() as session:
            session.run(self.save_model_query, params)

    load_model_query = """\
    MATCH (model:Model {name: {modelName} })-[:HAS_FIELD]->(field)
    RETURN model, collect(field) AS fields    
    """

    def load(self, name):
        with self.driver.session() as session:
            result = session.run(self.load_model_query, {"modelName": name})
            only_row = result.peek()
            model = only_row["model"]
            fields = only_row["fields"]

        ModelClass = getattr(importlib.import_module(model["module"]), model["class"])
        instance = ModelClass()
        for field in fields:
            if field["type"] == "numpy.ndarray":
                value = np.frombuffer(field["value"], dtype=field["dataType"]).reshape(tuple(field["shape"]))
            else:
                value = field["value"]
            setattr(instance, field["key"], value)
        return instance


persister = ModelPersistence(GraphDatabase.driver("bolt://localhost", auth=basic_auth("neo4j", "neo")))

iris = datasets.load_iris()

# GaussianNB

gnb = GaussianNB()
gnb.fit(iris.data, iris.target)

model_name = "{0}-{1}".format(gnb.__class__.__name__, int(datetime.datetime.timestamp(datetime.datetime.now())))
persister.save(gnb, model_name, ["classes_", "sigma_", "theta_", "class_prior_"])

y_pred = gnb.predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0], (iris.target != y_pred).sum()))

new_gnb = persister.load(model_name)

y_pred = new_gnb.predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0], (iris.target != y_pred).sum()))

# Lasso

reg = linear_model.Lasso(alpha=0.1)
reg.fit([[0, 0], [1, 1]], [0, 1])

model_name = "{0}-{1}".format(reg.__class__.__name__, int(datetime.datetime.timestamp(datetime.datetime.now())))
persister.save(reg, model_name, ["coef_", "n_iter_", "intercept_"])

print(reg.predict([[1, 1]]))

new_reg = persister.load(model_name)

print(new_reg.predict([[1, 1]]))
