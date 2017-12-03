import pandas as pd


def generate_output_file(pipe):
    test_df = pd.read_csv("test.csv")

    predictions = pipe.predict_proba(test_df["text"])

    output = pd.DataFrame(predictions, columns=pipe.classes_)
    output["id"] = test_df["id"]
    output.to_csv("output.csv", index=False)
