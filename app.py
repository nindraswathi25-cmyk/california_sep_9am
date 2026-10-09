from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load model
obj = joblib.load("colifornia.joblib")

model = obj["Model"]
columns = obj["columns"]
print(columns)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict")
def predict():

    input_data = []

    for i in columns:
        value = request.args.get(i)
        input_data.append(float(value))

    input_data = np.array([input_data])

    prediction = model.predict(input_data)

    return render_template(
        "index.html",
        prediction=prediction[0]
    )


if __name__ == "__main__":
    app.run(debug=True)

