from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        text = request.form["text"]

        # Convert text
        text_vector = vectorizer.transform([text])

        # Predict
        result = model.predict(text_vector)

        prediction = result[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)