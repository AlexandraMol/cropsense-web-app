from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home_page():
    return render_template("homepage.html")

@app.route("/resources")
def resources_page():
    return render_template("resources.html")

if __name__ == "__main__":
    app.run(debug=True)