from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def home_page():
    return render_template("homepage.html")


@app.route("/resources")
def resources_page():
    return render_template("resources.html")


@app.route("/upload-images")
def run_pipeline_page():
    return render_template("pipeline-upload.html")


@app.route("/run-pipeline", methods=["GET", "POST"])
def submit_pipeline():
    if request.method == "POST":
        return render_template("pipeline-result.html")

    # If someone just types the URL in manually (GET),
    # redirect them back to the upload page.
    return redirect(url_for('run_pipeline_page'))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
