from flask import Flask, render_template, request,url_for,redirect
from static.assets.members.members import members
from controllers.pipeline_controller import pipeline_page
from services.mongo_service import get_mongo_folders

app = Flask(__name__)


@app.route("/")
def home_page():
    return render_template("homepage.html")


@app.route("/resources")
def resources_page():
    return render_template("resources.html")


@app.route("/upload-images")
def run_pipeline_page():
    folders = get_mongo_folders()
    return render_template("pipeline-upload.html",folders=folders)


@app.route("/run-pipeline", methods=["GET", "POST"])
def submit_pipeline():
    if request.method == "POST":
        return pipeline_page()

    # If someone just types the URL in manually (GET),
    # redirect them back to the upload page.
    return redirect(url_for('run_pipeline_page'))

@app.route("/about-us")
def about_us_page():
    return render_template("about-us.html", members=members)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
