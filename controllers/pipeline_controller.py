from flask import render_template, request
from services.pipeline_service import run_pipeline

def pipeline_page():
    method = request.form.get("method", "standard")
    threshold = float(request.form.get("threshold", 0.5))
    wavelength = request.form.get("wavelength", "756")
    analysis = request.form.get("analysis", "profile")

    data = run_pipeline(method, threshold, wavelength, analysis)

    data["method"] = method
    data["threshold"] = threshold
    data["wavelength"] = wavelength
    data["analysis"] = analysis

    return render_template("pipeline-result.html", data=data)