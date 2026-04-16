from flask import render_template, request
from services.pipeline_service import run_pipeline

def pipeline_page():
    method = request.form.get("method", "standard")
    thresholdWhite = float(request.form.get("thresholdWhite", 0.5))
    thresholdBlack = float(request.form.get("thresholdBlack", 0.5))
    thresholdNDVI = float(request.form.get("thresholdNDVI", 0.5))
    wavelength = request.form.get("wavelength", "756")
    analysis = request.form.get("analysis", "profile")

    data = run_pipeline(method, thresholdWhite, thresholdBlack, thresholdNDVI, wavelength, analysis)

    data["method"] = method
    data["thresholdWhite"] = thresholdWhite
    data["thresholdBlack"] = thresholdBlack
    data["thresholdNDVI"] = thresholdNDVI
    data["wavelength"] = wavelength
    data["analysis"] = analysis

    return render_template("pipeline-result.html", data=data)