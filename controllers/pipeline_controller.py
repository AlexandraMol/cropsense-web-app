from flask import render_template, request
from services.pipeline_service import run_pipeline_service


def pipeline_page():
    data_source = request.form.get("data_source")

    print("DATA SOURCE:", data_source)
    data = None

    if data_source == "upload":
        zip_file = request.files.get("zip_file")
        print("ZIP FILE:", zip_file.filename if zip_file else None)

        # TODO: later connect upload pipeline
        return "Upload pipeline not implemented yet"

    elif data_source == "mongodb":

        data = run_pipeline_service(
            sample_id=request.form.get("mongo_folder"),
            method=request.form.get("method","standard"),
            thresholdWhite=float(request.form.get("thresholdWhite","0.5")),
            thresholdBlack=float(request.form.get("thresholdBlack","0.5")),
            thresholdNDVI=float(request.form.get("thresholdNDVI","0.5")),
            wavelength = request.form.get("wavelength", "756"),
            analysis=request.form.get("analysis","profile")
        )

    return render_template("pipeline-result.html", data=data)
