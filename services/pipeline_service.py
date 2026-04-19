from utils.pipeline_service import run_pipeline as core_run_pipeline
from utils.pipeline_service import plot_data


def run_pipeline_service(
        sample_id,
        method,
        thresholdWhite,
        thresholdBlack,
        thresholdNDVI,
        wavelength,
        analysis
):
    # 1. Call REAL pipeline
    result = core_run_pipeline(
        sample_id=sample_id,
        thresholds=[thresholdBlack, thresholdWhite, thresholdNDVI],
        method=method,
        wavelength=float(wavelength)
    )

    # 2. Get graph (separate endpoint)
    # plot_result = plot_data(analysis, sample_id)
    graph_b64 = None

    # 3. Transform indices for your template
    indices = []
    indexes_dict = result.get("indexes", {})
    images = result.get("images", {})

    # indexes_dict comes from pandas → nested dict
    for key in indexes_dict:
        value = list(indexes_dict[key].values())[0]

        indices.append({
            "name": key,
            "label": key,
            "value": float(value)
        })

    return {
        "indices": indices,

        # images from pipeline (BASE64 now)
        "before_image": images.get("before"),
        "after_image": images.get("after"),
        "pca_image": images.get("pca_reduction"),
        "graph": graph_b64,

        # keep UI state
        "method": method,
        "thresholdWhite": thresholdWhite,
        "thresholdBlack": thresholdBlack,
        "thresholdNDVI": thresholdNDVI,
        "wavelength": wavelength,
        "analysis": analysis
    }
