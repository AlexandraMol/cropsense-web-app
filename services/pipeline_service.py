from utils.pipeline import *
from services.mongo_service import build_hyperspectral_data

def run_pipeline_service(
        sample_id,
        method,
        thresholdWhite,
        thresholdBlack,
        thresholdNDVI,
        wavelength,
        analysis
):

    paths = build_hyperspectral_data(sample_id)

    if not paths:
        return None

    signature = prepare_hyperspectral_data(
        paths["raw"],
        paths["dark"],
        paths["white"],
        background="darken",
        min_brightness=thresholdBlack,
        max_brightness=thresholdWhite,
        ndvi_threshold=thresholdNDVI
    )

    if signature is None:
        return None

    # 🔥 ONLY compute indices (this is the whole point)
    indices = {
        "NDVI": calculate_mean_ndvi(signature),
        "GNDVI": calculate_mean_gndvi(signature),
        "RVI": calculate_mean_rvi(signature),
        "WI": calculate_mean_wi(signature),
        "NDWI": calculate_mean_ndwi(signature),
        "SIPI": calculate_mean_sipi(signature),
        "PRI": calculate_mean_pri(signature),
        "ARI": calculate_mean_ari(signature),
        "CARI": calculate_mean_cari(signature),
    }

    results = [
        {
            "name": k,
            "label": k,
            "value": float(v) if v is not None else 0.0
        }
        for k, v in indices.items()
    ]

    return {
        "indices": results,

        # optional UI stuff (keep for now)
        "method": method,
        "thresholdWhite": thresholdWhite,
        "thresholdBlack": thresholdBlack,
        "thresholdNDVI": thresholdNDVI,
        "wavelength": wavelength,

        # dummy placeholders for UI (you can remove later)
        "before_image": "assets/images/before.jpg",
        "after_image": "assets/images/after.jpg",
        "graph": "generated/output.png"
    }