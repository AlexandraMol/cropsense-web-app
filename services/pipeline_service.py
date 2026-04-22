from utils.cropsense_all import *
from utils.plot_utils import *
from utils.indices import INDEX_META

# -----------------------------
# INDEX FUNCTION MAP
# -----------------------------
INDEX_FUNCTIONS = {
    "NDVI": calculate_mean_ndvi,
    "GNDVI": calculate_mean_gndvi,
    "RVI": calculate_mean_rvi,
    "WI": calculate_mean_wi,
    "NDWI": calculate_mean_ndwi,
    "SIPI": calculate_mean_sipi,
    "PRI": calculate_mean_pri,
    "ARI": calculate_mean_ari,
    "CARI": calculate_mean_cari,
}

# -----------------------------
# CACHE (important)
# -----------------------------
PIPELINE_CACHE = {}


# -----------------------------
# FULL PIPELINE (RUN ONCE)
# -----------------------------
def run_full_pipeline(sample_id):

    run_pipeline()

    downloaded_paths = export_file_from_db(
        "./generated",
        plant_id=sample_id,
        sensor_type="hyperspectral"
    )

    hdr_files = [f for f in downloaded_paths if f.endswith(".hdr")]

    var_normal = next(f for f in hdr_files if "DARKREF" not in f and "WHITEREF" not in f)
    var_dark = next(f for f in hdr_files if "DARKREF" in f)
    var_white = next(f for f in hdr_files if "WHITEREF" in f)

    hyperspectral_data = prepare_hyperspectral_data(
        var_normal, var_dark, var_white, 'purple', 0.07, 0.72, 0.01
    )

    sick, healthy = separate_data(hyperspectral_data, cut_line=800)

    return {
        "hyperspectral_data": hyperspectral_data,
        "healthy": healthy,
        "sick": sick,

        # STATIC IMAGES (CACHE ONCE)
        "fig": fig_to_base64(show_hyperspectral_image(hyperspectral_data)),
        "healthy_fig": fig_to_base64(show_hyperspectral_image(healthy)),
        "sick_fig": fig_to_base64(show_hyperspectral_image(sick)),

        "healthy_graph": fig_to_base64(get_hyperspectral_graph(healthy)),
        "sick_graph": fig_to_base64(get_hyperspectral_graph(sick)),
        "compare_graph": fig_to_base64(
            get_multiple_hyperspectral_graphs(
                [healthy, sick],
                ["Control Plant", "Infected Plant"]
            )
        )
    }


# -----------------------------
# DYNAMIC ANALYSIS
# -----------------------------
def compute_analysis(hyperspectral_data, healthy, sick, selected_indices):

    # fallback (if nothing selected)
    if not selected_indices:
        selected_indices = list(INDEX_FUNCTIONS.keys())

    # -----------------------------
    # INDICES VALUES
    # -----------------------------
    indices_values = {}

    for idx in selected_indices:
        func = INDEX_FUNCTIONS.get(idx)

        if not func:
            continue

        try:
            indices_values[idx] = func(hyperspectral_data)
        except Exception as e:
            print(f"Error computing {idx}: {e}")

    indices = [
        {
            "name": key,
            "label": INDEX_META.get(key, key),
            "value": float(value)
        }
        for key, value in indices_values.items()
    ]

    # -----------------------------
    # COMPARISON TABLE
    # -----------------------------
    comparison = format_comparison_for_frontend(
        compare_plants_indices_json(
            [healthy, sick],
            ["Control Plant", "Infected Plant"]
        )
    )

    # -----------------------------
    # INDEX MAPS
    # -----------------------------
    maps_b64 = {}

    for idx in selected_indices:

        index_map, cmap = compute_index_map(hyperspectral_data, idx)

        if index_map is None:
            continue

        fig_map = get_index_map_figure(index_map, cmap, idx)

        maps_b64[idx] = fig_to_base64(fig_map) if fig_map else None

    return {
        "indices": indices,
        "comparison": comparison,
        "index_maps": maps_b64
    }


# -----------------------------
# SERVICE ENTRYPOINT
# -----------------------------
def run_pipeline_service(
    sample_id,
    method,
    thresholdWhite,
    thresholdBlack,
    thresholdNDVI,
    wavelength,
    analysis,
    selected_indices
):

    # -----------------------------
    # 1. CACHE PIPELINE
    # -----------------------------
    if sample_id not in PIPELINE_CACHE:
        PIPELINE_CACHE[sample_id] = run_full_pipeline(sample_id)

    base = PIPELINE_CACHE[sample_id]

    # -----------------------------
    # 2. DYNAMIC ANALYSIS ONLY
    # -----------------------------
    analysis_data = compute_analysis(
        base["hyperspectral_data"],
        base["healthy"],
        base["sick"],
        selected_indices
    )

    # -----------------------------
    # FINAL RESPONSE
    # -----------------------------
    return {
        "sample_id": sample_id,

        # STATIC IMAGES
        "fig": base["fig"],
        "healthy_plant_fig": base["healthy_fig"],
        "sick_plant_fig": base["sick_fig"],
        "healthy_plant_graph": base["healthy_graph"],
        "sick_plant_graph": base["sick_graph"],
        "compare_graph": base["compare_graph"],

        # DYNAMIC DATA
        "indices": analysis_data["indices"],
        "comparison": analysis_data["comparison"],
        "index_maps": analysis_data["index_maps"],

        "selected_indices": selected_indices,

        # UI STATE
        "method": method,
        "thresholdWhite": thresholdWhite,
        "thresholdBlack": thresholdBlack,
        "thresholdNDVI": thresholdNDVI,
        "wavelength": wavelength,
        "analysis": analysis
    }