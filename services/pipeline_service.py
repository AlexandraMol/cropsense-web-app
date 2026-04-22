from utils.cropsense_all import *
from utils.plot_utils import *
from utils.indices import INDEX_META, INDEX_MAPS

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
# DYNAMIC ANALYSIS (RE-RUN OK)
# -----------------------------
def compute_analysis(hyperspectral_data, healthy, sick):

    indices_values = {
        "NDVI": calculate_mean_ndvi(hyperspectral_data),
        "GNDVI": calculate_mean_gndvi(hyperspectral_data),
        "RVI": calculate_mean_rvi(hyperspectral_data),
        "WI": calculate_mean_wi(hyperspectral_data),
        "NDWI": calculate_mean_ndwi(hyperspectral_data),
        "SIPI": calculate_mean_sipi(hyperspectral_data),
        "PRI": calculate_mean_pri(hyperspectral_data),
        "ARI": calculate_mean_ari(hyperspectral_data),
        "CARI": calculate_mean_cari(hyperspectral_data),
    }

    indices = [
        {
            "name": key,
            "label": INDEX_META.get(key, key),
            "value": float(value)
        }
        for key, value in indices_values.items()
    ]

    comparison = format_comparison_for_frontend(
        compare_plants_indices_json(
            [healthy, sick],
            ["Control Plant", "Infected Plant"]
        )
    )

    maps_b64 = {}

    for idx in INDEX_MAPS:
        index_map, cmap = compute_index_map(hyperspectral_data, idx)

        if index_map is None:
            continue

        fig_map = get_index_map_figure(index_map, cmap, idx)
        maps_b64[idx] = fig_to_base64(fig_map)

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
    analysis
):

    # 1. CACHE PIPELINE
    if sample_id not in PIPELINE_CACHE:
        PIPELINE_CACHE[sample_id] = run_full_pipeline(sample_id)

    base = PIPELINE_CACHE[sample_id]

    # 2. DYNAMIC ANALYSIS ONLY
    analysis_data = compute_analysis(
        base["hyperspectral_data"],
        base["healthy"],
        base["sick"]
    )

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

        # UI STATE
        "method": method,
        "thresholdWhite": thresholdWhite,
        "thresholdBlack": thresholdBlack,
        "thresholdNDVI": thresholdNDVI,
        "wavelength": wavelength,
        "analysis": analysis
    }