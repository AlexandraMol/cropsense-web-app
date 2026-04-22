from utils.cropsense_all import  *
from utils.plot_utils import *
from utils.indices import INDEX_META, INDEX_MAPS

def run_pipeline_service(
        sample_id,
        method,
        thresholdWhite,
        thresholdBlack,
        thresholdNDVI,
        wavelength,
        analysis
):
    # 1. Call pipeline
    run_pipeline()
    downloaded_paths = export_file_from_db("./generated",plant_id=sample_id, sensor_type="hyperspectral")

    # keep only .hdr files
    hdr_files = [f for f in downloaded_paths if f.endswith('.hdr')]

    # classify
    var_normal = next(f for f in hdr_files if f.endswith('.hdr') and 'DARKREF' not in f and 'WHITEREF' not in f)
    var_dark = next(f for f in hdr_files if 'DARKREF' in f and f.endswith('.hdr'))
    var_white = next(f for f in hdr_files if 'WHITEREF' in f and f.endswith('.hdr'))

    hyperspectral_data = prepare_hyperspectral_data(var_normal, var_dark, var_white, 'purple', 0.07, 0.72, 0.01)
    sick_plant, healthy_plant = separate_data(hyperspectral_data, cut_line=800)


    fig = show_hyperspectral_image(hyperspectral_data)
    healthy_plant_fig =  show_hyperspectral_image(healthy_plant)
    sick_plant_fig = show_hyperspectral_image(sick_plant)
    healthy_plant_graph = get_hyperspectral_graph(healthy_plant)
    sick_plant_graph = get_hyperspectral_graph(sick_plant)
    compare_graph = get_multiple_hyperspectral_graphs([healthy_plant, sick_plant], ["Control Plant", "Inoculated Plant"])

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

    raw = compare_plants_indices_json([healthy_plant, sick_plant], ["Control Plant", "Inoculated Plant"])

    comparison_table = format_comparison_for_frontend(raw)

    maps_b64 = {}

    for idx in INDEX_MAPS:
        index_map, cmap = compute_index_map(hyperspectral_data, idx)

        if index_map is None:
            continue

        fig_map = get_index_map_figure(index_map, cmap, idx)

        maps_b64[idx] = fig_to_base64(fig_map) if fig_map else None
 
    return {
        "sample_id": sample_id,
        "indices": indices,
        "comparison": comparison_table,

        # images from pipeline (BASE64 now)
        "fig": fig_to_base64(fig),
        "healthy_plant_fig": fig_to_base64(healthy_plant_fig),
        "sick_plant_fig": fig_to_base64(sick_plant_fig),
        "healthy_plant_graph": fig_to_base64(healthy_plant_graph),
        "sick_plant_graph": fig_to_base64(sick_plant_graph),
        "compare_graph": fig_to_base64(compare_graph),

        "index_maps": maps_b64,

        # keep UI state
        "method": method,
        "thresholdWhite": thresholdWhite,
        "thresholdBlack": thresholdBlack,
        "thresholdNDVI": thresholdNDVI,
        "wavelength": wavelength,
        "analysis": analysis
    }
