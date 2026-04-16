import matplotlib
matplotlib.use('Agg')

import os
import matplotlib.pyplot as plt
import numpy as np
from utils.indices import indices

from utils.pipeline import *

def run_pipeline(method, thresholdWhite, thresholdBlack, thresholdNDVI, wavelength, analysis):

    # simulate data
    x = np.linspace(400, 1000, 10)
    y = np.random.rand(10)

    os.makedirs("static/generated", exist_ok=True)

    if analysis == "profile":
        plt.plot(x, y)
        plt.title("Spectral Profile")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")

    else:
        plt.hist(y)
        plt.title("Histogram")

    output_path = "static/generated/output.png"
    plt.savefig(output_path)
    plt.close()

    # mock data for indices
    results = []

    for index in indices:
        name = index["name"]

        # replace this with real computation
        value = compute_index(name, wavelength)

        results.append({
            "name": name,
            "label": index["label"],
            "value": value
        })

    return {
        "before_image": "assets/images/before.jpg",
        "after_image": "assets/images/after.jpg",
        "graph": "generated/output.png",
        "method": method,
        "thresholdWhite": thresholdWhite,
        "thresholdBlack": thresholdBlack,
        "thresholdNDVI": thresholdNDVI,
        "wavelength": wavelength,
        "indices": results
    }

def compute_index(name, wavelength):
    # mock logic for now
    import random
    return round(random.uniform(0, 1), 3)