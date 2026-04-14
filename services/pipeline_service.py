import matplotlib
matplotlib.use('Agg')

import os
import matplotlib.pyplot as plt
import numpy as np

def run_pipeline(method, threshold, wavelength, analysis):

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

    return {
        "before_image": "assets/images/before.jpg",
        "after_image": "assets/images/after.jpg",
        "graph": "generated/output.png",
        "method": method,
        "threshold": threshold,
        "wavelength": wavelength
    }