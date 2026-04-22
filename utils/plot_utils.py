import base64
from io import BytesIO
import matplotlib.pyplot as plt

def fig_to_base64(fig):
    """Convert a matplotlib figure to a base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    buf.close()
    plt.close(fig)

    return img_base64