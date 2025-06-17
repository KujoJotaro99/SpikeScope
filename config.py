import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

MODEL_PATHS = {
    "exoplanet_snn": os.path.join(BASE_DIR, "weights", "exoplanet_snn.pth")
}