import batman
import numpy as np
import matplotlib.pyplot as plt
from daneel.detection import TransitModel

# Definizione dei parametri
params = {
    'name_of_the_exoplanet': 'Wasp-39 b',
    'a': 0.0486,
    'star_radius': 0.895,
    'planet_radius': 1.27,
    'inclination': 87.83,
    'eccentricity': 0,
    'omega': 0,
    'period': 4.055,
    't0': 0,
    'transit_duration': 0.2,
    'u1': 0.42376,
    'u2': 0.21503,
    'save_image' : False,
    'LD_path' : "ExoCTK_results.txt"
}

# Creazione di un'istanza della classe
transit_model = TransitModel(params)

# Generazione e visualizzazione della curva di luce
transit_model.plot_light_curve()
