# Importa i moduli necessari
import numpy as np
import matplotlib.pyplot as plt
import batman

def transit(params):
    # Extracting parameters
    name_of_the_exoplanet = str(params.get('name_of_the_exoplanet','!!Name not found!!'))
    a = params.get('a', 1)  # Semi-asse maggiore (AU)
    star_radius = params.get('star_radius',1)  # Raggio della stella (in raggi solari)
    planet_radius = params.get('planet_radius',1)  # Raggio del pianeta (in raggi di Giove)
    inclination = params.get('inclination',0)  # Inclinazione orbitale (gradi)
    eccentricity = params.get('eccentricity', 0)  # Eccentricità
    omega = params.get('omega', 0)  # Longitudine del periasse (gradi)
    period = params.get('period', 10)  # Periodo orbitale (giorni)
    t0 = params.get('t0', 0)  # Tempo della congiunzione inferiore
    transit_duration = params.get('transit_duration', 0.3)  # Durata del transito in giorni
    u1 = params.get('u1', 0)  # Coefficiente di oscuramento al bordo 1
    u2 = params.get('u2', 0)  # Coefficiente di oscuramento al bordo 2

    # Costants
    au2km = 1.496e8  # Conversione da AU a km
    jupiter_radius = 71492  # Raggio di Giove in km
    sun_radius = 695700  # Raggio del Sole in km

    # Parameters input for Batman model
    parametri = batman.TransitParams()
    parametri.t0 = t0
    parametri.per = period
    parametri.rp = planet_radius * jupiter_radius / (star_radius * sun_radius)
    parametri.a = a * au2km / (star_radius * sun_radius)
    parametri.inc = inclination
    parametri.ecc = eccentricity
    parametri.w = omega
    parametri.u = [u1, u2]  # Coefficienti di oscuramento al bordo
    parametri.limb_dark = "quadratic"  # Modello di oscuramento al bordo

    # time array for the transit
    t = np.linspace(t0 -transit_duration / 2 - 0.1, t0 + transit_duration / 2 + 0.1, 100)

    # Light curve using batman
    m = batman.TransitModel(parametri, t)
    flux = m.light_curve(parametri)

    # Plot and saving immage
    plt.plot(t, flux)
    plt.xlabel("Time from central transit")
    plt.ylabel("Relative flux")
    plt.title(f"Transit Light Curve of {name_of_the_exoplanet}")
    plt.savefig(name_of_the_exoplanet+'_assignment1_taskF.png', format='png', dpi=300)
    plt.show()