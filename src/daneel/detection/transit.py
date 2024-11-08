# Importa i moduli necessari
import numpy as np
import matplotlib.pyplot as plt
import batman

class TransitModel:

    LD_table = np.genfromtxt("ExoCTK_results.txt")

    def __init__(self, params):

        # Estrazione dei parametri
        self.name_of_the_exoplanet = params.get('name_of_the_exoplanet', ['!!Name not found!!'])
        self.a = params.get('a', [1])  # Semi-asse maggiore (AU)
        self.star_radius = params.get('star_radius', 1)  # Raggio della stella (in raggi solari)
        self.planet_radius = params.get('planet_radius', [1])  # Raggio del pianeta (in raggi di Giove)
        self.inclination = params.get('inclination', [0])  # Inclinazione orbitale (gradi)
        self.eccentricity = params.get('eccentricity', [0])  # Eccentricità
        self.omega = params.get('omega', [0])  # Longitudine del periasse (gradi)
        self.period = params.get('period', [10])  # Periodo orbitale (giorni)
        self.t0 = params.get('t0', 0)  # Tempo della congiunzione inferiore
        self.transit_duration = params.get('transit_duration', 0.3)  # Durata del transito in giorni


        # Parametri di plot
        self.save = params.get("save_image", False)

        self.LD_path = params.get("LD_path", False)

        if self.LD_path:
            self.LD_table = np.genfromtxt(self.LD_path, skip_header=2)
            self.u1 = np.average(self.LD_table[:, 8])
            self.u2 = np.average(self.LD_table[:, 10])
        else:
            self.u1 = params.get('u1', 0)  # Coefficiente di oscuramento al bordo 1
            self.u2 = params.get('u2', 0)  # Coefficiente di oscuramento al bordo 2

        # Costanti
        self.au2km = 1.496e8  # Conversione da AU a km
        self.jupiter_radius = 71492  # Raggio di Giove in km
        self.sun_radius = 695700  # Raggio del Sole in km

        t = np.linspace(
            self.t0 - np.max(self.transit_duration) / 2 ,
            self.t0 + np.max(self.transit_duration) / 2 ,
            100
        )

        flux = []

        for i in range(len(self.name_of_the_exoplanet)):

            # Parametri per il modello Batman
            self.parametri = batman.TransitParams()
            self.parametri.t0 = self.t0
            self.parametri.per = self.period[i]
            self.parametri.rp = self.planet_radius[i] * self.jupiter_radius / (self.star_radius * self.sun_radius)
            self.parametri.a = self.a[i] * self.au2km / (self.star_radius * self.sun_radius)
            self.parametri.inc = self.inclination[i]
            self.parametri.ecc = self.eccentricity[i]
            self.parametri.w = self.omega[i]
            self.parametri.u = [self.u1, self.u2]  # Coefficienti di oscuramento al bordo
            self.parametri.limb_dark = "quadratic"  # Modello di oscuramento al bordo


            # Curva di luce usando batman
            m = batman.TransitModel(self.parametri, t)
            flux.append(m.light_curve(self.parametri))


        # Plot e salvataggio dell'immagine
        for i in range(len(self.name_of_the_exoplanet)):
            plt.plot(t, flux[i], label=f"{self.name_of_the_exoplanet[i]}")

        plt.xlabel("Relative time [days]")
        plt.ylabel("Relative Flux")
        plt.title(f"Transit Light Curve of {self.name_of_the_exoplanet}")
        plt.legend()

        if self.save:
            plt.savefig("assignment2_taskA.png", format='png', dpi=300)
            #plt.savefig(f"{self.name_of_the_exoplanet}_assignment2_taskA.png", format='png', dpi=300)

        plt.show()


