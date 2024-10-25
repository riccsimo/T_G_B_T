#Importo i moduli
import numpy as np 
import matplotlib.pyplot as plt
import batman 


#Definisco i parametri del pianeta K2-287 b dal sito: https://exoplanet.eu/catalog/k2_287_b--7012/
a = 0.1206         #AU
au2km = 1.496e8       #km
star_radius = 1.07    #Rsun
planet_radius = 0.833  #Rjupiter
jupiter_radius = 71492 #km
sun_radius = 695700 #km


#Limb darkening coefficients da: https://exoctk.stsci.edu/limb_darkening
try:
    LD_coef = np.genfromtxt('ExoCTK_results.txt',skip_header=2)
    u1 = np.average(LD_coef[:,8])
    u2 = np.average(LD_coef[:,10])
except:
    #Calcolati a partire dal file ExoCTK_results.txt
    u1 = 0.4237666666666667
    u2 = 0.21503333333333335


#Definisco l'oggetto parametri per batman
parametri = batman.TransitParams()
parametri.t0 = 0.                       #time of inferior conjunction
parametri.per = 14.893291                     #orbital period
parametri.rp = planet_radius * jupiter_radius /  (star_radius * sun_radius)                     #planet radius (in units of stellar radii)
parametri.a = a * au2km / (star_radius * sun_radius)                       #semi-major axis (in units of stellar radii)
parametri.inc = 88.13       #not known             #orbital inclination (in degrees)
parametri.ecc = 0.478                      #eccentricity
parametri.w = 10.1                      #longitude of periastron (in degrees)
parametri.u = [u1, u2]                #limb darkening coefficients [u1, u2]
parametri.limb_dark = "quadratic"       #limb darkening model


#Array di tempi per il transito
t = np.linspace(-0.15, 0.15, 100)

#Modello batman
m = batman.TransitModel(parametri, t)    #initializes model
flux = m.light_curve(parametri)          #calculates light curve


#Plot e salvataggio immagine
plt.plot(t, flux)
plt.xlabel("Time from central transit")
plt.ylabel("Relative flux")
plt.savefig('K2-287_b_assignment1_taskF.png', format='png', dpi=300)
plt.show()