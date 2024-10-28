# comp_astro_24

# Daneel: Exoplanet Detection and Characterization

**Daneel** è uno strumento progettato per rilevare e caratterizzare esopianeti. Questo software consente di generare curve di luce per il transito di un esopianeta utilizzando parametri forniti in un file `YAML`.

## Requisiti di Sistema

- **Python >= 3.10**
- Dipendenze principali:
  - `numpy`
  - `matplotlib`
  - `PyYAML`
  - `batman-package`
  - `scikit-learn` (opzionale, se necessario per altri moduli)

## Installazione

Assicurati di avere il file `pyproject.toml` nella directory principale del progetto, quindi installa il pacchetto in modalità sviluppo:

```bash
pip install -e .
```

Questo comando rende il comando `daneel` eseguibile da qualsiasi directory.

## Uso del Comando `daneel`

### Generare la Curva di Luce del Transito

Per generare la curva di luce del transito, utilizza il comando `daneel` con i seguenti argomenti:

```bash
daneel -i path_to/parameters.yaml -t
```

- `-i`, `--input`: Specifica il percorso al file `parameters.yaml`, che contiene i parametri per l'esopianeta.
- `-t`, `--transit`: Opzione per generare e visualizzare la curva di luce del transito.

### Esempio di Esecuzione

Dalla directory in cui si trova il file `parameters.yaml`, esegui:

```bash
daneel -i parameters.yaml -t
```

## Struttura di `parameters.yaml`

Il file `parameters.yaml` deve essere strutturato con le seguenti chiavi per definire i parametri del transito. Se una chiave è assente, verrà utilizzato il valore di default indicato a fianco.

```yaml
name_of_the_exoplanet: "K2-287_b"        # Nome dell'esopianeta (Default: "!!Name not found!!")
a: 0.1206                                # Semi-asse maggiore in unità astronomiche (AU) (Default: 1)
star_radius: 1.07                        # Raggio della stella in raggi solari (R_sun) (Default: 1)
planet_radius: 0.833                     # Raggio del pianeta in raggi di Giove (R_jupiter) (Default: 1)
inclination: 88.13                       # Inclinazione orbitale in gradi (Default: 0)
eccentricity: 0.478                      # Eccentricità orbitale (Default: 0)
omega: 10.1                              # Longitudine del periasse in gradi (Default: 0)
period: 14.893291                        # Periodo orbitale in giorni (Default: 10)
t0: 0.0                                  # Tempo della congiunzione inferiore (inizio del transito) (Default: 0)
transit_duration: 0.3                    # Durata del transito in giorni (Default: 0.3)
u1: 0.4237666666666667                   # Coefficiente di oscuramento al bordo 1 (Default: 0)
u2: 0.21503333333333335                  # Coefficiente di oscuramento al bordo 2 (Default: 0)
```

## Dettagli dei Parametri

- **name_of_the_exoplanet** (str, default: "!!Name not found!!"): Nome dell'esopianeta per il titolo del grafico.
- **a** (float, default: 1): Distanza tra l'esopianeta e la stella in unità astronomiche.
- **star_radius** (float, default: 1): Raggio della stella in unità di raggio solare.
- **planet_radius** (float, default: 1): Raggio dell'esopianeta in unità di raggio gioviano.
- **inclination** (float, default: 0): Inclinazione dell'orbita dell'esopianeta rispetto al piano della vista.
- **eccentricity** (float, default: 0): Eccentricità dell'orbita.
- **omega** (float, default: 0): Longitudine del periasse.
- **period** (float, default: 10): Periodo orbitale dell'esopianeta.
- **t0** (float, default: 0): Tempo dell'inizio del transito (congiunzione inferiore).
- **transit_duration** (float, default: 0.3): Durata del transito in giorni.
- **u1**, **u2** (float, default: 0): Coefficienti di oscuramento al bordo della stella.

## Esempio di Output

Il comando genererà un grafico della curva di luce del transito e lo salverà come `K2-287_b_assignment1_taskF.png` nella directory corrente.