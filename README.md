# Daneel: Exoplanet Detection and Characterization

**Daneel** is a tool designed to detect and characterize exoplanets. This software generates light curves for the transit of an exoplanet using parameters provided in a `YAML` file.

## System Requirements

- **Python >= 3.10**
- Main dependencies:
  - `numpy`
  - `matplotlib`
  - `PyYAML`
  - `batman-package`
  - `scikit-learn` (optional, if needed for additional modules)

## Installation

Ensure you have the `pyproject.toml` file in the main project directory, then install the package in development mode:

```bash
pip install -e .
```

This command makes the `daneel` command executable from any directory.

## Using the Daneel Command

### Generating the Transit Light Curve

To generate the light curve for the exoplanet transit, use the `daneel` command with the following arguments:

```bash
daneel -i path_to/parameters.yaml -t
```

- `-i, --input`: Specifies the path to the `parameters.yaml` file containing the parameters for the exoplanet.
- `-t, --transit`: Option to generate and display the transit light curve.

### Example Execution

From the directory containing the `parameters.yaml` file, execute:

```bash
daneel -i parameters.yaml -t
```

## Structure of `parameters.yaml`

The `parameters.yaml` file should be structured with the following keys to define the transit parameters. If a key is absent, the default value shown alongside each parameter will be used.

```yaml
name_of_the_exoplanet: "K2-287_b"        # Name of the exoplanet (Default: "!!Name not found!!")
a: 0.1206                                # Semi-major axis in astronomical units (AU) (Default: 1)
star_radius: 1.07                        # Radius of the star in solar radii (R_sun) (Default: 1)
planet_radius: 0.833                     # Radius of the planet in Jupiter radii (R_jupiter) (Default: 1)
inclination: 88.13                       # Orbital inclination in degrees (Default: 0)
eccentricity: 0.478                      # Orbital eccentricity (Default: 0)
omega: 10.1                              # Argument of periastron in degrees (Default: 0)
period: 14.893291                        # Orbital period in days (Default: 10)
t0: 0.0                                  # Time of inferior conjunction (start of transit) (Default: 0)
transit_duration: 0.3                    # Duration of the transit in days (Default: 0.3)
u1: 0.4237666666666667                   # Limb-darkening coefficient 1 (Default: 0)
u2: 0.21503333333333335                  # Limb-darkening coefficient 2 (Default: 0)
```

## Parameter Details

- **`name_of_the_exoplanet`** *(str, default: "!!Name not found!!")*: Name of the exoplanet used for the graph title.
- **`a`** *(float, default: 1)*: Distance between the exoplanet and the star in astronomical units.
- **`star_radius`** *(float, default: 1)*: Radius of the star in solar radii.
- **`planet_radius`** *(float, default: 1)*: Radius of the exoplanet in Jupiter radii.
- **`inclination`** *(float, default: 0)*: Orbital inclination of the exoplanet with respect to the observer's line of sight.
- **`eccentricity`** *(float, default: 0)*: Orbital eccentricity.
- **`omega`** *(float, default: 0)*: Argument of periastron.
- **`period`** *(float, default: 10)*: Orbital period of the exoplanet.
- **`t0`** *(float, default: 0)*: Time of inferior conjunction (beginning of the transit).
- **`transit_duration`** *(float, default: 0.3)*: Duration of the transit in days.
- **`u1`, `u2`** *(float, default: 0)*: Limb-darkening coefficients for the star.

## Sources for Parameters

- Most parameters for known exoplanets can be found at [https://exoplanet.eu/catalog/](https://exoplanet.eu/catalog/).
- The limb-darkening coefficients `u1` and `u2` are calculated by averaging columns `c1` and `c2` respectively from the table obtained at [https://exoctk.stsci.edu/limb_darkening](https://exoctk.stsci.edu/limb_darkening).

## Example Output

The command generates a graph of the transit light curve and saves it as `K2-287_b_assignment1_taskF.png` in the current directory.

---

Feel free to use and modify this tool to analyze various exoplanets. Contributions are welcome to improve or extend functionality!

