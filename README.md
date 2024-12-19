# Daneel: Exoplanet Detection and Characterization

**Daneel** is a tool designed to detect and characterize exoplanets. This software generates light curves for the transit of an exoplanet using parameters provided in a `YAML` file, and it also includes detection algorithms for exoplanet classification using SVM and Neural Network models. It includes also two option for atmospheric analysis (modelling and retrievals) based on TauREx 3 python library.

## System Requirements

- **Python >= 3.10**
- Main dependencies:
  - `numpy`
  - `matplotlib`
  - `PyYAML`
  - `batman-package`
  - `scikit-learn`
  - `tensorflow` (for Neural Network detection)
  - `taurex`

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

### Example `parameters.yaml` for Generating the Transit Light Curve

```yaml
name_of_the_exoplanet: "K2-287_b"        
a: 0.1206                                
star_radius: 1.07                        
planet_radius: 0.833                     
inclination: 88.13                       
eccentricity: 0.478                      
omega: 10.1                              
period: 14.893291                        
t0: 0.0                                  
transit_duration: 0.3                    
u1: 0.4237666666666667                   
u2: 0.21503333333333335                  
```

### Exoplanet Detection with SVM

Daneel provides an exoplanet detection feature using a Support Vector Machine (SVM) model. This model is used to classify potential exoplanet data based on the specified kernel and dataset parameters.

To use SVM for exoplanet detection:

```bash
daneel -i path_to/parameters.yaml -d svm
```

- `-i, --input`: Path to the `parameters.yaml` file with necessary dataset paths and SVM configuration.
- `-d, --detect`: Specify `svm` to use the SVM detection algorithm.

The SVM detection feature loads the datasets specified in the `parameters.yaml` file, preprocesses the data, trains an SVM classifier, and evaluates the model's performance on the development dataset. Results include metrics such as precision, recall, and a confusion matrix.

### Example `parameters.yaml` for SVM

```yaml
path_to_train_dataset: 'path/to/train_data.csv'
path_to_dev_dataset: 'path/to/dev_data.csv'
kernel: 'linear'              # Options: 'linear', 'rbf', 'poly', 'linear_svc'
degree: 3                     # Only used if kernel is 'poly'
```

### Exoplanet Detection with Neural Network (NN)

Daneel also includes an exoplanet detection feature using a Neural Network (NN) model. This model leverages a fully connected neural network to classify exoplanet data.

To use NN for exoplanet detection:

```bash
daneel -i path_to/parameters.yaml -d nn
```

- `-i, --input`: Path to the `parameters.yaml` file with necessary dataset paths and NN configuration.
- `-d, --detect`: Specify `nn` to use the Neural Network detection algorithm.

The NN detection feature loads the specified datasets, preprocesses the data, builds and trains a neural network model, and evaluates its performance. The NN detection algorithm also supports optional loading of pretrained weights and saves the model’s weights after training.

### Example `parameters.yaml` for NN

```yaml
path_to_train_dataset: 'path/to/train_data.csv'
path_to_dev_dataset: 'path/to/dev_data.csv'
complex_model: True                    # Use a more complex NN model if set to True
load_model: False                      # Load existing weights if True
render_plot: True                      # Display training accuracy and loss plots
save_weights: True                     # Save model weights after training
weights_path: False    # Path to save/load model weights
```

### Exoplanet Atmosphere analysis with TauREx 3

The AtmosphereForwardModel class uses the TauREx 3 framework to simulate and analyze exoplanetary spectra. It supports forward modeling (transmission, emission, direct imaging) and retrieval of atmospheric parameters from observed spectra.

```bash
daneel -i path_to/parameters.yaml -a model
```
```bash
daneel -i path_to/parameters.yaml -a retrieve
```

- `-i, --input`: Path to the `parameters.yaml` file with necessary dataset paths and NN configuration.
- `-a, --atmosphere`: Specify `model` to create an atmospheric model, `retrieve` for an atmospheric retrieval.

The first bash command is used to generate a syntetic spectrum. The second one is used to perform an atmospheric retrieval.
During the retrieval process, the tool includes interactive prompts to guide the user through key steps:

1. **Selecting Fitting Parameters**:
   - The program lists all available parameters for fitting.
   - Users are prompted to select which parameters to include by entering their indices (e.g., `1,3,5`).

2. **Specifying Parameter Boundaries**:
   - For each selected parameter, users define the fitting boundaries.
   - The program requests the boundaries in the format `lower,upper` (e.g., `1e-2,1.0`).
   - These boundaries ensure the retrieval process remains within meaningful value ranges.


These tools utilize a subset of features from the TauREx framework. For full access to TauREx capabilities, including its bash commands and comprehensive functionality, refer to the [official TauREx documentation](https://taurex3-public.readthedocs.io/).

### Example `parameters.yaml` for both atmospheric modelling and retrievals

```yaml
#Retrival (if used for modelling this section can be skipped)
path_to_observed_spectrum: ../../../data/atmosphere/quickstart.dat
num_live_points: 50
tol: 0.5
#Model
Global:
  xsec_path: ../../../data/atmosphere/xsecs
  cia_path: ../../../data/atmosphere/cia/hitran
  phoenix_path: ../../../data/atmosphere/phoenix/BT-Settl_M-0.0a+0.0
  output_file: taskF_spectrum.dat
  #log of the wavelenght (micrometers) between which compute the model
  log_lmbda0: -0.4
  log_lmbda1: 1.1

Atmospheric_model:
  type: isothermal  # It can be "isothermal" or "guillot"
  parameters:
    T: 1500.0 #K 
    #T_irr: 1500.0 #k if Guillot

Planet:
  planet_radius: 1. #Rj
  planet_mass: 1. #Mj

Star:
  type: blackbody # It can be "blackbody" or "phoenix"
  parameters:
    radius: 1. #Rsun
    temperature: 5700.0 #K

Chemistry:
  main_species:
    fill_gases: [H2,He]
    ratio: 0.172
  other_molecules:
    H2O:
      gas_type: constant
      mix_ratio: 1.2e-4  
    # CH4:
    #   gas_type: constant
    #   mix_ratio: 0.005495
    # CO2:
    #   gas_type: constant
    #   mix_ratio: 2.814e-06
    # CO:
    #   gas_type: constant
    #   mix_ratio: 7.73e-06

Pressure:
  atm_min_pressure: 1e-0  # bar
  atm_max_pressure: 1e6  # bar
  nlayers: 30

Model:
  type: transmission # It can be "trasmission" or "emission" o "direct_image"
  physical_processes:
      Absorption: True  # Enable absorption contribution
      CIA:
        enabled: True  # Enable CIA contribution
        pairs: 
          - H2-H2
          - H2-He
      Rayleigh: True  # Enable Rayleigh contribution

```

## Output

- **Transit Light Curve**: Generates a graph of the transit light curve, saved as `<exoplanet_name>_transit.png`.
- **SVM and NN**: Both detection algorithms print metrics including accuracy, precision, recall, and a confusion matrix, directly to the console. NN detection give as output also plots of acucuracy and loss history of the model.
- **Atmosphere - Model**: Generates a .dat file containing wavelengths (in micrometers), $(R_p/R_s)^2$, and the associated errors. Additionally, it produces a PNG of the simulated (artificial) spectrum.
- **Atmosphere - Retrieval**: Outputs a .dat file with wavelengths (in micrometers), $(R_p/R_s)^2$, and the associated errors. It also generates:
  1. A PNG showing the retrieved spectrum overlaid on the observed data.
  2. A PNG displaying the corner plot of the posterior distributions. 
---

