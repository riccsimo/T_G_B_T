# Daneel: Exoplanet Detection and Characterization

**Daneel** is a tool designed to detect and characterize exoplanets. This software generates light curves for the transit of an exoplanet using parameters provided in a `YAML` file, and it also includes detection algorithms for exoplanet classification using SVM and Neural Network models.

## System Requirements

- **Python >= 3.10**
- Main dependencies:
  - `numpy`
  - `matplotlib`
  - `PyYAML`
  - `batman-package`
  - `scikit-learn`
  - `tensorflow` (for Neural Network detection)

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

## Additional Example Commands

### Running SVM Detection

From the directory containing `parameters.yaml`, run:

```bash
daneel -i parameters.yaml -d svm
```

### Running Neural Network Detection

To run neural network detection and save the trained weights:

```bash
daneel -i parameters.yaml -d nn
```

## Structure of `parameters.yaml`

The `parameters.yaml` file should be structured with specific keys to define the transit, SVM, or NN parameters. If a key is absent, a default value (if applicable) will be used.

### For Transit Parameters:

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

### For SVM Parameters:

```yaml
path_to_train_dataset: 'path/to/train_data.csv'
path_to_dev_dataset: 'path/to/dev_data.csv'
kernel: 'linear'              # Options: 'linear', 'rbf', 'poly', 'linear_svc'
degree: 3                     # Only used if kernel is 'poly'
```

### For Neural Network Parameters:

```yaml
path_to_train_dataset: 'path/to/train_data.csv'
path_to_dev_dataset: 'path/to/dev_data.csv'
complex_model: True                    # Use a more complex NN model if set to True
load_model: False                      # Load existing weights if True
render_plot: True                      # Display training accuracy and loss plots
save_weights: True                     # Save model weights after training
weights_path: False    # Path to save/load model weights
```

## Output

- **Transit Light Curve**: Generates a graph of the transit light curve, saved as `<exoplanet_name>_transit.png`.
- **SVM and NN Metrics**: Both detection algorithms print metrics including accuracy, precision, recall, and a confusion matrix, directly to the console. 
- **Model accuracy and loss**: NN detection give as output also plots of acucuracy and loss history of the model

---

