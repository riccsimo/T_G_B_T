import datetime
import argparse
from daneel.parameters import Parameters
from daneel.detection import *
from daneel.atmosphere import *


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        required=True,
        help="Input par file to pass",
    )

    parser.add_argument(
        "-d",
        "--detect",
        dest="detect",
        type=str,
        required=False,
        help="Initialise detection algorithms for Exoplanets (e.g., -d svm)",
        choices=['svm', 'nn']
    )

    parser.add_argument(
        "-a",
        "--atmosphere",
        dest="atmosphere",
        required=False,
        type=str,
        help="Atmospheric Characterization options: model or retrieve",
        choices=['model', 'retrieve'],
    )

    #flag -t per il transito
    parser.add_argument(
        "-t",
        "--transit",
        dest="transit",
        required=False,
        help="Plot the transit light curve",
        action="store_true",
    )
    
    
    args = parser.parse_args()

    """Launch Daneel"""
    start = datetime.datetime.now()
    print(f"Daneel starts at {start}")

    input_pars = Parameters(args.input_file).params
    
    # Chiamata alla funzione di transito con i parametri estratti
    if args.transit:
        transit(input_pars)  # Passa il dizionario dei parametri a transit

    if args.detect:
        if args.detect == 'svm':
            svm_detector = SVMExoplanetDetector(input_pars)
            svm_detector.run()
            
        elif args.detect == 'nn':
            nn_detector = NNExoplanetDetector(input_pars)
            nn_detector.run()
        
    if args.atmosphere:
        if args.atmosphere == 'model':
            model=AtmosphereForwardModel(input_pars)
            model.construct_taurex_model()
            model.ForwardModel()
        elif args.atmosphere == 'retrieve':
            retrieve=AtmosphereForwardModel(input_pars)
            retrieve.construct_taurex_model()
            retrieve.Retrival()

    finish = datetime.datetime.now()
    print(f"Daneel finishes at {finish}")


if __name__ == "__main__":
    main()

