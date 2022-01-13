import logging
import argparse
import pandas as pd

import warnings
from Forecaster import Forecaster
from params import *
warnings.filterwarnings("ignore")

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="A simple script to test logging and argparse")

    # passing model arguments to the class
    # model_args = {'model_params': pmdarima_model_params,
    #               'time_windows': start_end_dates,
    #               'model_name': 'pmdarima',
    #               'model_type': 'pmdarima'}

    model_args = {'model_params': prophet_model_params,
                  'time_windows': start_end_dates,
                  'model_name': 'FBProphet',
                  'model_type': 'prophet'}

    df = pd.read_csv('data/AirPassengers.csv')
    df.columns = ['ds', 'y']

    # Create an instance of the Forecaster class
    f = Forecaster(data=df, **model_args)
    f.preprocess_data()
    f.train_test_split()
    my_forecast = f.make_forecast()
    f.plot_forecast()

    logging.info("Done-zo!!")
