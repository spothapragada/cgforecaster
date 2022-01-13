# A class to make time series predictions using PMDArima and Facebook Prophet
import logging

import matplotlib.pyplot as plt
import pandas as pd
from pmdarima import auto_arima

from params import *

from prophet import Prophet
from prophet.plot import add_changepoints_to_plot

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='test.log',
                    filemode='w'
                    )


class Forecaster:
    def __init__(self, data, model_params, time_windows, model_name, model_type):
        self.data = data
        self.model_type = model_type
        self.model_name = model_name
        self.model_params = model_params
        self.time_windows = time_windows
        self.forecast = None
        self.model = None
        self.df_test = None
        self.df_train = None

    def make_forecast(self):
        if self.model_type == 'prophet':
            logging.info('The Model Type is Prophet')
            self.forecast = self.prophet_forecast()
        elif self.model_type == 'pmdarima':
            self.forecast = self.pmdarima_forecast()
        else:
            logging.info('Model type not recognized')

        return self.forecast

    def preprocess_data(self):
        """
        Preprocess the data to ensure there is a date column and y column
        :return: Dataframe with a date column and y column
        """
        # Rename the columns to ds and y
        self.data.columns = ['ds', 'y']
        # Convert the ds column to date
        self.data['ds'] = pd.to_datetime(self.data['ds'], format='%Y-%m-%d')
        # Convert the ds column to index
        self.data.set_index('ds', inplace=True)
        # Create a new column for the y values
        self.data['y'] = self.data['y'].astype(float)

    def train_test_split(self):
        """
        Split the data into training and test using the start_end_dates
        :return: Dataframes df_train, df_test
        """
        # Read in the training and test dates
        train_start_date = self.time_windows['train']['start']
        train_end_date = self.time_windows['train']['end']
        test_start_date = self.time_windows['test']['start']
        test_end_date = self.time_windows['test']['end']

        # Create the training dataset from the training dates
        self.df_train = self.data[train_start_date:train_end_date]
        # Create the test dataset from the test dates
        self.df_test = self.data[test_start_date:test_end_date]

        return self.df_train, self.df_test

    def prophet_forecast(self):
        # Import the required packages

        # Create a dataframe from the data
        dframe = self.data

        logging.info('Making forecast using Facebook Prophet')
        # Create the model using the model_params


        # m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
        #             holidays=None, changepoint_prior_scale=0.05, changepoints=None,
        #             changepoint_range=0.8, interval_width=0.95, seasonality_mode='multiplicative',
        #             seasonality_prior_scale=10, holidays_prior_scale=10)

        m = Prophet(**self.model_params)
        logging.info('Model parameters: {}'.format(self.model_params))

        # Fit the model
        m.fit(dframe)

        # Make the forecast
        future = m.make_future_dataframe(periods=len(self.data))
        forecast = m.predict(future)

        # Plot the forecast
        fig = m.plot(forecast)
        add_changepoints_to_plot(fig.gca(), m, forecast)
        plt.show()

        # Return the forecast
        return forecast

    def pmdarima_forecast(self):
        # Import the required packages

        # Read in the prediction window from the model_params
        pred_start_date = self.time_windows['pred']['start']
        pred_end_date = self.time_windows['pred']['end']

        pred_periods = len(pd.date_range(pred_start_date, pred_end_date, freq='M'))
        logging.info('Making forecast using PMDArima')

        # Create the model
        pmdmodel = auto_arima(self.df_train, **self.model_params)
        # Fit the model (one the combined training and test dataset)
        pmdmodel.fit(self.df_train.append(self.df_test))

        # Collect the model
        self.model = pmdmodel

        # TODO - Write out the model to a file somewhere
        # pmdmodel.save('pmdmodel.pkl')

        # Make the forecast
        # Create a datetime index for the forecast
        ds = pd.date_range(start=pred_start_date, periods=pred_periods, freq='M')
        preds, intvs = pmdmodel.predict(n_periods=pred_periods, return_conf_int=True)
        # Create a dataframe with the predictions
        forecast = pd.DataFrame(preds, index=ds, columns=['yhat'])
        forecast['ci_lower'] = intvs[:, 0]
        forecast['ci_upper'] = intvs[:, 1]
        self.forecast = forecast.copy()

        return self.forecast

    # Method to plot the forecast
    def plot_forecast(self):
        # Plot the forecast and the actual data in different colors
        plt.plot(self.df_train.append(self.df_test)['y'], label='Training Data')
        plt.plot(self.df_test['y'], label='Test Data')
        # Plot the forcast with confidence intervals
        plt.fill_between(self.forecast.index, self.forecast['ci_lower'], self.forecast['ci_upper'], alpha=0.2)
        plt.plot(self.forecast['yhat'], label='Forecast')
        plt.legend(loc='best')
        plt.show()
