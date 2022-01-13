# Dictionary of model p for facebook prophet
prophet_model_params = {'changepoint_prior_scale': 0.05,
                        'changepoints': None,
                        'changepoint_range': 0.8,
                        'interval_width': 0.95,
                        'seasonality_mode': 'multiplicative',
                        'seasonality_prior_scale': 10,
                        'holidays_prior_scale': 10}

# Dictionary of model parameters for pmdarima
pmdarima_model_params = {'start_p': 1,
                         'start_q': 1,
                         'max_p': 3,
                         'max_q': 3,
                         'm': 12,
                         'start_P': 0,
                         'seasonal': True,
                         'd': 1,
                         'D': 1,
                         'trace': True,
                         'error_action': 'ignore',
                         'suppress_warnings': True,
                         'stepwise': True}

# Dictionary to specify the start and end dates of the training and prediction windows of the time series data
start_end_dates = {'train': {'start': '1949-01-01',
                             'end': '1958-11-01'},
                   'test': {'start': '1958-12-01',
                            'end': '1960-12-01'},
                   'pred': {'start': '1961-01-01',
                            'end': '1961-12-01'}}