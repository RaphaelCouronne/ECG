import kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files('shayanfazeli/heartbeat',
                                  path='../ECG_Heartbeat',
                                  unzip=True)
