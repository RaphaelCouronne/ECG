import kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files('shayanfazeli/heartbeat',
                                  path='../Data/ECG_Heartbeat_Categorization_Dataset',
                                  unzip=True)
