import numpy as np
import scipy.io
import os


def load_bp_data(bp_path, num_files_to_load=1):
    """
    Load the Cuff-Less Blood Pressure Estimation dataset from kaggle files
    :param bp_path:
    :param num_files_to_load:
    :return: Numpy array of size n_samples x 3 (BP, PPG, ECG)
    """

    sample_file_list = []
    for i in range(1, 1+num_files_to_load):
        sample_file_list.append(scipy.io.loadmat(os.path.join(bp_path, "part_{}.mat".format(i)))["p"][0])
    sample_file = np.concatenate(sample_file_list, axis=0)


    print("Number of recordings : ", sample_file.shape)
    print("Number of samples for first recording : {} for duration {}s".format(sample_file[0].shape[1],
                                                                               sample_file[0].shape[1] / 125))
    print("Number of samples for random recording : {} for duration {}s".format(sample_file[10].shape[1],
                                                                                sample_file[10].shape[1] / 125))
    print("Number of samples for random recording : {} for duration {}s".format(sample_file[20].shape[1],
                                                                                sample_file[20].shape[1] / 125))
    # Reshape Data
    data = np.concatenate(sample_file, axis=1)
    data = np.stack([data[1], data[0], data[2]], axis=0)
    data = data.T

    return data

def assert_gpu():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    from keras import backend as K
    K.tensorflow_backend._get_available_gpus()


def download_kaggle(dataset_name, path):
    #'shayanfazeli/heartbeat'
    # '../Data/ECG_Heartbeat_Categorization_Dataset',
    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset_name,
                                      path=path,
                                      unzip=True)
