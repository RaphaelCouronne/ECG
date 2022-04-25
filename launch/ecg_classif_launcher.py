import os
from core.ecg_classif_nn import launch_ecg_classif



if __name__ == "__main__":

    # Paths
    data_path = "../../ECG_Data"
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    output_path = os.path.abspath(os.path.join(this_dir_path, "../trained_models/ECG_classif"))
    print(f"Output Path : {output_path}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Parameters
    max_samples_train = 2000
    max_samples_test = 100
    n_epochs = 3
    model_name = "ecg-classif"
    batch_size = 8
    learning_rate = 0.01

    # Launch
    launch_ecg_classif(data_path=data_path, output_path=output_path,
                       model_name=model_name,
                       max_samples_train=max_samples_train, max_samples_test=max_samples_test,
                       n_epochs=n_epochs, learning_rate=learning_rate, batch_size=batch_size)

