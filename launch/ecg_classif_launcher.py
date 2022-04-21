import os
from core.ecg_classif_nn import launch_ecg_classif



if __name__ == "__main__":
    # Parameters
    data_path = "../../ECG_Data"
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    output_path = os.path.abspath(os.path.join(this_dir_path, "../trained_models"))
    print(f"Output Path : {output_path}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    max_samples_train = 1000
    max_samples_test = 200
    n_epochs = 5
    model_name = "ecg-classif"


    launch_ecg_classif(data_path=data_path, output_path=output_path,
                       model_name=model_name, n_epochs=n_epochs,
                       max_samples_train=max_samples_train, max_samples_test=max_samples_test)

