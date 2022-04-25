from core.bp_regr_nn import launch_bp_regression
import os

if __name__ == "__main__":

    # Path
    data_path = "../../ECG_Data"
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    output_path = os.path.abspath(os.path.join(this_dir_path, "../trained_models/SBP_regr"))
    print(f"Output Path : {output_path}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Parameters
    num_files_to_load = 1
    max_samples_train = 1000
    max_samples_test = 200
    freq = 125
    n_epochs = 2
    batch_size = 16
    learning_rate = 0.001

    # Launch
    launch_bp_regression(data_path=data_path, output_path=output_path, model_name="sbp_regression",
                         num_files_to_load=num_files_to_load,
                         freq=freq, max_samples_train=max_samples_train, max_samples_test=max_samples_test,
                         n_epochs=n_epochs, learning_rate=learning_rate, batch_size=batch_size)