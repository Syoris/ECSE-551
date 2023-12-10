import utils


def lenet5_lr_plots():
    runs = ['MP3-85', 'MP3-86', 'MP3-87', 'MP3-88']
    data_dict = {}
    for run in runs:
        data_dict[run] = utils.get_run_data(run)

    ...


if __name__ == '__main__':
    # --- LeNet5 Hyperparmeters study ---
    # 1. Learning rate
    lenet5_lr_plots()

    ...
