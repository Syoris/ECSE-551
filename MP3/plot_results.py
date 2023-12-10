import utils
import matplotlib.pyplot as plt


def lenet5_lr_plots():
    print('\n----- Learning rate -----')
    runs = ['MP3-85', 'MP3-86', 'MP3-87', 'MP3-88']
    legends = ['0.01', '0.001', '0.0001', '0.00001']

    data_dict = {}
    for run in runs:
        data_dict[run] = utils.get_run_data(run)

    plt.figure()
    for run in runs:
        train_loss_df = data_dict[run]['val_loss']
        plt.plot(train_loss_df['step'], train_loss_df['value'])

    plt.legend(legends, loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Impact of learning rate on validation loss for LeNet5")
    plt.show(block=False)


if __name__ == '__main__':
    print('############# Generating plots for results #############')
    # --- LeNet5 Hyperparmeters study ---
    # 1. Learning rate
    lenet5_lr_plots()

    ...
