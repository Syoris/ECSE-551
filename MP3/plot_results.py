import utils
import matplotlib.pyplot as plt
from pathlib import Path


def lenet5_lr_plots():
    print('\n----- Learning rate -----')
    runs = ['MP3-91', 'MP3-86', 'MP3-87', 'MP3-88']
    legends = ['0.01', '0.001', '0.0001', '0.00001']

    data_dict = {}
    for run in runs:
        data_dict[run] = utils.get_run_data(run)

    plt.figure()
    for run in runs:
        val_loss_df = data_dict[run]['val_loss']
        plt.plot(val_loss_df['step'], val_loss_df['value'])

    plt.legend(legends, loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.grid(True)
    plt.title("Impact of learning rate on validation loss for LeNet5")
    plt.savefig(Path('MP3') / 'figures' / 'Lenet5_lr.png')
    plt.show(block=False)


def lenet5_batch_size_plots():
    print('\n----- Batch sizes -----')
    runs = ['MP3-93', 'MP3-94', 'MP3-95', 'MP3-96', 'MP3-97']
    legends = ['32', '64', '128', '256', '512']

    data_dict = {}
    for run in runs:
        data_dict[run] = utils.get_run_data(run)

    plt.figure()
    for run in runs:
        train_loss_df = data_dict[run]['train_loss']
        plt.plot(train_loss_df['step'], train_loss_df['value'].rolling(window=7).mean())

    plt.legend(legends, loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.grid(True)
    plt.title("Impact of batch size on averaged training loss for LeNet5")
    plt.savefig(Path('MP3') / 'figures' / 'Lenet5_batch.png')
    plt.show(block=False)


def lenet5_act_fn_plots():
    print('\n----- Act FN -----')
    runs = ['MP3-98', 'MP3-99', 'MP3-100', 'MP3-101']
    legends = ["ReLu", "Tanh", "LeakyReLU", 'Sigmoid']

    data_dict = {}
    for run in runs:
        data_dict[run] = utils.get_run_data(run)

    plt.figure()
    for run in runs:
        val_loss_df = data_dict[run]['val_loss']
        plt.plot(val_loss_df['step'], val_loss_df['value'])

    plt.legend(legends, loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.grid(True)
    plt.title("Impact of activation function on validation loss for LeNet5")
    plt.savefig(Path('MP3') / 'figures' / 'Lenet5_act_fn.png')
    plt.show(block=False)


def lenet5_loss_plots():
    print('\n----- Loss -----')
    runs = ['MP3-102', 'MP3-103']
    legends = ["Cross-entropy", "Negative-log likelihood"]

    data_dict = {}
    for run in runs:
        data_dict[run] = utils.get_run_data(run)

    plt.figure()
    for run in runs:
        val_acc_df = data_dict[run]['val_acc']
        plt.plot(val_acc_df['step'], val_acc_df['value'])

    plt.legend(legends, loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.title("Impact of activation function on validation accuracy for LeNet5")
    plt.savefig(Path('MP3') / 'figures' / 'Lenet5_loss.png')
    plt.show(block=False)


def plot_best_model():
    print('\n----- Best model -----')
    runs = ['MP3-109']
    legends = ["MP3-109"]

    data_dict = {}
    for run in runs:
        data_dict[run] = utils.get_run_data(run)

    plt.figure()
    for run in runs:
        train_loss_df = data_dict[run]['train_loss']
        val_loss_df = data_dict[run]['val_loss']
        plt.plot(train_loss_df['step'], train_loss_df['value'].rolling(window=20).mean())
        plt.plot(val_loss_df['step'], val_loss_df['value'].rolling(window=2).mean())

    plt.legend(legends, loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.title("...")
    # plt.savefig(Path('MP3') / 'figures' / 'Lenet5_loss.png')
    plt.show(block=False)


if __name__ == '__main__':
    print('############# Generating plots for results #############')
    # --- LeNet5 Hyperparmeters study ---
    # # 1. Learning rate
    # lenet5_lr_plots()

    # # 2. Batch size
    # lenet5_batch_size_plots()

    # # 3. Act Function
    # lenet5_act_fn_plots()

    # # 4. Loss Fn
    # lenet5_loss_plots()

    # --- Best model ---
    plot_best_model()
    ...
