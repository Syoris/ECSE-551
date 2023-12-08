from time import time
import multiprocessing as mp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_loader import MyDataset

if __name__ == '__main__':
    data_folder = f"MP3/data"

    train_file = "Train.pkl"
    train_label_file = "Train_labels.csv"
    test_file = "Test.pkl"
    val_size = 0.15

    # Datasets
    ds = MyDataset(
        train_file,
        label_file_name=train_label_file,
        transform=None,
        folder_path=data_folder,
    )

    for num_workers in range(0, mp.cpu_count(), 2):  
        train_loader = DataLoader(ds,shuffle=True,num_workers=num_workers,batch_size=64,pin_memory=True)

        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))