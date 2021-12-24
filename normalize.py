import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dset


def main():

    train_data1 = dset.ImageFolder(
        './data/amazon/', transform=transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ]))
    train_data2 = dset.ImageFolder(
        './data/dslr/', transform=transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ]))
    train_data3 = dset.ImageFolder(
        './data/webcam/', transform=transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ]))

    train_dataloader1 = DataLoader(dataset=train_data1, batch_size=64)
    train_dataloader2 = DataLoader(dataset=train_data2, batch_size=64)
    train_dataloader3 = DataLoader(dataset=train_data3, batch_size=64)

    print(len(train_dataloader1))
    print(len(train_dataloader2))
    print(len(train_dataloader3))

    mean1, std1 = get_mean_and_std(train_dataloader1)
    print("amazon: ", mean1, std1)
    mean2, std2 = get_mean_and_std(train_dataloader2)
    print("dslr: ", mean2, std2)
    mean3, std3 = get_mean_and_std(train_dataloader3)
    print("webcam: ", mean3, std3)


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


if __name__ == '__main__':
    main()
