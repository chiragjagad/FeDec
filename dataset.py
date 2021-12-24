from torchvision import datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


def get_office_dataset(dataset_name, path):
    root_dir = path.format(dataset_name)
    _datasets_ = ['amazon', 'dslr', 'webcam']
    if dataset_name not in _datasets_:
        raise ValueError("must introduce one of the datasets in office")
    mean_std = {"amazon": {"mean": [0.7923, 0.7862, 0.7841],
                           "std": [0.3149, 0.3174, 0.3193]
                           }, "dslr": {
        "mean": [0.4708, 0.4486, 0.4063],
        "std": [0.2039, 0.1920, 0.1996]
    }, "webcam": {
        "mean": [0.6119, 0.6187, 0.6173],
        "std": [0.2506, 0.2555, 0.2577]
    }
    }
    # data_transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(256), transforms.ToTensor(
    # ), transforms.Normalize(mean=mean_std[dataset_name]["mean"], std=mean_std[dataset_name]["std"])])
    data_transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(256), transforms.ToTensor(
    )])
    dataset = datasets.ImageFolder(root=root_dir, transform=data_transforms)
    return dataset


def get_dataloader(dataset, batch_size, num_workers, train_ratio=0.70):
    def get_subset(indices, start, end):
        return indices[start:start+end]
    Train_Ratio, Validation_Ratio = train_ratio, 1-train_ratio
    train_set_size = int(len(dataset)*Train_Ratio)
    validation_set_size = int(len(dataset)*Validation_Ratio)

    # Generate Random Indices from Train and Val Sets
    indices = torch.randperm(len(dataset))
    train_indices = get_subset(indices, 0, train_set_size)
    validation_indices = get_subset(
        indices, train_set_size, validation_set_size)

    # Create sampler objects
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(validation_indices)
    # Create data Loader
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler, num_workers=num_workers, drop_last=True)
    return train_loader, val_loader
