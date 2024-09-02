
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()
def create_dataloaders(train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int):
    train_data = ImageFolder(root= train_dir, transform= transform)
    test_data = ImageFolder(root= test_dir, transform= transform)

    train_data_loader = DataLoader(train_data, batch_size= batch_size, shuffle= True, num_workers= NUM_WORKERS)
    test_data_loader = DataLoader(test_data, batch_size= batch_size, num_workers= NUM_WORKERS)

    class_names = train_data.classes

    return train_data_loader, test_data_loader, class_names
