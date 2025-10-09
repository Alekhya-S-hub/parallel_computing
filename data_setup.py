import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image

# Custom dataset class that skips unreadable images
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except Exception as e:
            print(f"⚠️ Skipping file {path}: {e}")
            # return a blank RGB image (150x150) to keep indices consistent
            sample = Image.new("RGB", (150, 150))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

def get_dataloaders(data_dir):
    """
    Loads dataset from directory and returns train/test DataLoaders and device.
    """
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load dataset safely
    dataset = SafeImageFolder(root=data_dir, transform=transform)
    print(f"Total images: {len(dataset)} | Classes: {dataset.classes}")

    # Split into train/test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # DataLoaders — num_workers=0 (for Windows safety)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # GPU / CPU device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    return train_loader, test_loader, device

# ✅ Run directly for quick testing
if __name__ == "__main__":
    data_dir = r"recycle_subset"
    train_loader, test_loader, device = get_dataloaders(data_dir)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")


data_dir = r"recycle_subset"
train_loader, test_loader, device = get_dataloaders(data_dir)
print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")