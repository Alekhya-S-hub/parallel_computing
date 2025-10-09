import torch
from torchvision import models
from data_setup import get_dataloaders
import time

# === Load test data ===
data_dir = r"recycle_subset"
_, test_loader, device = get_dataloaders(data_dir)

# === Load trained model ===
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)
model.load_state_dict(torch.load("recycle_resnet18.pth", map_location=device))
model.to(device)
model.eval()

# === Batch Inference Timing ===
print("Running batch inference test on:", device)
torch.cuda.synchronize()

num_images = 0
start_time = time.time()

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        num_images += images.size(0)

torch.cuda.synchronize()
end_time = time.time()

total_time = end_time - start_time
throughput = num_images / total_time
print(f"\n✅ Inference complete!")
print(f"Processed {num_images} images in {total_time:.2f} seconds.")
print(f"Throughput: {throughput:.2f} images/second")
