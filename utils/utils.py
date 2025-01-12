from pathlib import Path
import torch
from tqdm import tqdm 
import torch.optim as optim
from torch import nn
from torch.nn import functional as F

def download_from_gdrive_and_extract_zip(file_id: str, save_path: Path, extract_path: Path):
    """
    Downloads a ZIP file from Google Drive using its file ID and extracts its contents to a specified directory.

    Args:
        file_id (str): The Google Drive file ID of the ZIP file to download.
        save_path (Path): The path where the downloaded ZIP file will be saved.
        extract_path (Path): The directory where the ZIP file will be extracted.
    """
    import os
    import zipfile

    import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    if not save_path.exists():
        gdown.download(url, str(save_path), quiet=False)
        print(f"File downloaded and saved to {save_path}")

    if not extract_path.exists():
        print(f"Starting to extract... {extract_path}")
        # Unzip the file
        with zipfile.ZipFile(save_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        print(f"File extracted to {extract_path}")


def train_model(model, device, loader):
    # Parameters
    torch.manual_seed(123)
    num_epochs = 50

    # Create Loss-Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=10e-5)

    pbar = tqdm(total=num_epochs * len(loader))

    step = 0
    for epoch in range(0, num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(loader):

            images, label_masks, ground_truth = data

            # Forward-Pass
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss = criterion(logits.to(torch.float32), label_masks.to(device))

            # backprop
            probs = F.softmax(logits, dim=1)
            loss.backward()
            optimizer.step()

            pred = probs.argmax(dim=(1), keepdim=True).to(torch.int).cpu()
            pixel_acc = (ground_truth == pred).to(torch.float).mean()

            # print statistics
            running_loss += loss.item()
            running_acc += pixel_acc
            step += 1
            print_every = 10
            if (i % print_every) == (print_every - 1):
                desc = f"Epoch: {epoch + 1}, Iteration: {i + 1:5d}] Loss: {running_loss / print_every:.3f} Acc: {running_acc / print_every:.3f}"
                _ = pbar.update(print_every)
                _ = pbar.set_description(desc)
                running_loss = 0.0
                running_acc = 0.0
    pbar.close()

    print("Finished Training")

def evaluate_model(model, dataloader, device, num_samples):
    model = model.to(device)
    running_acc = 0.0
    iou = 0
    dice = 0
    step = 0
    pbar = tqdm(total=len(dataloader))
    
    with torch.no_grad():
        for i, (images, _, ground_truth) in enumerate(dataloader):
            
            # Get predictions
            ground_truth = ground_truth.to(device)
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1, keepdim=True).to(torch.int)

            # Pixel Accuracy
            pixel_acc = (ground_truth == pred).to(torch.float).mean()

            # IoU and Dice Score
            intersection = (pred & ground_truth).to(torch.float).sum()
            union = (pred | ground_truth).to(torch.float).sum()
            dice += 2 * intersection / (intersection + union + 1e-8)
            iou += intersection / (union + 1e-8)

            # print statistics
            running_acc += pixel_acc.cpu()
            step += 1
            print_every = 10
            if (i % print_every) == (print_every - 1):
                desc = f"Iteration: {i + 1:5d}] Acc: {running_acc / print_every:.3f}"
                _ = pbar.update(print_every)
                _ = pbar.set_description(desc)
                running_acc = 0.0
        pbar.close()

        # Average metrics over all samples
        iou = iou.cpu() / num_samples
        dice = dice.cpu() / num_samples

        return pixel_acc, iou, dice