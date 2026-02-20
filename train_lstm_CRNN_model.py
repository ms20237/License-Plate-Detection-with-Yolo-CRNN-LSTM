import argparse
import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, random_split

from utils import PlateDataset, collate_fn, decode_predictions, cer, clean_and_analyze_labels, downsample_digits
from CRNN import CRNN


def init():
    parser = argparse.ArgumentParser(description="train LSTM/CRNN model.")
    parser.add_argument("--images_folder", 
                        type=str, 
                        required=True,
                        help="Path to you images dataset.")
    
    parser.add_argument("--labels_file", 
                        type=str, 
                        required=True,
                        help="Path to you txt label file.")
    
    parser.add_argument("--epochs", 
                        type=int, 
                        default=20,
                        help="Number of epochs for training model.")
    
    parser.add_argument("--val_split", 
                        type=float, 
                        default=0.1,
                        help="Val Split percentage for Training.")
    
    parser.add_argument("--test_split", 
                        type=float, 
                        default=0.1,
                        help="Test Split percentage for Training")
    
    parser.add_argument("--output_model_path", 
                        type=str, 
                        default="./models/recg_models",
                        help="Path for save trained model.")
    
    parser.add_argument("--balance_dataset",  
                        action='store_true',
                        help="Enable balance between letters in dataset.")

    return parser.parse_args()


def run(images_folder: str,
        labels_file: str,
        epochs: int,
        val_split: float,
        test_split: float,
        output_model_path: str,
        balance_dataset: bool,
        lr: float = 1e-4):

    os.makedirs(output_model_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    num_classes = len(alphabet) + 1

    # Clean dataset
    labels_file = clean_and_analyze_labels(labels_file, min_char_freq=20)

    # Optional balancing
    if balance_dataset:
        labels_file = downsample_digits(labels_file, max_digit_ratio=0.7)
    
    dataset = PlateDataset(images_folder, labels_file)

    # SPLIT 
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = CRNN(num_classes).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    val_cers = []

    best_val_loss = float("inf")

    # TRAIN LOOP 
    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for imgs, labels, label_lengths in tqdm(train_loader):

            imgs = imgs.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            outputs = model(imgs)

            output_lengths = torch.full(
                size=(imgs.size(0),),
                fill_value=outputs.size(1),
                dtype=torch.long,
                device=device
            )

            loss = criterion(
                outputs.permute(1, 0, 2),
                labels,
                output_lengths,
                label_lengths
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for imgs, labels, label_lengths in val_loader:

                imgs = imgs.to(device)
                labels = labels.to(device)
                label_lengths = label_lengths.to(device)

                outputs = model(imgs)

                output_lengths = torch.full(
                    size=(imgs.size(0),),
                    fill_value=outputs.size(1),
                    dtype=torch.long,
                    device=device
                )

                loss = criterion(
                    outputs.permute(1, 0, 2),
                    labels,
                    output_lengths,
                    label_lengths
                )

                val_loss += loss.item()

                preds = decode_predictions(outputs, alphabet)

                # reconstruct targets
                start = 0
                targets = []
                for l in label_lengths:
                    l = l.item()
                    text = ""
                    for i in range(l):
                        text += alphabet[labels[start+i].item() - 1]
                    targets.append(text)
                    start += l

                all_preds.extend(preds)
                all_targets.extend(targets)

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        val_error = cer(all_preds, all_targets)
        val_cers.append(val_error)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val CER: {val_error:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(output_model_path, "best_crnn.pth"))

    print("\nEvaluating on Test Set...")
    model.load_state_dict(torch.load(os.path.join(output_model_path, "best_crnn.pth")))
    model.eval()

    test_preds = []
    test_targets = []

    with torch.no_grad():
        for imgs, labels, label_lengths in test_loader:

            imgs = imgs.to(device)
            outputs = model(imgs)

            preds = decode_predictions(outputs, alphabet)
            test_preds.extend(preds)

    print("Test evaluation done.")

    plt.figure()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])
    plt.savefig(os.path.join(output_model_path, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(val_cers)
    plt.title("Validation CER")
    plt.xlabel("Epoch")
    plt.ylabel("CER")
    plt.savefig(os.path.join(output_model_path, "cer_curve.png"))
    plt.close()

    print("Training complete. Plots saved.")

        
if __name__ == "__main__":
    args = init()
    run(args.images_folder,
        args.labels_file,
        args.epochs,
        args.val_split,
        args.test_split,
        args.output_model_path,
        args.balance_dataset)    
    
    