import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

# --- hyperparameters ---
image_size = (224, 224)
batch_size = 64
epochs = 10
learning_rate = 1e-3

root_dir = pathlib.Path("/home/pc/dataset/Rice_Image_Dataset")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)

# --- transforms ---
transform = T.Compose([
    T.Resize(image_size),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)),
])

# --- dataset (CPU-side; no .to(device) here) ---
dataSetImage = torchvision.datasets.ImageFolder(
    root=root_dir,
    transform=transform,
)
print("classes:", dataSetImage.classes)
print("num images:", len(dataSetImage))

# --- optional: train/val split ---
val_ratio = 0.2
n_val = int(len(dataSetImage) * val_ratio)
n_train = len(dataSetImage) - n_val
train_set, val_set = random_split(
    dataSetImage, [n_train, n_val],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(
    train_set, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_set, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=4, 
    pin_memory=True
)



import torchvision.models as models

num_classes = len(dataSetImage.classes)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)


model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)



import time

def train_one_epoch(model, loader, criterion, optimizer):
    
    model.train() # Sets the model to training mode
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels) # Calculates the prediction error.

        loss.backward() # Calculates the error gradient.
        optimizer.step() # It updates the model weights using the gradients.

        running_loss += loss.item() * images.size(0) # It keeps track of the sum of errors for calculating the average.
        correct += (outputs.argmax(1) == labels).sum().item() # For each image, it selects the class that has received the highest score.
        total += labels.size(0)

    return running_loss / total, correct / total # The function returns the mean error and accuracy.



# Model Evaluation

@torch.no_grad() # Do not calculate the gradient.
def evaluate(model, loader, criterion):
    model.eval() # Sets the model to evaluation mode.
    
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total # It is the average error and accuracy on the validation data.



history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_acc = 0.0

for epoch in range(1, epochs + 1):
    t0 = time.time()
    
    tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    va_loss, va_acc = evaluate(model, val_loader, criterion)
    
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc)

    print(
        f"epoch {epoch:02d}/{epochs} | "
        f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
        f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
        f"{time.time()-t0:.1f}s"
    )

    if va_acc > best_acc:
        best_acc = va_acc
        torch.save(model.state_dict(), "best_resnet18_rice.pth")
        print(f"  ✓ saved (best val acc = {best_acc:.4f})")


model.load_state_dict(torch.load("best_resnet18_rice.pth", map_location=device))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images.to(device))
        y_pred.extend(outputs.argmax(1).cpu().tolist())
        y_true.extend(labels.tolist())

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print(classification_report(y_true, y_pred, target_names=dataSetImage.classes))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=dataSetImage.classes)
fig, ax = plt.subplots(figsize=(7, 7))
disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()


from PIL import Image

def predict(img_path, model, transform, classes):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    conf, idx = probs.max(0)
    return classes[idx.item()], conf.item(), probs.cpu().numpy()

# label, conf, probs = predict("test.jpg", model, transform, dataSetImage.classes)
# print(label, f"{conf:.3f}")

