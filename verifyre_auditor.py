import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import argparse
import random
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
import ssl

# Bypass SSL certificate verify failed errors for downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- CONFIGURATION ---
# We append "Deforestation" as the 11th class (Index 10)
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
    "Deforestation" 
]
MODEL_PATH = "auditor_weights.pth"

# Integration Files
SNOWFLAKE_EXPORT = "audit_manifest_export.json"
AUDIO_BRIEFING = "agent_daily_briefing.txt"
SECTOR_DATABASE = {
    "SEC-999-DEMO": {
        "owner": "Drax Biomass Inc.",
        "timber_mark": "EM2960",
        "coordinates": "54.2°N, 125.7°W",
        "region": "Primary Rainforest Zone 4",
        "permit_type": "Restricted-B"
    },
    "SEC-001-DEMO": {
        "owner": "GreenGuard Forestry Ltd.",
        "timber_mark": "GG-9000",
        "coordinates": "55.1°N, 126.2°W",
        "region": "Sustainable Harvest Zone A",
        "permit_type": "unrestricted"
    }
}

# --- CUSTOM DATASET CLASS ---
class CustomOversampleDataset(Dataset):
    def __init__(self, root_dir, class_index, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_index = class_index
        # Find all images
        self.image_paths = glob.glob(os.path.join(root_dir, "*.*"))
        
    def __len__(self):
        # We OVERSAMPLE widely to ensure the model learns this sparse class
        # 11 images * 200 repetitions = 2200 samples (comparable to EuroSAT classes)
        return len(self.image_paths) * 200

    def __getitem__(self, idx):
        # Modulo to cycle through the small set
        real_idx = idx % len(self.image_paths)
        img_path = self.image_paths[real_idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.class_index
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), self.class_index

# --- HEATMAP ENGINE ---
class HeatmapGenerator:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        
    def register_hooks(self):
        self.hook_handles.append(self.model.layer4[1].register_forward_hook(self.save_activation))
        self.hook_handles.append(self.model.layer4[1].register_full_backward_hook(self.save_gradient))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, output_idx):
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activation = self.activations[0]
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activation, dim=0).cpu().detach()
        heatmap = np.maximum(heatmap, 0)
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
        return heatmap.numpy()

# --- AUDIT RULES ---
def apply_audit_rules(claim, prediction, confidence, sector_info=None):
    owner_str = f" ({sector_info['owner']})" if sector_info and 'owner' in sector_info else ""
    conf_str = f" [{confidence:.1%}]"

    # --- Rule Set 1: Explicit Deforestation Detection ---
    if prediction == "Deforestation":
        return f"CRITICAL: Deforestation Confirmed{owner_str}{conf_str}", "CRITICAL", "SOL-SLASH"

    # --- Rule Set 2: Protected Area Violations (The 'Lying' Logic) ---
    # If the company claims "Forest" or "HerbaceousVegetation" (Protected Land), 
    # ANY detecting of Industrial, Crop, Highway, or Residential is a FRAUD.
    protected_claims = ["Forest", "HerbaceousVegetation", "Protected"]
    fraud_indicators = ["Industrial", "AnnualCrop", "PermanentCrop", "Highway", "Residential"]

    if claim in protected_claims and prediction in fraud_indicators:
         return f"CRITICAL: Protected Area Violation - {prediction} detected in {claim} zone{owner_str}{conf_str}", "CRITICAL", "SOL-SLASH"

    # --- Rule Set 3: Water Protection (Dark Vessel / Pollution) ---
    if claim in ["SeaLake", "River"] and prediction == "Industrial":
         return f"CRITICAL: 'Dark Vessel'{owner_str} Signature Detected", "CRITICAL", "SOL-SLASH"

    # --- Rule Set 4: Verification ---
    if claim == prediction:
        return f"VERIFIED: {prediction} confirmed{conf_str}", "LOW", "SOL-MINT"
    
    # --- Default Mismatch ---
    return f"MISMATCH: {prediction} detected (Claim: {claim}){conf_str}", "HIGH", "SOL-HOLD"

# --- VISUALIZATION ---
def save_evidence_heatmap(image_tensor, heatmap, filename="evidence_locker/heatmap_evidence.jpg"):
    os.makedirs("evidence_locker", exist_ok=True)
    img = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.imshow(img); plt.title("Satellite Feed"); plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    # Highlight anomalies in RED (jet = Blue to Red)
    # We use 'jet' so that Low(0)=Blue (Cold) and High(1)=Red (Hot/Anomalous)
    plt.imshow(heatmap, cmap='jet', alpha=0.35, extent=[0, 224, 224, 0], interpolation='bicubic') 
    plt.title(f"Deforestation Risk Map")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

# --- LOADING & TRAINING ---
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(EUROSAT_CLASSES)) # 11 Classes
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
            print(f"[SYSTEM] Loaded weights from {MODEL_PATH}")
        except:
            print("[WARNING] Weight mismatch (class count changed?). Starting fresh.")
    model.eval()
    return model

def train_auditor(epochs=1):
    console = Console()
    console.print(Panel.fit("[bold green]TRAINING VISUAL CORTEX (HYBRID DATASET)[/bold green]", border_style="green"))

    # Transforms with Augmentation
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. EuroSAT
    datasets_list = []
    try:
        eurosat = datasets.EuroSAT(root="./data", download=True, transform=train_transform)
        datasets_list.append(eurosat)
    except:
        console.print("[red]EuroSAT download failed. Aborting.[/red]")
        return

    # 2. Custom Deforestation Data (Class 10 - Deforestation)
    defo_path = os.path.abspath("data/custom_classification/Deforestation")
    if os.path.exists(defo_path):
        defo_dataset = CustomOversampleDataset(defo_path, class_index=10, transform=train_transform)
        console.print(f"[INFO] Injected {len(defo_dataset)} oversampled Deforestation samples.")
        datasets_list.append(defo_dataset)

    # 3. Custom Forest Data (Class 1 - Forest) for Balancing
    forest_path = os.path.abspath("data/custom_classification/Forest")
    if os.path.exists(forest_path):
        # We assume 'Forest' is index 1 in EuroSAT (Alphabetical: AnnualCrop, Forest...)
        forest_dataset = CustomOversampleDataset(forest_path, class_index=1, transform=train_transform)
        console.print(f"[INFO] Injected {len(forest_dataset)} oversampled Forest samples (Balancing).")
        datasets_list.append(forest_dataset)

    full_dataset = ConcatDataset(datasets_list)
    
    # Check Classes
    # EuroSAT targets are 0-9. Defo is 10. Perfect.
    
    loader = torch.utils.data.DataLoader(full_dataset, batch_size=32, shuffle=True)
    
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(EUROSAT_CLASSES))
    model.train()
    
    optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    
    for ep in range(epochs):
        console.print(f"--- Epoch {ep+1}/{epochs} ---")
        run_loss = 0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(loader):
            optim.zero_grad()
            outputs = model(inputs)
            loss = crit(outputs, labels)
            loss.backward()
            optim.step()
            
            run_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if i % 20 == 0:
                print(f"\rBatch {i}/{len(loader)} Loss: {loss.item():.4f} Acc: {100*correct/total:.1f}%", end="")
        
        print(f"\nEpoch {ep+1} Acc: {100*correct/total:.2f}%")
        
    torch.save(model.state_dict(), MODEL_PATH)
    console.print("[bold green]Training Complete. Weights Saved.[/bold green]")

# --- MAIN EXECUTION ---
def audit_single_image(image_path, claim):
    console = Console()
    console.print(f"[bold cyan]Auditing:[/bold cyan] {image_path}")
    
    model = load_model()
    cam = HeatmapGenerator(model)
    cam.register_hooks()
    
    tf = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    try:
        img = Image.open(image_path).convert('RGB')
        tens = tf(img).unsqueeze(0)
    except: return
    
    # Predict
    out = model(tens)
    probs = torch.nn.functional.softmax(out, dim=1)
    top_p, top_idx = torch.max(probs, 1)
    pred_class = EUROSAT_CLASSES[top_idx.item()]
    
    # Heatmap
    model.zero_grad()
    out[:, top_idx.item()].backward()
    hm = cam.generate(top_idx.item())
    ev_path = save_evidence_heatmap(tens, hm)
    
    # Info
    s_info = {}
    fname = os.path.basename(image_path)
    if "sector-4" in fname: s_info = SECTOR_DATABASE["SEC-999-DEMO"]
    elif "sector-1" in fname: s_info = SECTOR_DATABASE["SEC-001-DEMO"]
    
    msg, risk, token = apply_audit_rules(claim, pred_class, top_p.item(), s_info)
    
    # Table
    tab = Table(title="AUDIT RESULT", box=box.HEAVY_EDGE)
    tab.add_column("Target"); tab.add_column("Prediction", style="yellow"); tab.add_column("Risk", style="red" if risk=="CRITICAL" else "green")
    tab.add_row(fname, f"{pred_class} ({top_p.item():.2%})", f"{risk}\n{msg}")
    console.print(tab)
    console.print(f"Evidence: {ev_path}")

# --- API HELPER FOR SERVER ---
def get_fraud_probability(sector_id, claim, image_path=None):
    """
    Live inference function for the API.
    Loads model, runs audit on image_path, saves heatmap, returns JSON result.
    """
    try:
        # Load Model
        model = load_model()
        cam = HeatmapGenerator(model)
        cam.register_hooks()
        
        # Load Image
        if not image_path:
             return {"error": "No image path provided"}
             
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)
        except Exception as e:
            return {"error": f"Image load failed: {e}"}

        # Predict
        model.zero_grad()
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, predicted_idx = torch.max(probabilities, 1)
        predicted_class = EUROSAT_CLASSES[predicted_idx.item()]
        
        # Generate Heatmap
        outputs[:, predicted_idx.item()].backward()
        heatmap = cam.generate(predicted_idx.item())
        cam.remove_hooks()
        
        # Save Heatmap (ensure static/heatmaps exists)
        heatmap_filename = f"{sector_id}_heatmap.jpg"
        heatmap_path = os.path.join("static", "heatmaps", heatmap_filename)
        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
        
        save_evidence_heatmap(image_tensor, heatmap, filename=heatmap_path)
        
        # Lookup DB Info
        sector_info = SECTOR_DATABASE.get(sector_id, SECTOR_DATABASE.get("SEC-999-DEMO"))
        
        # Apply Rules
        status_msg, risk_level, token = apply_audit_rules(claim, predicted_class, top_prob.item(), sector_info)
        
        # Construct Response
        return {
            "sector_id": sector_id,
            "claim": claim,
            "prediction": predicted_class,
            "confidence": float(top_prob.item()),
            "risk_level": risk_level,
            "status_message": status_msg,
            "action_token": token,
            "heatmap_url": f"http://localhost:8000/static/heatmaps/{heatmap_filename}",
            "timestamp": "2026-02-07T14:00:00Z", # In real app, use datetime.now()
            "owner": sector_info["owner"],
            "timber_mark": sector_info.get("timber_mark", "N/A"),
            "coordinates": sector_info.get("coordinates", "N/A"),
            "region": sector_info.get("region", "N/A"),
            "permit_type": sector_info.get("permit_type", "N/A")
        }
        
    except Exception as e:
        return {"error": f"Audit Engine Failure: {str(e)}"} 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--image", type=str)
    parser.add_argument("--claim", type=str, default="Forest")
    args = parser.parse_args()
    
    if args.train: train_auditor()
    elif args.image: audit_single_image(args.image, args.claim)
