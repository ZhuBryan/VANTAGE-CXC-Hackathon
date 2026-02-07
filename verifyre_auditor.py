import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import argparse
import random
import os
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
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]
MODEL_PATH = "auditor_weights.pth"

# Integration Output Files
SNOWFLAKE_EXPORT = "audit_manifest_export.json"
AUDIO_BRIEFING = "agent_daily_briefing.txt"
SOLANA_TX_LOG = "solana_transaction_queue.json"

# --- THE "WIZARD OF OZ" OWNERSHIP DATABASE ---
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
    },
    "SEC-3391": {
        "owner": "Oceanic Transport Corp",
        "timber_mark": "N/A",
        "coordinates": "41.3°N, 19.8°E",
        "region": "Marine Protected Area z7",
        "permit_type": "No-Entry"
    },
    "SEC-8842": {
        "owner": "AgriBiz MegaCorp",
        "timber_mark": "AG-1234",
        "coordinates": "45.0°N, 0.5°E",
        "region": "Carbon Offset Block C",
        "permit_type": "Conservation-Only"
    }
}

# --- GRAD-CAM ENGINE (The "Heatmap" Logic) ---
class HeatmapGenerator:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        
    def register_hooks(self):
        # Hook into the final convolutional layer of ResNet
        self.hook_handles.append(self.model.layer4[1].register_forward_hook(self.save_activation))
        # Use register_full_backward_hook to catch gradients
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
        # Pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the channels by corresponding gradients
        activation = self.activations[0]
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the activations
        heatmap = torch.mean(activation, dim=0).cpu().detach()
        
        # ReLU on top to keep only positive features
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
            
        return heatmap.numpy()

# --- AUDIT LOGIC KERNEL (Advanced) ---
def apply_audit_rules(claim, prediction, confidence, sector_info=None):
    """
    Advanced business logic for Verifyre ESG Audit.
    Mocks "Dark Vessel" and "Machinery" detection based on class contradictions.
    Includes "Blame Logic" to specifically name shamed companies.
    """
    risk_level = "LOW"
    action_token = "SOL-MINT"
    msg = "VERIFIED: Land use matches reported manifest."
    
    owner_str = f" ({sector_info['owner']})" if sector_info and 'owner' in sector_info else ""

    # Scenario A: The "Dark Vessel" (Industrial activity in Water/Nature zones)
    # EuroSAT 'Industrial' or 'Highway' inside 'SeaLake' or 'River'
    if claim in ["SeaLake", "River"] and prediction in ["Industrial"]:
         return f"CRITICAL: 'Dark Vessel'{owner_str} Signature Detected (AIS Off)", "CRITICAL", "SOL-SLASH"

    # Scenario B: Heavy Machinery in Protected Rainforest
    # Claiming Forest, but finding Industrial/Highway
    if claim == "Forest" and prediction in ["Industrial", "Highway"]:
        return f"CRITICAL: Heavy Machinery / Illegal Logging by{owner_str}", "CRITICAL", "SOL-SLASH"

    # Scenario C: Carbon Fraud
    if claim == "Forest" and prediction in ["AnnualCrop", "PermanentCrop", "Pasture"]:
        return f"FRAUD: Iliicit Agriculture{owner_str} in Carbon Credit Zone", "HIGH", "SOL-BURN"

    # Scenario D: Urban Sprawl (Habitat Destruction)
    if claim in ["HerbaceousVegetation", "Pasture"] and prediction == "Residential":
        return f"WARNING: Unreported Urban Encroachment{owner_str}", "MEDIUM", "SOL-FLAG"

    # Scenario E: Clean
    if claim == prediction:
        return f"VERIFIED: {prediction} confirmed ({confidence:.1%})", "LOW", "SOL-MINT"
    
    return "MISMATCH: Land Use Discrepancy", "UNKNOWN", "SOL-HOLD"

# --- VISUALIZATION ("The Wow") ---
def save_evidence_heatmap(image_tensor, heatmap, filename="evidence_locker/heatmap_evidence.jpg"):
    os.makedirs("evidence_locker", exist_ok=True)
    
    # Denormalize image for display
    img = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)

    plt.figure(figsize=(10, 4))
    
    # Original
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Satellite Feed (Optical)")
    plt.axis('off')
    
    # Heatmap Overlay
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    # User requested "smaller pixels" -> smoother interpolation (bicubic)
    # User requested "bare ground red" (it was blue) -> Invert colormap (jet_r) to highlight the "cold" spots of the original map
    plt.imshow(heatmap, cmap='jet_r', alpha=0.6, extent=[0, 224, 224, 0], interpolation='bicubic') 
    plt.title("AI Attention Map (Anomaly)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

# --- MODEL SETUP ---
def load_auditor_model():
    print("[SYSTEM] Initializing Neural Link (ResNet-18)...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Modify FC layer for EuroSAT 10 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(EUROSAT_CLASSES))
    
    # Load trained weights if available
    if os.path.exists(MODEL_PATH):
        print(f"[SYSTEM] Loading optimized weights ({MODEL_PATH})...")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("[WARNING] No trained weights found. Model is running in UNCALIBRATED (Random) mode.")
    
    # Ensure gradients can be calculated even in Eval mode for GradCAM
    model.eval()
    return model

# --- DATASET HANDLING ---
def get_dataset(root="./data"):
    print("[SYSTEM] Acquiring Satellite Feed (EuroSAT)...")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    try:
        dataset = datasets.EuroSAT(root=root, download=True, transform=transform)
        return dataset
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return None

# --- TRAINING MODE ---
def train_auditor(epochs=1):
    console = Console()
    console.print(Panel.fit("[bold green]INITIATING TRAINING SEQUENCE[/bold green]", border_style="green"))

    # Setup
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(EUROSAT_CLASSES))
    model.train() 

    dataset = get_dataset()
    if not dataset: return

    BATCH_SIZE = 32
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Calculation for user:
    total_batches = len(dataloader)
    console.print(f"[INFO] Dataset Size: {len(dataset)} images")
    console.print(f"[INFO] Batch Size: {BATCH_SIZE}")
    console.print(f"[INFO] Total Batches per Epoch: [bold cyan]{total_batches}[/bold cyan]")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    console.print(f"[SYSTEM] Training on {len(dataset)} satellite images for {epochs} epochs...")

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 10 == 0:
                print(f"\rEpoch {epoch+1}/{epochs} | Batch {i}/{total_batches} | Loss: {loss.item():.4f} | Acc: {100 * correct / total:.2f}%", end="")

        print(f"\n[EPOCH {epoch+1} COMPLETE] Average Accuracy: {100 * correct / total:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    console.print(f"[SUCCESS] Weights saved to [bold cyan]{MODEL_PATH}[/bold cyan]")
    console.print("[SYSTEM] Neural Link Calibrated. You may now run the audit.")

# --- SINGLE IMAGE AUDIT (With Heatmap) ---
def audit_single_image(image_path, claimed_label):
    console = Console()
    console.print(f"[bold cyan]Initiating Targeted Audit:[/bold cyan] {image_path}")

    model = load_auditor_model()
    
    # Setup Grad-CAM
    cam_generator = HeatmapGenerator(model)
    cam_generator.register_hooks()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
    except Exception as e:
        console.print(f"[bold red]ERROR: Could not process image. {e}[/bold red]")
        return

    # Inference with Gradient Tracking for Heatmap
    model.zero_grad()
    outputs = model(image_tensor)
    
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    top_prob, predicted_idx = torch.max(probabilities, 1)
    predicted_class = EUROSAT_CLASSES[predicted_idx.item()]
    
    # Backward pass to get gradients for Heatmap
    outputs[:, predicted_idx.item()].backward()
    
    heatmap = cam_generator.generate(predicted_idx.item())
    cam_generator.remove_hooks()
    
    # --- INTELLIGENT LOOKUP (Simulating DB connection based on filename) ---
    sector_id = "SEC-UNKNOWN"
    sector_info = {"owner": "Unknown Entity", "coordinates": "Unknown", "region": "Unmapped"}
    
    fname = os.path.basename(image_path)
    if "sector-4" in fname:
        sector_id = "SEC-999-DEMO" # Drax
    elif "sector-1" in fname:
        sector_id = "SEC-001-DEMO" # Clean
    
    if sector_id in SECTOR_DATABASE:
        sector_info = SECTOR_DATABASE[sector_id]
    
    # Logic
    status_msg, risk_level, token = apply_audit_rules(claimed_label, predicted_class, top_prob.item(), sector_info)
    
    # Heatmap Saving
    evidence_path = save_evidence_heatmap(image_tensor, heatmap)
    
    # Display
    table = Table(title="TARGETED AUDIT RESULT", box=box.HEAVY_EDGE, border_style="bright_magenta")
    table.add_column("Target", style="cyan")
    table.add_column("Owner", style="magenta")
    table.add_column("Coordinates", style="dim")
    table.add_column("Claimed", style="white")
    table.add_column("AI Prediction", style="yellow")
    table.add_column("Confidence", justify="right")
    table.add_column("Risk Level", justify="center")
    table.add_column("Action", style="green")

    risk_style = "green"
    if risk_level == "HIGH": risk_style = "bold red"
    elif risk_level == "CRITICAL": risk_style = "bold red blink"
    elif risk_level == "MEDIUM": risk_style = "yellow"

    table.add_row(
        os.path.basename(image_path),
        sector_info["owner"],
        sector_info["coordinates"],
        claimed_label,
        predicted_class,
        f"{top_prob.item():.2%}",
        f"[{risk_style}]{status_msg}[/{risk_style}]",
        token
    )
    
    console.print(table)
    console.print(f"[dim]Evidence saved to:[/dim] [link=file://{os.path.abspath(evidence_path)}]{evidence_path}[/link]")
    
    # Generate Integration Files
    generate_integration_files([{
        "id": sector_id, 
        "claim": claimed_label, 
        "prediction": predicted_class, 
        "risk": risk_level, 
        "token": token,
        "owner": sector_info["owner"],
        "coordinates": sector_info["coordinates"]
    }])


# --- INTEGRATION OUTPUT WRITER ---
def generate_integration_files(audit_results):
    # 1. Member 2 (Snowflake / Gemini) - JSON Manifest
    with open(SNOWFLAKE_EXPORT, "w") as f:
        json.dump({
            "audit_session": "SESSION_ID_X99",
            "results": audit_results,
            "metadata": {"source": "EuroSAT V2", "auditor": "Verifyre-AI"}
        }, f, indent=4)
        print(f"[INTEGRATION] Manifest exported to {SNOWFLAKE_EXPORT}")

    # 2. Member 3 (ElevenLabs) - Text Briefing
    with open(AUDIO_BRIEFING, "w") as f:
        f.write("ATTENTION UNIT 4. PRIORITY ALERT. \n")
        cnt = 0
        for res in audit_results:
            if res["risk"] in ["CRITICAL", "HIGH"]:
                f.write(f"Detected {res['risk']} anomaly in sector {res['id']}. Claimed {res['claim']} but sensors indicate {res['prediction']}. Recommendation: {res['token']}.\n")
                cnt += 1
        if cnt == 0:
            f.write("All sectors reporting nominal status. No action required.\n")
    print(f"[INTEGRATION] Audio script generated at {AUDIO_BRIEFING}")

# --- API HELPER FUNCTION ---
def get_fraud_probability(sector_id, claim, image_path=None):
    """
    API-ready function to return JSON result for a specific sector audit.
    If image_path is None, it selects a random image from the dataset to simulate the sector.
    """
    model = load_auditor_model()
    
    # Lookup Sector Info
    sector_info = SECTOR_DATABASE.get(sector_id, {})

    # If no image provided, pick a random one from dataset to simulate 'real-time feed'
    if not image_path:
        # Code omitted for brevity, logic remains same: pick random image
        # ...
        try:
           dataset = get_dataset()
           if not dataset: return {"error": "Dataset unavailable"}
           
           # If random choice logic:
           root_dir = "data/eurosat/2750"
           if not os.path.exists(root_dir): return {"error": "Data dir missing"}
           
           all_classes = os.listdir(root_dir)
           random_class = random.choice(all_classes)
           class_dir = os.path.join(root_dir, random_class)
           random_file = random.choice(os.listdir(class_dir))
           image_path = os.path.join(class_dir, random_file)
        except Exception as e:
            return {"error": f"Image selection failed: {e}"}

    # Setup Grad-CAM
    cam_generator = HeatmapGenerator(model)
    cam_generator.register_hooks()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
    except Exception as e:
        return {"error": f"Could not process image: {str(e)}"}

    # Inference
    model.zero_grad()
    outputs = model(image_tensor)
    
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    top_prob, predicted_idx = torch.max(probabilities, 1)
    predicted_class = EUROSAT_CLASSES[predicted_idx.item()]
    
    # Heatmap
    outputs[:, predicted_idx.item()].backward()
    heatmap = cam_generator.generate(predicted_idx.item())
    cam_generator.remove_hooks()
    
    # Logic with Sector Info
    status_msg, risk_level, token = apply_audit_rules(claim, predicted_class, top_prob.item(), sector_info)
    
    # Save Heatmap for API to serve
    filename = f"static/heatmaps/{sector_id}_heatmap.jpg"
    os.makedirs("static/heatmaps", exist_ok=True)
    save_evidence_heatmap(image_tensor, heatmap, filename=filename)
    
    result = {
        "sector_id": sector_id,
        "claim": claim,
        "prediction": predicted_class,
        "confidence": float(top_prob.item()),
        "risk_level": risk_level,
        "status_message": status_msg,
        "action_token": token,
        "heatmap_url": f"http://localhost:8000/{filename}",
        "timestamp": "2026-02-07T12:00:00Z"
    }
    
    # Merge DB info directly into result for frontend display
    if sector_info:
        result.update(sector_info)
        
    return result

# --- SIMULATION ---
def simulate_audit_batch(n=10):
    console = Console()
    
    # Header
    console.print(Panel.fit(
        "[bold cyan]VERIFYRE[/bold cyan] [green]SATELLITE AUDIT PROTOCOL v2.0[/green]\n"
        "[dim]Connecting to Solana Mainnet... Connected.[/dim]\n"
        "[dim]Syncing with Snowflake DB... Connected.[/dim]",
        border_style="cyan"
    ))

    model = load_auditor_model()
    dataset = get_dataset()
    
    if not dataset:
        console.print("[bold red]FATAL: Satellite uplink offline. (Dataset download failed)[/bold red]")
        return
    
    train_warning = "[bold red]UNCALIBRATED[/bold red]" if not os.path.exists(MODEL_PATH) else "[green]CALIBRATED[/green]"
    console.print(f"[SYSTEM] Model Status: {train_warning}")

    table = Table(title="audit_batch_result_log", box=box.HEAVY_EDGE, border_style="bright_blue")
    
    table.add_column("Sector ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Claimed Land Use", style="magenta")
    table.add_column("Satellite Reality", style="yellow")
    table.add_column("Risk Level", justify="center")
    table.add_column("Action Token", justify="right", style="green")

    results_for_export = []

    # Simulation Loop
    indices = random.sample(range(len(dataset)), n)
    
    for idx in indices:
        image_tensor, label_idx = dataset[idx]
        true_label = EUROSAT_CLASSES[label_idx] 
        
        # Run Inference
        with torch.no_grad():
            outputs = model(image_tensor.unsqueeze(0))
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, predicted_idx = torch.max(probabilities, 1)
            predicted_class = EUROSAT_CLASSES[predicted_idx.item()]
            
            # If untrained, force 'Reality' to be the true label for valid logic testing
            # If trained, use predicted_class
            if not os.path.exists(MODEL_PATH):
                 satellite_reality = true_label 
                 confidence = 0.99
            else:
                 satellite_reality = predicted_class
                 confidence = top_prob.item()

        # Generate a "Claim" to test our Audit Logic (Simulation Bias)
        roll = random.random()
        if roll < 0.4:
            claimed_use = satellite_reality
        elif roll < 0.7:
             if satellite_reality in ["Industrial", "Highway"]:
                 claimed_use = "Forest" # Detection: Heavy Machinery
             elif satellite_reality in ["Industrial"]:
                 claimed_use = "SeaLake" # Detection: Dark Vessel
             elif satellite_reality == "Pasture":
                 claimed_use = "HerbaceousVegetation"
             else:
                 claimed_use = random.choice(EUROSAT_CLASSES)
        else:
            claimed_use = random.choice(EUROSAT_CLASSES)

        # Apply Audit Logic
        status_msg, risk_level, token = apply_audit_rules(claimed_use, satellite_reality, confidence)

        # Formatting Risk Level
        risk_style = "green"
        if risk_level == "HIGH": risk_style = "bold red"
        elif risk_level == "CRITICAL": risk_style = "bold red blink"
        elif risk_level == "MEDIUM": risk_style = "yellow"
        
        sec_id = f"SEC-{random.randint(1000, 9999)}"
        table.add_row(
            sec_id,
            claimed_use,
            satellite_reality,
            f"[{risk_style}]{status_msg}[/{risk_style}]",
            token
        )
        
        results_for_export.append({
            "id": sec_id, 
            "claim": claimed_use, 
            "prediction": satellite_reality, 
            "risk": risk_level, 
            "token": token
        })

    console.print(table)
    generate_integration_files(results_for_export)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifyre Satellite Auditor")
    parser.add_argument("--demo", action="store_true", help="Run the simulated audit batch")
    parser.add_argument("--train", action="store_true", help="Train the model on the downloaded EuroSAT dataset")
    parser.add_argument("--image", type=str, help="Path to a specific image file to audit")
    parser.add_argument("--claim", type=str, default="Forest", choices=EUROSAT_CLASSES, help="The claimed land use for the single image audit")

    args = parser.parse_args()
    
    if args.demo:
        try:
            simulate_audit_batch(n=10) # Simulating 10 items for a good table
        except KeyboardInterrupt:
            print("\n[Aborted by user]")
        except Exception as e:
            from rich.console import Console
            c = Console()
            c.print_exception(show_locals=True)
    elif args.train:
        train_auditor()
    elif args.image:
        audit_single_image(args.image, args.claim)
    else:
        print("Run with --demo to trigger the simulation, --train to learn, or --image <path> --claim <type> for a specific file.")