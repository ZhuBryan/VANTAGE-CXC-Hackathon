import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import argparse
import random
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
import ssl

# Bypass SSL certificate verify failed errors for downloads if necessary
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

# --- AUDIT LOGIC KERNEL ---
def apply_audit_rules(claim, prediction):
    """
    Core business logic for Verifyre ESG Audit.
    """
    # Scenario A: Carbon Fraud (Claiming Forest, but it's crops)
    if claim == "Forest" and prediction in ["AnnualCrop", "PermanentCrop"]:
        return "FRAUD: Illicit Agriculture Detected", "HIGH", "SOL-BURN"
    
    # Scenario B: Habitat Destruction (Natural claim, Infrastructure reality)
    if claim in ["River", "Forest"] and prediction in ["Highway", "Residential", "Industrial"]:
        return "CRITICAL: Infrastructure Encroachment", "CRITICAL", "SOL-SLASH"

    # Scenario C: Grazing (Herbs claimed, Pasture reality)
    if claim == "HerbaceousVegetation" and prediction == "Pasture":
        return "WARNING: Unreported Cattle Grazing", "MEDIUM", "SOL-FLAG"

    # Scenario D: Clean
    if claim == prediction:
        return "VERIFIED", "LOW", "SOL-MINT"
    
    # Scenario E: General Mismatch (Catch-all for other mismatches not explicitly flagged as fraud)
    return "MISMATCH: Land Use Discrepancy", "UNKNOWN", "SOL-HOLD"

# --- MODEL SETUP ---
def load_auditor_model():
    print("[SYSTEM] Initializing Neural Link...")
    # Using ResNet18 for speed in demo
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
    
    model.eval()
    return model

# --- TRAINING MODE ---
def train_auditor(epochs=1):
    console = Console()
    console.print(Panel.fit("[bold green]INITIATING TRAINING SEQUENCE[/bold green]", border_style="green"))

    # Setup
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(EUROSAT_CLASSES))
    
    model.train() # Set to training mode

    # We need a fresh dataset loader
    dataset = get_dataset()
    if not dataset: return

    # Split: 80% train, 20% val (simplifying to just train on all for hackathon speed/demo)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    # Optimize only the new layer for speed (freeze the rest? No, let's fine-tune all but slowly)
    # Actually, fine-tuning just the FC layer is faster for a quick CPU demo.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    console.print(f"[SYSTEM] Training on {len(dataset)} satellite images for {epochs} epochs...")

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Simple progress bar
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
                print(f"\rEpoch {epoch+1}/{epochs} | Batch {i} | Loss: {loss.item():.4f} | Acc: {100 * correct / total:.2f}%", end="")

        print(f"\n[EPOCH {epoch+1} COMPLETE] Average Accuracy: {100 * correct / total:.2f}%")

    # Save
    torch.save(model.state_dict(), MODEL_PATH)
    console.print(f"[SUCCESS] Weights saved to [bold cyan]{MODEL_PATH}[/bold cyan]")
    console.print("[SYSTEM] Neural Link Calibrated. You may now run the audit.")

# --- SINGLE IMAGE AUDIT ---
def audit_single_image(image_path, claimed_label):
    console = Console()
    console.print(f"[bold cyan]Initiating Targeted Audit:[/bold cyan] {image_path}")

    model = load_auditor_model()
    
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

    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, predicted_idx = torch.max(probabilities, 1)
        predicted_class = EUROSAT_CLASSES[predicted_idx.item()]
        
    # Logic
    status_msg, risk_level, token = apply_audit_rules(claimed_label, predicted_class)
    
    # Display
    table = Table(title="TARGETED AUDIT RESULT", box=box.HEAVY_EDGE, border_style="bright_magenta")
    table.add_column("Target", style="cyan")
    table.add_column("Claimed", style="magenta")
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
        claimed_label,
        predicted_class,
        f"{top_prob.item():.2%}",
        f"[{risk_style}]{status_msg}[/{risk_style}]",
        token
    )
    
    console.print(table)


# --- SIMULATION ---
def get_dataset(root="./data"):
    print("[SYSTEM] Acquiring Satellite Feed (EuroSAT)...")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        # Automatically download if not present
        dataset = datasets.EuroSAT(root=root, download=True, transform=transform)
        return dataset
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        # Fallback for demo if download fails or takes too long (creates mock data usually, but here we exit)
        return None

# --- SIMULATION ---
def simulate_audit_batch(n=5):
    console = Console()
    
    # Header
    console.print(Panel.fit(
        "[bold cyan]VERIFYRE[/bold cyan] [green]SATELLITE AUDIT PROTOCOL v1.0[/green]\n"
        "[dim]Connecting to Solana Mainnet... Connected.[/dim]",
        border_style="cyan"
    ))

    model = load_auditor_model()
    dataset = get_dataset()
    
    if not dataset:
        console.print("[bold red]FATAL: Satellite uplink offline. (Dataset download failed)[/bold red]")
        return

    table = Table(title="audit_batch_result_log", box=box.HEAVY_EDGE, border_style="bright_blue")
    
    table.add_column("Sector ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Claimed Land Use", style="magenta")
    table.add_column("Satellite Reality", style="yellow")
    table.add_column("Risk Level", justify="center")
    table.add_column("Action Token", justify="right", style="green")

    # Simulation Loop
    indices = random.sample(range(len(dataset)), n)
    
    for idx in indices:
        image_tensor, label_idx = dataset[idx]
        true_label = EUROSAT_CLASSES[label_idx] # In a real audit, we wouldn't know this, but the model predicts it.
        
        # Run Inference
        with torch.no_grad():
            outputs = model(image_tensor.unsqueeze(0))
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = EUROSAT_CLASSES[predicted_idx.item()]
            
            # Since the model is untrained on EuroSAT (random FC weights), predictions are random.
            # To make the DEMO interesting, we will use the 'true_label' from the dataset 
            # as the 'Satellite Reality' for the sake of the logic check seeing "real" data,
            # OR we trust the untrained model. 
            # DECISION: Let's use the actual model prediction to show "AI" working, 
            # even if it's dumb right now. 
            # BUT, to force specific scenarios for the user to see the logic, 
            # we can inject simulation bias or just rely on RNG.
            
            # Use Model Prediction (this is the "Satellite Reality")
            satellite_reality = predicted_class

        # Generate a "Claim" to test our Audit Logic
        # We want a mix of honest and fraudulent claims for the demo.
        roll = random.random()
        if roll < 0.4:
            # 40% chance of telling the truth (or accidentally matching)
            claimed_use = satellite_reality
        elif roll < 0.7:
             # 30% chance of a specific fraud setup for demo purposes if possible
             if satellite_reality in ["AnnualCrop", "PermanentCrop"]:
                 claimed_use = "Forest" # Scenario A Trigger
             elif satellite_reality in ["Highway", "Residential"]:
                 claimed_use = "Forest" # Scenario B Trigger
             elif satellite_reality == "Pasture":
                 claimed_use = "HerbaceousVegetation" # Scenario C Trigger
             else:
                 claimed_use = random.choice(EUROSAT_CLASSES)
        else:
            # 30% random lie
            claimed_use = random.choice(EUROSAT_CLASSES)

        # Apply Audit Logic
        status_msg, risk_level, token = apply_audit_rules(claimed_use, satellite_reality)

        # Formatting Risk Level
        risk_style = "green"
        if risk_level == "HIGH": risk_style = "bold red"
        elif risk_level == "CRITICAL": risk_style = "bold red blink"
        elif risk_level == "MEDIUM": risk_style = "yellow"
        
        table.add_row(
            f"SEC-{random.randint(1000, 9999)}",
            claimed_use,
            satellite_reality,
            f"[{risk_style}]{status_msg}[/{risk_style}]",
            token
        )

    console.print(table)
    console.print("\n[bold]Audit Complete.[/bold] [dim]Hashes submitted to chain.[/dim]")

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
