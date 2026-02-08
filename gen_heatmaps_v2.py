import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# --- CONFIG ---
# Define the source images for each scenario
SCENARIOS = {
    "verified": {
        "source": "static/sector-1_view.jpg",
        "output": "static/heatmaps/real_verified_clean.jpg",
        "overlay_text": "PREDICTION: FOREST (1.00)",
        "color": "green", # Green border/text
        "colormap": cv2.COLORMAP_OCEAN # Just a placeholder, custom logic handles green
    },
    "industrial": {
        "source": "static/sector-4_view.jpg",
        "output": "static/heatmaps/real_fraud_clean.jpg",
        "overlay_text": "PREDICTION: DEFORESTATION (Industrial)",
        "color": "red",
        "colormap": cv2.COLORMAP_JET
    },
    "ghost": {
        "source": "static/sector-veridian-final.jpg", # NEW HIGH RES SOURCE FROM DATASET
        "output": "static/heatmaps/real_ghost_clean.jpg",
        "overlay_text": "PREDICTION: DEFORESTATION (Arid/Scrub)",
        "color": "red",
        "colormap": cv2.COLORMAP_JET
    }
}

def generate_cam(model, target_layer, image_path, save_path, overlay_text, color_theme="red", colormap=cv2.COLORMAP_JET):
    print(f"Processing {image_path}...")
    
    # 1. Load and Preprocess Image
    img = Image.open(image_path).convert('RGB')
    
    # transform for the model
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(img).unsqueeze(0)
    
    # 2. Hook into the final convolutional layer
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    
    handle = target_layer.register_forward_hook(hook_feature)
    
    # 3. Forward Pass
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        
    handle.remove()
    
    # 4. Generate CAM (Class Activation Map)
    features = features_blobs[0]
    bz, nc, h, w = features.shape
    params = list(model.fc.parameters())
    weight_softmax = np.squeeze(params[0].data.cpu().numpy())
    idx = np.argmax(probs.data.cpu().numpy())
    cam = weight_softmax[idx].dot(features.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    
    # Resize CAM to original
    orig_w, orig_h = img.size
    cam = cv2.resize(cam, (orig_w, orig_h))
    
    # 5. Create Heatmap
    if color_theme == "green":
        # Custom Green Map for Verified: Green = High Activation, Transparent/Blue = Low
        # We can simulate this by using a specific colormap or creating a constant color map
        # Let's use SUMMER (Green to Yellow). 
        # Or better: Create a monochromatic Green map.
        # Create an empty image with green channel = cam
        zeros = np.zeros_like(cam)
        # BGR
        heatmap = cv2.merge([zeros, cam, zeros]) # Pure Green map
        # Amplify
        heatmap = heatmap * 1.5
        heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
    else:
        heatmap = cv2.applyColorMap(cam, colormap)
        # Convert BGR to RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img_np = np.array(img)
    result = heatmap * 0.4 + img_np * 0.6
    result = result.astype(np.uint8)
    
    # 6. Draw UI
    # REMOVED: User requested no labels/borders
    
    Image.fromarray(result).save(save_path, quality=95)
    print(f"Saved {save_path}")

def main():
    print("Loading AI Model (Fine-Tuned EuroSAT)...")
    # Load ResNet18 and adapt final layer to 11 classes (as per verifyre_auditor.py)
    model = models.resnet18(weights=None) # Start blank
    num_ftrs = model.fc.in_features
    # 11 classes: "AnnualCrop", "Forest", ..., "Deforestation"
    model.fc = torch.nn.Linear(num_ftrs, 11) 
    
    # Load weights
    if os.path.exists("auditor_weights.pth"):
        model.load_state_dict(torch.load("auditor_weights.pth", map_location=torch.device('cpu')))
        print("Loaded auditor_weights.pth")
    else:
        print("WARNING: auditor_weights.pth not found, using random weights (Bad Heatmaps!)")
        
    target_layer = model.layer4[1].conv2
    
    for _, config in SCENARIOS.items():
        if os.path.exists(config["source"]):
            generate_cam(model, target_layer, config["source"], config["output"], config["overlay_text"], config["color"])
        else:
            print(f"Source missing: {config['source']}")

if __name__ == "__main__":
    main()
