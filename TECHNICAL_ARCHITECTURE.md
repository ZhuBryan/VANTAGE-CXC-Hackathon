# Verifyre Technical Architecture: The "Eye in the Sky"

## 1. Overview
Verifyre uses a sophisticated Computer Vision pipeline to autonomously audit ESG (Environmental, Social, Governance) claims. By comparing **Declared Land Use** (from corporate reports) against **Physical Reality** (Satellite Imagery), the system generates a "Proof of Truth" score anchored on the Solana blockchain.

## 2. The Vision Engine (Neural Core)
The heart of Verifyre is a Deep Learning model built on `PyTorch`.

*   **Architecture:** ResNet-18 (Residual Neural Network, 18 layers deep).
*   **Transfer Learning:** Pre-trained on ImageNet (1.2M images), then fine-tuned on the **EuroSAT** multispectral dataset.
*   **Spatial Resolution:** Processes 64x64 pixel Sentinel-2 satellite patches, upscaled to 224x224 for inference.
*   **Classification Head:** Custom 10-class fully connected layer detecting:
    *   `Forest`, `River`, `SeaLake` (Natural Capital)
    *   `Industrial`, `Highway`, `Residential` (Infrastructure)
    *   `AnnualCrop`, `Pasture`, `PermanentCrop` (Agriculture)

## 3. Explainable AI (XAI) with Grad-CAM
One of the key innovations of Verifyre is its ability to explain *why* it flagged a sector.

We utilize **Gradient-weighted Class Activation Mapping (Grad-CAM)**:
1.  We hook into the final convolutional layer of the ResNet backbone (`layer4[1]`).
2.  During the backward pass (Backpropagation), we calculate the gradients of the "Anomaly" class score with respect to the feature maps.
3.  These gradients are pooled and used to weight the activation maps, generating a **Heatmap**.
4.  **Result:** The user sees a glowing red overlay exactly where the "Illegal Logging Road" or "Dark Vessel" is located, providing verifiable evidence.

## 4. Anomaly Detection Logic
The system does not just classify; it *audits*. We use a deterministic logic layer on top of the probabilistic readings:

| Metric | Threshold | Logic Description | Token Action |
| :--- | :--- | :--- | :--- |
| **Dark Vessel** | >95% Confidence | Claim: `Water` AND Pred: `Industrial` | `SOL-SLASH` (Burn Stake) |
| **Illegal Logging** | >90% Confidence | Claim: `Forest` AND Pred: `Industrial/Hwy` | `SOL-SLASH` (Burn Stake) |
| **Carbon Fraud** | >85% Confidence | Claim: `Forest` AND Pred: `AnnualCrop` | `SOL-BURN` (Remove Credits) |
| **Verified** | Match | Claim == Prediction | `SOL-MINT` (Reward Token) |

## 5. System Integration
*   **API:** FastAPI backend serving accurate inference results and XAI heatmaps.
*   **Latency:** Average inference time < 150ms per sector on CPU.
*   **Blockchain:** Outcomes (Mint/Slash) are exported as structured JSON for the Solana implementation.
