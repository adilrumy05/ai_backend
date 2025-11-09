import argparse
import json
import sys
from pathlib import Path

# Import PyTorch and image processing libraries
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from torchvision.models import mobilenet_v2

# Load the mapping between class indices and species names
def load_class_map(path):
    with open(path, 'r') as f:
        mp = json.load(f) # Load the file into a Python dictionary
    # Convert all keys to integers (class indices) and ensure values are strings (species names)
    return {int(k): str(v) for k, v in mp.items()} 

# Load and preprocess an image for inference
def load_image(image_path):
    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    return tfm(img).unsqueeze(0)

# Command-line entry point for running inference
def main():
    # keep PyTorch lightweight on small servers
    torch.set_num_threads(1)

    # Create an argument parser to allow passing model, class map, and image paths
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True) # Path to trained model (.pth file)
    ap.add_argument('--classmap', required=True) # Path to JSON class mapping
    ap.add_argument('--image', required=True) # Path to input image
    ap.add_argument('--topk', type=int, default=5, help='Top-K classes to include')
    args = ap.parse_args() # Parse CLI arguments
    
    # Load class mapping
    class_map = load_class_map(args.classmap)
    num_classes = len(class_map)

    model = mobilenet_v2(weights=None) # Initialize a MobileNetV2 model (no pretrained weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes) # Replace the final classification layer with one matching the dataset class count

    # Load the trained model checkpoint from disk
    state = torch.load(args.model, map_location='cpu') # Load model parameters to CPU memory
    if isinstance(state, dict) and 'state_dict' in state: # Handle saved checkpoints with "state_dict"
        state = state['state_dict']
    # strip any "module." prefixes
    clean_state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=False)
    model.eval()

    # Preprocess the input image and run inference
    x = load_image(args.image)
    with torch.no_grad(): # Disable gradient calculations for inference
        logits = model(x) # Forward pass through the model
        probs = torch.softmax(logits, dim=1).squeeze(0) # Convert logits to probabilities

        conf, idx = torch.max(probs, dim=0) # Get the highest probability and its index
        idx = int(idx.item())
        conf = float(conf.item())  
        species = class_map.get(idx, 'unknown') # Map index to species name

        # Top-K (as percentages)
        k = max(1, min(args.topk, probs.numel())) # Ensure k <= number of classes
        topk_vals, topk_idxs = torch.topk(probs, k=k) # Retrieve top-k values and indices
        topk = [
            {
                "index": int(i.item()),
                "name": class_map.get(int(i.item()), "unknown"),
                "confidence": round(float(v.item()), 5)  
            }
            for v, i in zip(topk_vals, topk_idxs)
        ]
    
    # Output Results
    out = { # Prepare final structured JSON output
        "index": idx,
        "species_name": species,
        "confidence": round(conf, 5),
        "topk": topk
    }
    print("DEBUG:", idx, conf, species, file=sys.stderr) # Print debugging info to stderr (useful for backend logs)
    # Print debugging info to stderr (useful for backend logs)
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    sys.stdout.flush() # Ensure immediate delivery of output


if __name__ == '__main__':
    # To avoid accidental prints
    try:
        main()
    except Exception as e:
        # Print error to stderr so the backend can detect failure
        print(f"[infer.py error] {e}", file=sys.stderr)
        sys.exit(1)
