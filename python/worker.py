import os, sys, json
import torch, torch.nn as nn
from torchvision.models import mobilenet_v2
from PIL import Image
import torchvision.transforms as T

# Set default file paths for the model weights and class mapping
MODEL_PATH = os.getenv("MODEL_PATH", "./mobilenet_v2_SmartPlant.pth")
CLASSMAP_PATH = os.getenv("CLASS_MAP_PATH", "./class_mapping.json")

# Load the mapping of class indices to species names
def load_class_map(path):
    with open(path, "r") as f:
        mp = json.load(f)
    return {int(k): str(v) for k, v in mp.items()} # Convert all keys to integers and ensure values are strings

# Preprocess the input image for model inference
def load_image(image_path):
    tfm = T.Compose([
        T.Resize(256), # Resize image to 256 pixels on the shorter side
        T.CenterCrop(224), # Crop the center 224x224 pixels
        T.ToTensor(), # Convert image to tensor
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), # Normalize with ImageNet means and stds
    ])
    img = Image.open(image_path).convert("RGB") # Ensure image is in RGB format
    return tfm(img).unsqueeze(0) # Add batch dimension

class_map = load_class_map(CLASSMAP_PATH) # Read species labels from JSON fil
num_classes = len(class_map) 

model = mobilenet_v2(weights=None) # Initialize a MobileNetV2 model (no pretrained weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes) # Replace the final classification layer with one matching the dataset class count

# Load trained model weights from file
state = torch.load(MODEL_PATH, map_location="cpu") # Load model parameters to CPU memory
if isinstance(state, dict) and "state_dict" in state: 
    state = state["state_dict"]
state = {k.replace("module.", ""): v for k, v in state.items()} # Remove 'module.' prefix from keys
model.load_state_dict(state, strict=False)
model.eval()

# Optimize CPU thread usage
try:
    torch.set_num_threads(os.cpu_count() or 1) # Use available CPU cores for inference
except Exception:
    pass # Ignore if system does not support this setting

# Define inference function for one image
def infer(image_path, topk=5):
    with torch.no_grad(): # Disable gradient calculations for inference
        x = load_image(image_path) # Preprocess the input image
        logits = model(x) # Forward pass through the model
        probs = torch.softmax(logits, dim=1).squeeze(0) # Convert logits to probabilities

        conf, idx = torch.max(probs, dim=0) # Get the highest probability and its index
        conf = float(conf.item()) 
        idx = int(idx.item()) 

        # Get the Top-K predictions for better interpretability
        k = max(1, min(int(topk), probs.numel())) # Ensure k <= number of classes
        vals, idcs = torch.topk(probs, k=k) # Retrieve top-k values and indices
        top = [ # Build a list of top-k predictions with names and confidences
            {
                "index": int(i.item()),
                "name": class_map.get(int(i.item()), "unknown"),
                "confidence": float(v.item())  # 0..1
            }
            for v, i in zip(vals, idcs)
        ]
        # Return both the top-1 prediction and top-k list
        return {
            "index": idx,
            "species_name": class_map.get(idx, "unknown"),
            "confidence": round(conf, 5),  
            "topk": top
        }

# Worker loop: listen for JSON commands from Node.js (stdin/stdout)
# The Node backend sends one line per request:
# {"image": "path/to/file.jpg", "topk": 5}
for line in sys.stdin:
    line = line.strip() # Remove whitespace/newlines
    if not line: # Skip empty lines
        continue
    try:
        req = json.loads(line) # Parse JSON request
        out = infer(req["image"], req.get("topk", 5)) # Perform inference
        sys.stdout.write(json.dumps(out) + "\n") # Send JSON response
        sys.stdout.flush() # Ensure immediate delivery
    except Exception as e: # Handle any errors 
        sys.stderr.write(f"[worker error] {e}\n")
        sys.stderr.flush()
