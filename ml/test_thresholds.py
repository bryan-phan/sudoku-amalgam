import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
import sys
sys.path.insert(0, '..')
from ml.model import DigitCNN

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DigitCNN().to(device)
model.load_state_dict(torch.load('ml/best_model.pth'))
model.eval()

base_transform = transforms.Compose([
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict_with_threshold(image_path, threshold=127):
    """Predict digit with specific threshold"""
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # Check if empty
    black_pixels = np.sum(img_array < 50)
    black_ratio = black_pixels / img_array.size
    if black_ratio > 0.9:
        return 0
    
    # Apply threshold and predict
    _, thresh = cv2.threshold(img_array, threshold, 255, cv2.cv2.THRESH_BINARY_INV)
    img_tensor = transforms.ToTensor()(Image.fromarray(thresh)).unsqueeze(0).to(device)
    img_tensor = base_transform(img_tensor)
    
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()

# Test different thresholds
cell_export_path = Path('ml/cell_export')
thresholds = [80, 100, 120, 140, 160, 180, 200]

print("Testing different thresholds on same cells...\n")

for thresh in thresholds:
    print(f"\nThreshold: {thresh}")
    print("-" * 40)
    grid = np.zeros((9, 9), dtype=int)
    
    for i in range(9):
        for j in range(9):
            cell_file = cell_export_path / f'cell_{i}_{j}.png'
            try:
                grid[i, j] = predict_with_threshold(str(cell_file), thresh)
            except:
                grid[i, j] = 0
    
    print(grid)
