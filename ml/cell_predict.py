import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ml.model import DigitCNN

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load model (best_model.pth is in ml folder)
model = DigitCNN().to(device)
model.load_state_dict(torch.load('ml/best_model.pth'))
model.eval()

def predict_digit(image_path):
    """Predict single digit from image"""
    img = Image.open(image_path).convert('L')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()

def predict_grid(image_paths_9x9):
    """
    Predict 9x9 Sudoku grid
    image_paths_9x9: 9x9 numpy array of image file paths
    """
    predictions = np.zeros((9, 9), dtype=int)
    
    for i in range(9):
        for j in range(9):
            path = image_paths_9x9[i, j]
            if Path(path).exists():
                predictions[i, j] = predict_digit(path)
                print(f"({i},{j}): {predictions[i, j]}")
            else:
                print(f"({i},{j}): File not found")
                predictions[i, j] = 0
    
    return predictions

# Example usage
if __name__ == "__main__":
    # Auto-load cells from cell_export folder
    cell_export_path = Path('ml/cell_export')
    
    if cell_export_path.exists():
        # Create 9x9 array of cell image paths
        grid_paths = np.zeros((9, 9), dtype=object)
        
        for i in range(9):
            for j in range(9):
                cell_file = cell_export_path / f'cell_{i}_{j}.png'
                grid_paths[i, j] = str(cell_file)
        
        print("Predicting Sudoku grid from extracted cells...")
        sudoku_predictions = predict_grid(grid_paths)
        
        print("\n[OK] Sudoku Grid Predictions:")
        print(sudoku_predictions)
        
        # Optional: Save to file
        np.savetxt('sudoku_predictions.txt', sudoku_predictions, fmt='%d')
        print("\n[DONE] Saved to sudoku_predictions.txt")
    else:
        print(f"Cell export folder not found at {cell_export_path}")