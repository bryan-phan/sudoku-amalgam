import cv2
import os
import numpy as np
from scan import Scan

# Get warped image from Scan
scan = Scan()
warped = scan.warped

height, width = warped.shape[:2]
cell_h = height // 9
cell_w = width // 9

# Test different thresholds
thresholds = [80, 100, 110, 120, 140, 160, 180, 200]

print("Testing different threshold values...\n")

for thresh_val in thresholds:
    print(f"\nThreshold: {thresh_val}")
    print("-" * 50)
    
    # Save cells with this threshold
    output_dir = os.path.join(os.getcwd(), "ml", "cell_export")
    
    for r in range(9):
        for c in range(9):
            y1 = r * cell_h
            y2 = (r + 1) * cell_h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w
            
            cell = warped[y1:y2, x1:x2]
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            
            # Resize to 28x28
            small = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
            
            file_name = f"cell_{r}_{c}_thresh{thresh_val}.png"
            save_path = os.path.join(output_dir, file_name)
            cv2.imwrite(save_path, small)
    
    print(f"Cells extracted with threshold {thresh_val}")

print("\n✓ All thresholds tested! Sample cells saved.")
print("Check which threshold looks best visually, then use that value in scan.py")
