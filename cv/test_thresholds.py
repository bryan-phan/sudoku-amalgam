import cv2
import os
import numpy as np
from scan import Scan

# Test different threshold values
thresholds_to_test = [140, 145, 150, 155, 160, 165, 170]

original_path = os.getcwd() + r"\cv" + r"\test_imgs" + r"\angled.jpg"
original = cv2.imread(original_path)
original = cv2.resize(original, None, fx=0.5, fy=0.4)

# Create a Scan object to get warped image
scan = Scan()
warped = scan.warped

height, width = warped.shape[:2]
cell_h = height // 9
cell_w = width // 9

print("Testing different threshold values...\n")

for thresh_val in thresholds_to_test:
    print(f"\n{'='*60}")
    print(f"Threshold: {thresh_val}")
    print(f"{'='*60}")
    
    cells = []
    for r in range(9):
        row_cells = []
        for c in range(9):
            y1 = r * cell_h
            y2 = (r + 1) * cell_h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w
            
            cell = warped[y1:y2, x1:x2]
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            
            small = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
            row_cells.append(small)
            
            # Save sample cells for inspection
            if r == 0 and c == 0:
                cv2.imwrite(f'test_cell_{thresh_val}.png', small)
        
        cells.append(row_cells)
    
    print(f"Sample cells saved as test_cell_{thresh_val}.png")

print("\n✓ Test complete! Check which threshold value works best.")
print("Then update cv/scan.py with the best threshold value.")
