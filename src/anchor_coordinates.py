import cv2
import numpy as np

# Load your full reference screenshot (where you defined original territory coords)
reference_image_path = "war_reference_ss_1.png"
reference_img = cv2.imread(reference_image_path, cv2.IMREAD_COLOR)
if reference_img is None:
    print(f"Error: Could not load reference image from {reference_image_path}")
    exit()

reference_img_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)

# Load your anchor template
anchor_template_path = "anchor_template.png" # The template you'll use for runtime detection
anchor_template_img = cv2.imread(anchor_template_path, cv2.IMREAD_GRAYSCALE) # Load as grayscale if matching grayscale
if anchor_template_img is None:
    print(f"Error: Could not load anchor template from {anchor_template_path}")
    exit()

# Perform template matching
result = cv2.matchTemplate(reference_img_gray, anchor_template_img, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# max_loc is the top-left coordinate of the best match in the reference image
reference_anchor_x, reference_anchor_y = max_loc

print(f"In your reference screenshot ({reference_image_path}),")
print(f"the anchor template ({anchor_template_path}) was found with its top-left at: ({reference_anchor_x}, {reference_anchor_y})")
print(f"Confidence of this match: {max_val:.4f}")
print(f"Use this as your REFERENCE_ANCHOR_POSITION: ({reference_anchor_x}, {reference_anchor_y})")

# Optional: Draw a rectangle to verify
ah, aw = anchor_template_img.shape[:2]
cv2.rectangle(reference_img, (reference_anchor_x, reference_anchor_y), 
              (reference_anchor_x + aw, reference_anchor_y + ah), (0, 0, 255), 2)
cv2.imshow("Reference with Anchor Found", reference_img)
cv2.waitKey(0)
cv2.destroyAllWindows()