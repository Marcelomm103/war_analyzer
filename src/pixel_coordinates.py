import cv2

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Print coordinates to terminal
        print(f"Clicked at (x, y): ({x}, {y})")
        
        # Draw a crosshair at clicked point
        display_frame = frame.copy()
        cv2.drawMarker(display_frame, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        
        # Display coordinates on image
        coord_text = f"({x}, {y})"
        cv2.putText(display_frame, coord_text, (x + 25, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Image", display_frame)

# Replace with your image file path
image_path = "war_reference_ss_1.png"  # Change this to your image file
frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# Print image shape to terminal
height, width, channels = frame.shape
print(f"Image shape: Height={height}, Width={width}, Channels={channels}")

# Create window and set callback
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Image", mouse_callback)

# Display the image
cv2.imshow("Image", frame)
print("Click on the image to get pixel coordinates. Press 'q' to quit.")

# Wait for key press to exit
while True:
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()