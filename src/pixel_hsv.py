import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the BGR pixel value
        bgr_pixel = frame[y, x]
        
        # Convert to HSV
        hsv_pixel = cv2.cvtColor(np.uint8([[bgr_pixel]]), cv2.COLOR_BGR2HSV)[0][0]
        
        print(f"\nPixel at ({x}, {y})")
        print(f"BGR: {bgr_pixel}")
        print(f"HSV: {hsv_pixel}")
        
        # Draw a marker and show values on the image
        cv2.drawMarker(frame, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        text = f"HSV: {hsv_pixel}"
        cv2.putText(frame, text, (x + 25, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)

# Initialize video capture (replace with your video path)
cap = cv2.VideoCapture("war_match_2.mp4")
cv2.namedWindow("HSV Picker")
cv2.setMouseCallback("HSV Picker", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow("HSV Picker", frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()