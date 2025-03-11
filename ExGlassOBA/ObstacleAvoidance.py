import cv2
import numpy as np

def detect_edges(frame):
    """Convert frame to grayscale, blur it, and perform Canny edge detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def region_of_interest(edges, vertices):
    """Mask the edges image to only keep the ROI defined by the vertices."""
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(edges, mask)

def detect_obstacle(frame):
    """
    Detect obstacles in the frame:
    - Define a ROI (for example, the bottom-center part of the image).
    - Calculate the edge density and optionally count lines using Hough Transform.
    - Overlay ROI, debug info, and detection alert on the frame.
    """
    height, width = frame.shape[:2]
    # Define a ROI: bottom central rectangle (you can adjust these values)
    roi_vertices = np.array([[
        (int(width * 0.3), int(height * 0.8)),
        (int(width * 0.7), int(height * 0.8)),
        (int(width * 0.7), int(height * 0.5)),
        (int(width * 0.3), int(height * 0.5))
    ]], dtype=np.int32)
    
    # Get edges and apply ROI
    edges = detect_edges(frame)
    roi_edges = region_of_interest(edges, roi_vertices)
    
    # Calculate edge density in the ROI
    edge_pixels = cv2.countNonZero(roi_edges)
    total_pixels = roi_edges.size
    edge_density = edge_pixels / total_pixels

    # Use Hough Transform to detect lines in the ROI
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=5)
    line_count = len(lines) if lines is not None else 0

    # Draw ROI for debugging
    cv2.polylines(frame, roi_vertices, isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Overlay debug text with edge density and line count
    cv2.putText(frame, f"Edge Density: {edge_density:.4f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"Line Count: {line_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Determine if obstacle is detected based on thresholds.
    # Adjust these thresholds based on calibration in your environment.
    obstacle_detected = False
    if edge_density > 0.005 or line_count > 15:
        cv2.putText(frame, "Obstacle Detected!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        obstacle_detected = True

    return obstacle_detected, frame

# Open the MJPEG stream from your ESP32-CAM
cap = cv2.VideoCapture("http://192.168.1.55/stream")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to retrieve frame.")
        break

    # Run obstacle detection on the frame
    obstacle, processed_frame = detect_obstacle(frame)

    # Optionally, if you want to visualize the raw edges, you can do:
    edges = detect_edges(frame)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(processed_frame, 0.8, edges_colored, 0.2, 0)
    cv2.imshow("Obstacle Detection", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
