import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# List to store points
points = []

def select_points(event, x, y, flags, param):
    """ Callback function to capture mouse click points. """
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)} selected: {x, y}")
        if len(points) == 4:
            cv2.destroyAllWindows()

def compute_homography(image_path):
    """ Function to compute and apply homography transformation. """
    global points
    
    # Load image
    image = cv2.imread(image_path)
    
    # Display image and collect 4 points
    cv2.imshow("Select 4 Points", image)
    cv2.setMouseCallback("Select 4 Points", select_points)
    cv2.waitKey(0)
    
    # Compute homography if 4 points are selected
    if len(points) == 4:
        src_pts = np.array(points, dtype=np.float32)

        width = int(max(np.linalg.norm(src_pts[0] - src_pts[1]), np.linalg.norm(src_pts[2] - src_pts[3])))
        height = int(max(np.linalg.norm(src_pts[0] - src_pts[3]), np.linalg.norm(src_pts[1] - src_pts[2])))
        
        dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        # Compute homography
        H, _ = cv2.findHomography(src_pts, dst_pts)

        # Warp the image using the homography matrix
        warped = cv2.warpPerspective(image, H, (width, height))
        
        # Show the result
        cv2.imshow("Warped Image", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return H, warped
    else:
        print("Error: Select exactly 4 points.")
        return None, None

img_path = filedialog.askopenfilename(title="Select image file")

H_matrix, warped_image = compute_homography(img_path)
cv2.imwrite("./homography_results/result.png", warped_image)
