import cv2
import numpy as np
import os
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

        # Compute new width and height correctly
        width_A = np.linalg.norm(src_pts[0] - src_pts[1])
        width_B = np.linalg.norm(src_pts[2] - src_pts[3])
        height_A = np.linalg.norm(src_pts[0] - src_pts[3])
        height_B = np.linalg.norm(src_pts[1] - src_pts[2])

        width = int(max(width_A, width_B))
        height = int(max(height_A, height_B))

        # Define destination points (rectangle)
        dst_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)

        # Compute homography
        H, _ = cv2.findHomography(src_pts, dst_pts)

        # Warp the image using the homography matrix
        warped = cv2.warpPerspective(image, H, (width, height))
        
        # Show the result with the same size as the warped image
        window_name = "Warped Image"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Make window resizable
        cv2.resizeWindow(window_name, width, height)  # Match window size to image
        cv2.imshow(window_name, warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return H, warped
    else:
        print("Error: Select exactly 4 points.")
        return None, None

# Open file dialog to select image
img_path = filedialog.askopenfilename(title="Select image file")

# Compute homography
H_matrix, warped_image = compute_homography(img_path)

if warped_image is not None:
    # Ensure output directory exists
    output_dir = "./homography_results"
    os.makedirs(output_dir, exist_ok=True)

    # Save the result
    output_path = os.path.join(output_dir, "result.png")
    cv2.imwrite(output_path, warped_image)
    print(f"Saved warped image to {output_path}")
