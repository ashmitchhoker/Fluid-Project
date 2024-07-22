import cv2
import numpy as np

frame = cv2.imread('IMG_5515.PNG')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow('Frame', frame)



# Detect edges using Canny edge detector
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)



# Find the contour with the maximum area (presumably the tube)
if len(contours) > 0:
    max_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(max_contour)
    # Draw the bounding rectangle
    start_point = (467, 455-60)
    end_point = (588, 455-61)


    # Define the color of the line in BGR format (Blue, Green, Red)
    color = (255, 0, 0)  # Green color in this example
    pwid = ((end_point[1]-start_point[1])**2+(end_point[0]-start_point[0])**2)**0.5
    mwid = 0.05945
    print("Thickness (pixels):",pwid)
    print("Thickness (m/pixel):",mwid/pwid)
    # Draw the line on the image
    thickness = 2  # You can adjust the thickness of the line as needed
    cv2.line(frame, start_point, end_point, color, thickness)

    # Calculate the width of the tube
    width = w

    # Display the width on the frame
    cv2.putText(frame, f"Width: {width}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
