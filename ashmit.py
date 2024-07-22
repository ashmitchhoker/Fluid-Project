import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from statistics import mean
from scipy.stats import gaussian_kde


def remove_edges(image, edge_threshold):
    edges = cv2.Canny(image, 50, 200)
    edges = cv2.threshold(edges, edge_threshold, 255, cv2.THRESH_BINARY)[1]
    result = cv2.bitwise_and(image, cv2.bitwise_not(edges))
    return result


cap = cv2.VideoCapture('castor_3mm.MP4')
cap.read()
fps = cap.get(cv2.CAP_PROP_FPS)

print("FPS:", fps)
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=False)
crop_size = 200
edge_removal_threshold = 200

fgmask_frames = []
prev_centroid = None
frame_count = 0
t = []
v = []
d = []
time_difference = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break
    fgmask = fgbg.apply(frame)

    fgmask_no_edges = remove_edges(fgmask, edge_removal_threshold)

    fgmask_frames.append(fgmask_no_edges)

    cv2.imshow('fg', fgmask_no_edges)

    _, binary_mask = cv2.threshold(fgmask_no_edges, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea, default=None)

    if largest_contour is not None:
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

            if frame_count % 4 == 0:
                if prev_centroid is not None: 

                    displacement_vector = np.array(
                        [cx, cy]) - np.array(prev_centroid)
                    time_difference = time_difference + 4/50
                    velocity = (
                        ((displacement_vector[0])**2+(displacement_vector[1])**2)**(1/2)) / (4/50)
                    t.append(time_difference)
                    d.append(
                        ((displacement_vector[0])**2+(displacement_vector[1])**2)**(1/2))
                    v.append(velocity)

                    print("Velocity:", velocity)

                prev_centroid = np.array([cx, cy])

        cv2.imshow('Velocity Estimation', frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    frame_count += 1
print(len(v))
v_new = []
t_new = []
d_new = []
pixel_to_m= 0.0004913055359214876
for i in range(len(v)-1):
    if v[i] > 110 and v[i] < 140:
        # converting into m/s from pixels/s
        v_new.append(v[i]*pixel_to_m)
        t_new.append(t[i])
        # converting into m from pixels
        d_new.append(d[i]*pixel_to_m)

        
# print(v_new)
# print(t_new)
# print(d_new)

# Define a function to compute moving averages
def moving_average(data, window_size):
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size


# Smooth the velocity data using a moving average
window_size = 25 # Adjust this parameter to change the smoothness
smoothed_velocity = moving_average(v_new, window_size)

# Plotting velocity versus time
plt.plot(t_new[window_size-1:], smoothed_velocity)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity v/s Time [Castor Oil, 3 mm ball]')
plt.show()

# cap.release()
# cv2.destroyAllWindows()


# Find stabilization point automatically
def find_stabilization_point(velocity_data, threshold=0.01, window_size=50):
    """
    Find the stabilization point in the velocity data.

    Parameters:
        velocity_data (array): Array containing velocity data.
        threshold (float): Threshold for considering a change in velocity as negligible.
        window_size (int): Size of the window for smoothing the velocity data.

    Returns:
        stabilization_point (int): Index of the stabilization point in the velocity data.
    """
    velocity_changes = np.diff(velocity_data)
    stabilization_point = None

    for i, change in enumerate(velocity_changes):
        if abs(change) < threshold:
            stabilization_point = i + window_size // 2
            break

    return stabilization_point

stabilization_point = find_stabilization_point(v_new,window_size)

if stabilization_point is not None:
    terminal_velocity = np.mean(v_new[stabilization_point:])
    print("Stabilization Point Index:", stabilization_point)
    print("Terminal velocity:", terminal_velocity, "m/s")
else:
    print("Terminal velocity could not be determined.")

