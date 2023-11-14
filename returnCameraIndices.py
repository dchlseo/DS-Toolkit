import cv2

def returnCameraIndices():
    """
    Scans the first 10 camera indices to check which ones are available.

    This function attempts to open a video capture with each index using OpenCV.
    If the capture is successful, it means a camera is available at that index.

    Returns:
        arr (list of int): A list containing the indices of available cameras.
    """
    # Initialize variables
    index = 0  # Start checking from index 0
    arr = []   # List to store indices of available cameras
    i = 10     # Number of indices to check

    # Check each index to see if a camera is available
    while i > 0:
        cap = cv2.VideoCapture(index)  # Attempt to open video capture at current index
        if cap.read()[0]:              # If capture is successful, add index to the list
            arr.append(index)
            cap.release()              # Release the capture object
        index += 1                     # Move to the next index
        i -= 1                         # Decrement the counter

    return arr  # Return the list of available camera indices

# Example usage
camera_indices = returnCameraIndices()
print("Available camera indices:", camera_indices)
