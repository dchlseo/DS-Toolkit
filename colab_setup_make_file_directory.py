import glob
import os

# Original directory of the images
image_directory = "./photos/"
# New directory for the output paths
new_directory = "/content/drive/MyDrive/Colab Notebooks/yolo/darknet/darknet_test/data/images/"
# Extension of the files
extension = "*.jpeg"
# File for saving the output paths
save_at = "./train.txt"

# Get a list of file paths from the original directory
original_files = sorted(glob.glob(image_directory + extension))

# Open train.txt file in write mode
with open(save_at, 'w') as f:
    for file_path in original_files:
        # Extract just the filename
        filename = os.path.basename(file_path)
        # Create the new file path
        new_file_path = os.path.join(new_directory, filename)
        # Write the new file path to train.txt
        f.write(new_file_path + '\n')
        # Print the new file path
        print(new_file_path)

