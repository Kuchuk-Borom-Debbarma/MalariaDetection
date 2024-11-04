import os
import shutil


def move_files_by_extension(source_dir, destination_dir, extension):
    # Ensure destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Walk through all subdirectories in the source directory
    for root, _, files in os.walk(source_dir):
        for file in files:
            # Check if file matches the specified extension
            if file.lower().endswith(f".{extension.lower()}"):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_dir, file)

                # Move the file
                print(f"Moving {source_path} to {destination_path}")
                shutil.move(source_path, destination_path)

    print("All files moved successfully.")


if __name__ == "__main__":
    # Get user inputs
    source_dir = input("Enter the source directory path: ")
    destination_dir = input("Enter the destination directory path: ")
    extension = input("Enter the file extension (e.g., 'png'): ")

    # Move files
    move_files_by_extension(source_dir, destination_dir, extension)
