import os

# Path to your real images folder
folder_path = "/Users/towfeeqrameez/Documents/pranay/image_verifier/dataset/real"

# Get all files
files = os.listdir(folder_path)

# Sort files so order is consistent
files.sort()

count = 1
for filename in files:
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        old_path = os.path.join(folder_path, filename)
        new_name = f"real{count}.jpg"
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)
        count += 1

print("✅ All real images renamed successfully!")
