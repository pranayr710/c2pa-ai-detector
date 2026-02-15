import os

folder_path = "/Users/towfeeqrameez/Documents/pranay/image_verifier/dataset/ai"

files = os.listdir(folder_path)

# Sort to keep order consistent
files.sort()

count = 1
for filename in files:
    old_path = os.path.join(folder_path, filename)

    # Skip non-image files
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    new_name = f"ai_{count}.jpg"
    new_path = os.path.join(folder_path, new_name)

    os.rename(old_path, new_path)
    count += 1

print("✅ Renaming completed.")
