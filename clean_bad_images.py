import glob
import os
import shutil
import tensorflow as tf

files = (
    glob.glob("data/raw/fractured/*")
    + glob.glob("data/raw/not_fractured/*")
)

bad_dir = "data/raw/bad_images"
os.makedirs(bad_dir, exist_ok=True)

bad = []

for filepath in files:
    if not filepath.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    try:
        data = tf.io.read_file(filepath)

        if filepath.lower().endswith((".jpg", ".jpeg")):
            tf.io.decode_jpeg(data, channels=3)
        else:
            tf.io.decode_png(data, channels=3)

    except Exception:
        bad.append(filepath)

for filepath in bad:
    destination = os.path.join(bad_dir, os.path.basename(filepath))
    shutil.move(filepath, destination)

print("Bad images found:", len(bad))
print("Moved to:", bad_dir)

for filepath in bad:
    print(filepath)