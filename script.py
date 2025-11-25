import os
import glob

videos = glob.glob("*.mp4")   # pick all mp4 in current folder
output_root = "real_world"    # main dataset folder
os.makedirs(output_root, exist_ok=True)

for video in videos:
    video_name = os.path.splitext(os.path.basename(video))[0]
    img_dir = os.path.join(output_root, video_name, "images")
    os.makedirs(img_dir, exist_ok=True)

print("Done creating folders.")
