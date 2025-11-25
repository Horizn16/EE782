import cv2
import os
import glob

videos = glob.glob("*.mp4")         # all videos in current folder
output_root = "real_world"

for video_path in videos:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    img_dir = os.path.join(output_root, video_name, "images")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:     # safety guard for corrupted metadata
        fps = 25
    interval = int(fps)  # one frame per second

    frame_index = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # save 1 frame every second
        if frame_index % interval == 0:
            filename = f"frame_{saved:05d}.jpg"
            cv2.imwrite(os.path.join(img_dir, filename), frame)
            saved += 1
        frame_index += 1

    cap.release()
    print(f"{video_name}: saved {saved} frames to {img_dir}")
