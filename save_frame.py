import cv2
import os
import numpy as np

def save_frames_horizontal(video_path, output_dir, max_width=None, max_height=None, group_size=4):
    """
    Extract frames from a video and save them in groups stacked horizontally.
    
    Args:
        video_path (str): Path to video.
        output_dir (str): Output folder.
        max_width (int, optional): Max width of each frame.
        max_height (int, optional): Max height of each frame.
        group_size (int): Number of frames per horizontal image.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_buffer = []
    group_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize keeping aspect ratio
        if max_width or max_height:
            h, w = frame.shape[:2]
            scale_w = max_width / w if max_width else 1.0
            scale_h = max_height / h if max_height else 1.0
            scale = min(scale_w, scale_h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        frame_buffer.append(frame)

        # When enough frames for one horizontal group
        if len(frame_buffer) == group_size:
            # Resize to the smallest height among frames
            h_min = min(f.shape[0] for f in frame_buffer)
            resized = [cv2.resize(f, (int(f.shape[1]*h_min/f.shape[0]), h_min)) for f in frame_buffer]

            horizontal_img = np.hstack(resized)
            filename = os.path.join(output_dir, f"group_{group_count:05d}.jpg")
            cv2.imwrite(filename, horizontal_img)
            print(f"Saved {filename}")

            group_count += 1
            frame_buffer = []

    cap.release()

    # Handle leftover frames
    if frame_buffer:
        h_min = min(f.shape[0] for f in frame_buffer)
        resized = [cv2.resize(f, (int(f.shape[1]*h_min/f.shape[0]), h_min)) for f in frame_buffer]
        # Fill missing frames with black images
        while len(resized) < group_size:
            w = resized[0].shape[1]
            resized.append(np.zeros((h_min, w, 3), dtype=np.uint8))
        horizontal_img = np.hstack(resized)
        filename = os.path.join(output_dir, f"group_{group_count:05d}.jpg")
        cv2.imwrite(filename, horizontal_img)
        print(f"Saved {filename}")

    print(f"Done! Saved {group_count+1} horizontal images to '{output_dir}'.")


if __name__ == "__main__":
    video_file = "../Kaggle/20240912_101331.mp4"
    output_folder = "horizontal_frames"

    # Example: resize each frame to max width 160
    save_frames_horizontal(video_file, output_folder, max_width=160, group_size=4)
