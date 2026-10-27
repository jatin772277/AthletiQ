
from pathlib import Path

import cv2


# ============================================================
# ATHLETIQ: CALIBRATION FRAME EXTRACTION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "football"
    / "raw"
    / "videos"
    / "football2.mp4"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "calibration" / "frames"

# Extract a frame every 2 seconds.
FRAME_INTERVAL_SECONDS = 2.0

# Also save a frame near the end of the video.
SAVE_LAST_FRAME = True


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError(
            "Invalid video metadata: FPS or frame count is zero."
        )

    duration = frame_count / fps
    interval_frames = max(
        1, round(FRAME_INTERVAL_SECONDS * fps)
    )

    print("=" * 58)
    print("ATHLETIQ CALIBRATION FRAME EXTRACTION")
    print("=" * 58)
    print(f"Video: {VIDEO_PATH}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.3f}")
    print(f"Frames: {frame_count}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Extraction interval: {FRAME_INTERVAL_SECONDS} seconds")
    print(f"Output directory: {OUTPUT_DIR}")

    saved = 0
    frame_index = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if frame_index % interval_frames == 0:
            timestamp = frame_index / fps

            output_path = OUTPUT_DIR / (
                f"frame_{frame_index:06d}_"
                f"{timestamp:07.2f}s.jpg"
            )

            success = cv2.imwrite(
                str(output_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            if success:
                saved += 1
                print(
                    f"[SAVED] t={timestamp:6.2f}s  "
                    f"{output_path.name}"
                )

        frame_index += 1

    cap.release()

    # Ensure the last frame is represented even if it does not
    # fall on the regular extraction interval.
    if SAVE_LAST_FRAME and frame_count > 1:
        last_index = frame_count - 1
        cap = cv2.VideoCapture(str(VIDEO_PATH))
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_index)

        ok, frame = cap.read()
        cap.release()

        if ok:
            timestamp = last_index / fps
            output_path = OUTPUT_DIR / (
                f"frame_{last_index:06d}_"
                f"{timestamp:07.2f}s.jpg"
            )

            if not output_path.exists():
                if cv2.imwrite(
                    str(output_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                ):
                    saved += 1
                    print(f"[SAVED LAST] {output_path.name}")

    print("\nExtraction complete.")
    print(f"Frames saved in this run: {saved}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
