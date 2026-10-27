
from pathlib import Path

import cv2
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = ROOT / "datasets/football/raw/videos/football2.mp4"
TRACKS_PATH = ROOT / "outputs/joint_tracking/football2_player_tracks.csv"

OUTPUT_DIR = ROOT / "datasets/football/team_identification"
CROPS_DIR = OUTPUT_DIR / "crops"
MANIFEST_PATH = OUTPUT_DIR / "team_crop_manifest.csv"
CONTACT_SHEET_PATH = OUTPUT_DIR / "team_crop_contact_sheet.jpg"

MAX_CROPS_PER_PLAYER = 20
MIN_PLAYER_WIDTH = 16
MIN_PLAYER_HEIGHT = 36
MIN_TORSO_HEIGHT = 16
JPEG_QUALITY = 95


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    if not TRACKS_PATH.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {TRACKS_PATH}")

    tracks = pd.read_csv(TRACKS_PATH)

    required = {"frame_id", "canonical_id", "x1", "y1", "x2", "y2"}
    missing = required - set(tracks.columns)

    if missing:
        raise ValueError(f"Missing tracking columns: {sorted(missing)}")

    tracks = tracks.dropna(subset=list(required)).copy()
    tracks["frame_id"] = tracks["frame_id"].astype(int)
    tracks["canonical_id"] = tracks["canonical_id"].astype(int)
    tracks = tracks.sort_values(["canonical_id", "frame_id"])

    # Select evenly spaced observations for each temporary tracking ID.
    selected_groups = []

    for _, group in tracks.groupby("canonical_id"):
        group = group.drop_duplicates("frame_id")

        count = min(MAX_CROPS_PER_PLAYER, len(group))

        if count == 0:
            continue

        indices = [
            round(i * (len(group) - 1) / max(count - 1, 1))
            for i in range(count)
        ]

        selected_groups.append(group.iloc[indices])

    if not selected_groups:
        raise RuntimeError("No usable player observations were found.")

    selected = pd.concat(selected_groups).sort_values("frame_id")

    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    manifest = []
    cache_frame_id = None
    cached_frame = None
    skipped = 0

    for row in selected.itertuples(index=False):
        frame_id = int(row.frame_id)
        player_id = int(row.canonical_id)

        if not 0 <= frame_id < frame_count:
            skipped += 1
            continue

        if frame_id != cache_frame_id:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, cached_frame = cap.read()
            cache_frame_id = frame_id

            if not ok:
                cached_frame = None

        if cached_frame is None:
            skipped += 1
            continue

        x1 = max(0, min(frame_width, int(row.x1)))
        y1 = max(0, min(frame_height, int(row.y1)))
        x2 = max(0, min(frame_width, int(row.x2)))
        y2 = max(0, min(frame_height, int(row.y2)))

        width = x2 - x1
        height = y2 - y1

        if width < MIN_PLAYER_WIDTH or height < MIN_PLAYER_HEIGHT:
            skipped += 1
            continue

        # Crop the upper body, including the jersey and shoulders.
        torso_y2 = min(y2, y1 + round(height * 0.65))
        torso_height = torso_y2 - y1

        if torso_height < MIN_TORSO_HEIGHT:
            skipped += 1
            continue

        crop = cached_frame[y1:torso_y2, x1:x2]

        if crop.size == 0:
            skipped += 1
            continue

        filename = f"player_{player_id:03d}_frame_{frame_id:05d}.jpg"
        output_path = CROPS_DIR / filename

        if not cv2.imwrite(
            str(output_path),
            crop,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        ):
            skipped += 1
            continue

        manifest.append({
            "image_path": str(output_path.relative_to(ROOT)),
            "frame_id": frame_id,
            "canonical_id": player_id,
            "team_label": "",
            "notes": "",
        })

    cap.release()

    result = pd.DataFrame(manifest)
    result.to_csv(MANIFEST_PATH, index=False)

    if result.empty:
        raise RuntimeError(
            "No crops passed the quality filters. "
            "Review the minimum size thresholds."
        )

    # Build a labeled contact sheet for quick visual inspection.
    thumb_w, thumb_h = 120, 110
    label_h = 30
    columns = 8
    rows = (len(result) + columns - 1) // columns

    sheet = 255 * __import__("numpy").ones(
        (rows * (thumb_h + label_h), columns * thumb_w, 3),
        dtype="uint8",
    )

    for i, item in enumerate(result.itertuples(index=False)):
        image_path = ROOT / item.image_path
        crop = cv2.imread(str(image_path))

        if crop is None:
            continue

        thumb = cv2.resize(crop, (thumb_w, thumb_h))
        r, c = divmod(i, columns)
        y = r * (thumb_h + label_h)
        x = c * thumb_w

        sheet[y:y + thumb_h, x:x + thumb_w] = thumb

        cv2.putText(
            sheet,
            f"ID {item.canonical_id} F{item.frame_id}",
            (x + 3, y + thumb_h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(CONTACT_SHEET_PATH), sheet)
    unique_ids = result["canonical_id"].nunique()

    print("\nCrop extraction complete")
    print(f"Selected observations: {len(selected)}")
    print(f"Saved crops: {len(result)}")
    print(f"Temporary IDs represented: {unique_ids}")
    print(f"Skipped observations: {skipped}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Contact sheet: {CONTACT_SHEET_PATH}")
    print("\nAll team labels remain blank.")


if __name__ == "__main__":
    main()
