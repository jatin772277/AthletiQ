import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_VIDEO = ROOT / "datasets/football/raw/videos/football2.mp4"
DEFAULT_TRACKS = ROOT / "outputs/joint_tracking/football2_player_tracks.csv"
DEFAULT_MODEL = ROOT / "ai-engine/models/team_identification/best_team_classifier.pt"
DEFAULT_OUT = ROOT / "outputs/team_identification"


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )

    if checkpoint.get("model_name") != "mobilenet_v3_small":
        raise ValueError(
            f"Unexpected model architecture: {checkpoint.get('model_name')}"
        )

    classes = checkpoint["classes"]
    image_size = int(checkpoint["image_size"])

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(
        model.classifier[3].in_features, len(classes)
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    mean = checkpoint["mean"]
    std = checkpoint["std"]

    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return model, classes, preprocess


def crop_player(frame, row, min_width, min_height):
    height, width = frame.shape[:2]

    x1 = int(np.floor(float(row.x1)))
    y1 = int(np.floor(float(row.y1)))
    x2 = int(np.ceil(float(row.x2)))
    y2 = int(np.ceil(float(row.y2)))

    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    if x2 - x1 < min_width or y2 - y1 < min_height:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Training crops were saved as image files; convert OpenCV BGR to RGB.
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def main():
    parser = argparse.ArgumentParser(
        description="Classify player teams and aggregate predictions by canonical ID."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-width", type=int, default=16)
    parser.add_argument("--min-height", type=int, default=35)
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=0,
        help="Add this value to frame_id when seeking the video (normally 0).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.60,
        help="Predictions below this confidence are marked unknown.",
    )
    args = parser.parse_args()

    for path in (args.video, args.tracks, args.model):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("--min-confidence must be between 0 and 1")

    df = pd.read_csv(args.tracks)

    required = {
        "frame_id", "canonical_id", "x1", "y1", "x2", "y2"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Tracking CSV is missing columns: {sorted(missing)}")

    df = df.copy()
    df["team_prediction"] = "unknown"
    df["team_confidence"] = np.nan
    df["team_a_probability"] = np.nan
    df["team_b_probability"] = np.nan
    df["team_crop_valid"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, classes, preprocess = load_model(args.model, device)
    print(f"Classes: {classes}")
    print(f"Tracking observations: {len(df)}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video frames: {total_frames}")

    # Process all observations belonging to a frame together.
    groups = df.groupby("frame_id", sort=True).indices
    class_index = {name: i for i, name in enumerate(classes)}
    probability_columns = {
        name: f"prob_{name}" for name in classes
    }

    for frame_id, indices in tqdm(groups.items(), desc="Classifying frames"):
        video_frame_id = int(frame_id) + args.frame_offset

        if video_frame_id < 0 or video_frame_id >= total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_id)
        ok, frame = cap.read()
        if not ok:
            continue

        batch_tensors = []
        batch_rows = []

        def predict_batch():
            if not batch_tensors:
                return

            tensor = torch.stack(batch_tensors).to(device)

            with torch.inference_mode():
                logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()

            for row_index, probs in zip(batch_rows, probabilities):
                for name, idx in class_index.items():
                    df.at[row_index, probability_columns[name]] = float(probs[idx])

                best_idx = int(np.argmax(probs))
                confidence = float(probs[best_idx])
                predicted_class = classes[best_idx]

                df.at[row_index, "team_confidence"] = confidence
                df.at[row_index, "team_crop_valid"] = True
                df.at[row_index, "team_prediction"] = (
                    predicted_class
                    if confidence >= args.min_confidence
                    else "unknown"
                )

            batch_tensors.clear()
            batch_rows.clear()

        for row_index in indices:
            row = df.loc[row_index]
            crop = crop_player(
                frame, row, args.min_width, args.min_height
            )
            if crop is None:
                continue

            batch_tensors.append(preprocess(crop))
            batch_rows.append(row_index)

            if len(batch_tensors) >= args.batch_size:
                predict_batch()

        predict_batch()

    cap.release()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate model probabilities across observations for each temporary ID.
    # This does not merge or rename any tracking IDs.
    aggregation_rows = []

    for canonical_id, group in df.groupby("canonical_id", dropna=False):
        valid = group[group["team_crop_valid"]]

        record = {
            "canonical_id": canonical_id,
            "observations": len(group),
            "valid_crops": len(valid),
        }

        if valid.empty:
            record.update({
                "team_label": "unknown",
                "team_confidence": np.nan,
                "team_a_probability": np.nan,
                "team_b_probability": np.nan,
            })
        else:
            # Average probabilities, rather than majority-voting hard labels.
            mean_probs = {}
            for name in classes:
                col = probability_columns[name]
                mean_probs[name] = float(valid[col].mean())

            best_class = max(mean_probs, key=mean_probs.get)
            confidence = mean_probs[best_class]

            record.update({
                "team_label": (
                    best_class
                    if confidence >= args.min_confidence
                    else "unknown"
                ),
                "team_confidence": confidence,
                "team_a_probability": mean_probs.get("team_a", np.nan),
                "team_b_probability": mean_probs.get("team_b", np.nan),
            })

        aggregation_rows.append(record)

    summary = pd.DataFrame(aggregation_rows)
    summary_map = summary.set_index("canonical_id")

    df["aggregated_team"] = df["canonical_id"].map(
        summary_map["team_label"]
    )
    df["aggregated_team_confidence"] = df["canonical_id"].map(
        summary_map["team_confidence"]
    )

    observations_path = args.output_dir / "football2_team_predictions.csv"
    summary_path = args.output_dir / "football2_team_summary.csv"
    metrics_path = args.output_dir / "football2_team_identification_metrics.json"

    df.to_csv(observations_path, index=False)
    summary.to_csv(summary_path, index=False)

    metrics = {
        "model": "MobileNetV3-Small",
        "classes": classes,
        "video": str(args.video),
        "tracking_csv": str(args.tracks),
        "observations": int(len(df)),
        "valid_crops": int(df["team_crop_valid"].sum()),
        "invalid_crops": int((~df["team_crop_valid"]).sum()),
        "unique_temporary_ids": int(df["canonical_id"].nunique()),
        "team_summary_counts": {
            str(k): int(v)
            for k, v in summary["team_label"].value_counts().items()
        },
        "min_confidence": args.min_confidence,
        "frame_offset": args.frame_offset,
        "note": (
            "Team predictions are model estimates. Temporary canonical IDs "
            "are preserved and are not verified real-player identities. "
            "Predictions from the same video are not an independent accuracy test."
        ),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nTeam summary:")
    print(summary["team_label"].value_counts(dropna=False).to_string())
    print(f"\nObservation predictions: {observations_path}")
    print(f"Per-ID summary:         {summary_path}")
    print(f"Run metrics:            {metrics_path}")


if __name__ == "__main__":
    main()
