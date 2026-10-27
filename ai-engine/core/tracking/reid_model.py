
from pathlib import Path

import cv2
import numpy as np
import torch
try:
    import torchreid
except ImportError:
    torchreid = None


class OSNetReID:
    """
    Production OSNet person Re-ID module for AthletiQ.

    Responsibilities:
        1. Load the trained OSNet MSMT17 checkpoint.
        2. Extract player crops from bounding boxes.
        3. Generate normalized 512-D embeddings.
        4. Compute cosine similarity between player embeddings.
    """

    def __init__(
        self,
        checkpoint_path,
        device=None,
        input_size=(256, 128),
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.input_size = input_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Re-ID checkpoint not found: {self.checkpoint_path}"
            )

        if torchreid is None:
            raise ImportError(
                "torchreid is required to initialize OSNetReID, but it is not installed. "
                "Please install torchreid in your environment."
            )

        self.model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=4101,
            pretrained=False,
        )

        # ---------------------------------------------------------
        # Load MSMT17 checkpoint
        # ---------------------------------------------------------

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        state_dict = checkpoint

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith("module."):
                key = key[len("module."):]

            cleaned_state_dict[key] = value

        self.model.load_state_dict(
            cleaned_state_dict,
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------
        # ImageNet preprocessing
        # ---------------------------------------------------------

        from torchvision import transforms

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(
                self.input_size,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.embedding_dimension = 512

    # =============================================================
    # PLAYER CROP
    # =============================================================

    @staticmethod
    def crop_player(
        frame,
        bbox,
        padding=0.0,
    ):
        """
        Crop a player from a BGR OpenCV frame.

        bbox:
            [x1, y1, x2, y2]
        """

        h, w = frame.shape[:2]

        x1, y1, x2, y2 = bbox

        bw = x2 - x1
        bh = y2 - y1

        x1 -= bw * padding
        y1 -= bh * padding
        x2 += bw * padding
        y2 += bh * padding

        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(w, int(round(x2)))
        y2 = min(h, int(round(y2)))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return crop

    # =============================================================
    # EMBEDDING
    # =============================================================

    @torch.no_grad()
    def extract_embedding(self, player_crop):
        """
        Extract one normalized 512-D embedding.
        """

        if player_crop is None:
            return None

        if player_crop.size == 0:
            return None

        # OpenCV BGR -> RGB
        rgb = cv2.cvtColor(
            player_crop,
            cv2.COLOR_BGR2RGB,
        )

        tensor = self.transform(rgb)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)

        embedding = self.model(tensor)

        if isinstance(embedding, tuple):
            embedding = embedding[0]

        embedding = F.normalize(
            embedding,
            p=2,
            dim=1,
        )

        return embedding[0].detach()

    # =============================================================
    # FRAME + BBOX
    # =============================================================

    @torch.no_grad()
    def extract_from_bbox(
        self,
        frame,
        bbox,
        padding=0.0,
    ):
        """
        Extract a normalized player embedding directly
        from a frame and bounding box.
        """

        crop = self.crop_player(
            frame,
            bbox,
            padding=padding,
        )

        return self.extract_embedding(crop)

    # =============================================================
    # COSINE SIMILARITY
    # =============================================================

    @staticmethod
    def cosine_similarity(
        embedding_a,
        embedding_b,
    ):
        """
        Cosine similarity between two normalized embeddings.

        Returns:
            float in approximately [-1, 1].
        """

        if embedding_a is None or embedding_b is None:
            return None

        if not torch.is_tensor(embedding_a):
            embedding_a = torch.tensor(
                embedding_a,
                dtype=torch.float32,
            )

        if not torch.is_tensor(embedding_b):
            embedding_b = torch.tensor(
                embedding_b,
                dtype=torch.float32,
            )

        embedding_a = F.normalize(
            embedding_a.float(),
            p=2,
            dim=-1,
        )

        embedding_b = F.normalize(
            embedding_b.float(),
            p=2,
            dim=-1,
        )

        similarity = torch.sum(
            embedding_a * embedding_b,
            dim=-1,
        )

        return float(similarity.item())

    # =============================================================
    # MODEL INFO
    # =============================================================

    def info(self):
        return {
            "model": "OSNet x1.0",
            "checkpoint": str(self.checkpoint_path),
            "device": str(self.device),
            "input_size": self.input_size,
            "embedding_dimension": self.embedding_dimension,
        }
