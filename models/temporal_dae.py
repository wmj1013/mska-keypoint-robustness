import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)


class TemporalDAE(nn.Module):
    """
    Temporal denoising autoencoder for MSKA keypoint recovery.

    Input keypoints:
        [B, 3, T, N]

    Missing mask:
        [B, T, N]

        True  = observed joint
        False = missing joint

    Output:
        Predicted keypoints [B, 3, T, N]

    The model predicts all selected keypoints, but during final
    recovery only artificially missing positions should be replaced.
    """

    def __init__(
        self,
        num_joints=79,
        num_channels=3,
        projection_dim=256,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.num_joints = num_joints
        self.num_channels = num_channels

        # 79 joints × 3 features = 237
        self.keypoint_dim = (
            num_joints * num_channels
        )

        # 237 keypoint features + 79 mask values = 316
        self.input_dim = (
            self.keypoint_dim + num_joints
        )

        # Convert each frame to a compact feature representation.
        self.input_projection = nn.Sequential(
            nn.Linear(
                self.input_dim,
                projection_dim,
            ),
            nn.ReLU(),
        )

        # Bidirectional GRU:
        # one direction processes past -> future,
        # the other future -> past.
        self.temporal_encoder = nn.GRU(
            input_size=projection_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        # BiGRU output dimension = hidden_size × 2.
        self.decoder = nn.Linear(
            hidden_size * 2,
            self.keypoint_dim,
        )

    def forward(
        self,
        keypoints,
        missing_mask,
        valid_lengths,
    ):
        """
        Args:
            keypoints:
                Corrupted selected keypoints,
                shape [B, 3, T, 79].

            missing_mask:
                Boolean mask,
                shape [B, T, 79].

            valid_lengths:
                True sequence length for each sample,
                shape [B].

        Returns:
            predictions:
                Predicted selected keypoints,
                shape [B, 3, T, 79].
        """

        batch_size, channels, time_steps, num_joints = (
            keypoints.shape
        )

        if channels != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, "
                f"but received {channels}."
            )

        if num_joints != self.num_joints:
            raise ValueError(
                f"Expected {self.num_joints} joints, "
                f"but received {num_joints}."
            )

        expected_mask_shape = (
            batch_size,
            time_steps,
            num_joints,
        )

        if missing_mask.shape != expected_mask_shape:
            raise ValueError(
                "missing_mask has incorrect shape. "
                f"Expected {expected_mask_shape}, "
                f"received {tuple(missing_mask.shape)}."
            )

        # --------------------------------------------------
        # Convert keypoints from:
        # [B, 3, T, 79]
        #
        # to:
        # [B, T, 79, 3]
        #
        # then flatten each frame:
        # [B, T, 237]
        # --------------------------------------------------

        keypoint_features = (
            keypoints
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(
                batch_size,
                time_steps,
                self.keypoint_dim,
            )
        )

        # --------------------------------------------------
        # Add the explicit observed/missing mask.
        #
        # True  -> 1
        # False -> 0
        #
        # Final frame representation:
        # 237 + 79 = 316 dimensions.
        # --------------------------------------------------

        mask_features = missing_mask.to(
            dtype=keypoints.dtype
        )

        model_input = torch.cat(
            [
                keypoint_features,
                mask_features,
            ],
            dim=-1,
        )

        model_input = self.input_projection(
            model_input
        )

        # --------------------------------------------------
        # Ignore padding inside the GRU.
        # --------------------------------------------------

        packed_input = pack_padded_sequence(
            model_input,
            lengths=valid_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.temporal_encoder(
            packed_input
        )

        temporal_features, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=time_steps,
        )

        # --------------------------------------------------
        # Predict all 79 × 3 keypoint features.
        # --------------------------------------------------

        predictions = self.decoder(
            temporal_features
        )

        # [B, T, 237]
        # -> [B, T, 79, 3]
        predictions = predictions.view(
            batch_size,
            time_steps,
            self.num_joints,
            self.num_channels,
        )

        # -> [B, 3, T, 79]
        predictions = (
            predictions
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return predictions


def replace_missing_keypoints(
    corrupted_keypoints,
    predictions,
    missing_mask,
):
    """
    Replace only artificially missing joint locations.

    Observed keypoints are preserved exactly.

    Args:
        corrupted_keypoints:
            [B, 3, T, 79]

        predictions:
            [B, 3, T, 79]

        missing_mask:
            [B, T, 79]

    Returns:
        recovered_keypoints:
            [B, 3, T, 79]
    """

    if corrupted_keypoints.shape != predictions.shape:
        raise ValueError(
            "Prediction shape must match keypoint shape."
        )

    # [B, T, 79]
    # -> [B, 1, T, 79]
    observed_mask = missing_mask.unsqueeze(1)

    recovered_keypoints = torch.where(
        observed_mask,
        corrupted_keypoints,
        predictions,
    )

    return recovered_keypoints
