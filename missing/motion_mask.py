import zlib
import torch


def generate_motion_aware_mask(
    keypoints,
    selected_joints,
    missing_ratio=0.01,
    valid_lengths=None,
    sample_names=None,
    seed=42,
    temperature=0.5,
    coordinate_limit=2.0,
):
    """
    Generate motion-aware joint missingness.

    Motion intensity is estimated from normalized x-y displacement
    between adjacent frames and converted into masking probabilities
    using temperature-scaled Softmax.

    Faster-moving joints therefore have a higher probability
    of being masked.

    Args:
        keypoints:
            Tensor with shape [B, C, T, J].
            Channel 0 = x
            Channel 1 = y
            Channel 2 = detection score

        selected_joints:
            Raw indices of the 79 keypoints used by MSKA.

        missing_ratio:
            Proportion of selected joints to mask per valid frame.

        valid_lengths:
            Number of valid frames for each sample.

        sample_names:
            Sample names used for reproducible masking.

        seed:
            Base random seed.

        temperature:
            Softmax temperature.
            Lower values give stronger preference to high-motion joints.
            Higher values produce a flatter distribution.

    Returns:
        corrupted_keypoints:
            Keypoints after motion-aware masking.

        missing_mask:
            Boolean mask with shape [B, T, N].
            True = observed joint
            False = missing joint
    """

    if not 0.0 <= missing_ratio <= 1.0:
        raise ValueError(
            "missing_ratio must be between 0 and 1."
        )

    if temperature <= 0:
        raise ValueError(
            "temperature must be positive."
        )

    corrupted_keypoints = keypoints.clone()

    batch_size, _, time_steps, _ = keypoints.shape
    num_selected = len(selected_joints)

    num_missing = round(
        num_selected * missing_ratio
    )

    missing_mask = torch.ones(
        batch_size,
        time_steps,
        num_selected,
        dtype=torch.bool,
        device=keypoints.device,
    )

    if num_missing == 0:
        return corrupted_keypoints, missing_mask

    for b in range(batch_size):

        if valid_lengths is None:
            valid_length = time_steps
        else:
            valid_length = int(valid_lengths[b])

        if sample_names is None:
            sample_name = f"sample-{b}"
        else:
            sample_name = str(sample_names[b])

        # Shape: [2, T, 79]
        xy = keypoints[
            b,
            :2,
            :valid_length,
            selected_joints,
        ]

        # Coordinates are expected to be roughly within [-1, 1]
        # after dataset normalization. A wider limit is used here
        # only to reject extreme detector outliers when estimating
        # motion. The original keypoint values are not modified.
        coordinate_valid = (
            xy.abs() <= coordinate_limit
        ).all(dim=0)

        # Motion intensity for every frame and joint.
        motion = torch.zeros(
            valid_length,
            num_selected,
            dtype=keypoints.dtype,
            device=keypoints.device,
        )

        if valid_length > 1:

            previous_xy = xy[:, :-1, :]
            next_xy = xy[:, 1:, :]

            # Adjacent-frame x-y displacement.
            frame_difference = (
                next_xy - previous_xy
            )

            # Euclidean motion magnitude.
            step_motion = torch.linalg.vector_norm(
                frame_difference,
                dim=0,
            )

            # A transition is valid only when both endpoints
            # contain plausible normalized coordinates.
            valid_transition = (
                coordinate_valid[:-1]
                & coordinate_valid[1:]
            )

            step_motion = torch.where(
                valid_transition,
                step_motion,
                torch.zeros_like(step_motion),
            )

            # Boundary frames.
            motion[0] = step_motion[0]
            motion[-1] = step_motion[-1]

            # Interior frames use both temporal directions.
            if valid_length > 2:
                motion[1:-1] = (
                    step_motion[:-1]
                    + step_motion[1:]
                ) / 2.0

        for t in range(valid_length):

            # Convert motion intensity into masking probability.
            logits = motion[t] / temperature

            # Do not interpret an extreme detector coordinate
            # as genuine high-speed human motion.
            valid_candidates = coordinate_valid[t]

            if valid_candidates.sum().item() >= num_missing:
                logits = logits.masked_fill(
                    ~valid_candidates,
                    -1e9,
                )

            probabilities = torch.softmax(
                logits,
                dim=0,
            )

            probabilities_cpu = (
                probabilities.detach().cpu()
            )

            # Stable random seed for this video and frame.
            frame_seed = zlib.crc32(
                f"{seed}-{sample_name}-{t}".encode()
            )

            generator = torch.Generator()
            generator.manual_seed(frame_seed)

            # Select a fixed number of joints.
            # High-motion joints have higher probability.
            missing_positions = torch.multinomial(
                probabilities_cpu,
                num_samples=num_missing,
                replacement=False,
                generator=generator,
            )

            missing_mask[
                b,
                t,
                missing_positions.to(
                    missing_mask.device
                ),
            ] = False

            joints_to_remove = [
                selected_joints[i]
                for i in missing_positions.tolist()
            ]

            # Remove x, y and detection score together.
            corrupted_keypoints[
                b,
                :,
                t,
                joints_to_remove,
            ] = 0.0

    return corrupted_keypoints, missing_mask
