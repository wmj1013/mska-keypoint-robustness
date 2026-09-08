import zlib
import torch


def generate_random_joint_mask(
    keypoints,
    selected_joints,
    missing_ratio=0.2,
    valid_lengths=None,
    sample_names=None,
    seed=42,
):
    """
    Generate reproducible random joint-level missingness.

    Args:
        keypoints:
            Tensor with shape [B, C, T, J].
            C contains x, y and detection score.

        selected_joints:
            Indices of the joints used by MSKA.

        missing_ratio:
            Proportion of selected joints to remove in each valid frame.

        valid_lengths:
            Number of valid frames for each sample.
            If None, all frames are treated as valid.

        sample_names:
            Names of samples in the batch.
            These are used to generate a stable mask for each
            video and frame.

        seed:
            Base random seed.

    Returns:
        corrupted_keypoints:
            Keypoints after masking.

        missing_mask:
            Binary mask with shape [B, T, N].
            True  = observed joint
            False = missing joint
            N = number of selected joints.
    """

    if not 0.0 <= missing_ratio <= 1.0:
        raise ValueError("missing_ratio must be between 0 and 1.")

    corrupted_keypoints = keypoints.clone()

    batch_size, _, time_steps, _ = keypoints.shape
    num_selected = len(selected_joints)

    missing_mask = torch.ones(
        batch_size,
        time_steps,
        num_selected,
        dtype=torch.bool,
        device=keypoints.device,
    )

    num_missing = round(num_selected * missing_ratio)

    for b in range(batch_size):

        if valid_lengths is None:
            valid_length = time_steps
        else:
            valid_length = int(valid_lengths[b])

        if sample_names is None:
            sample_name = f"sample-{b}"
        else:
            sample_name = str(sample_names[b])

        for t in range(valid_length):

            # Create a stable seed for this exact video and frame.
            frame_seed = zlib.crc32(
                f"{seed}-{sample_name}-{t}".encode()
            )

            generator = torch.Generator()
            generator.manual_seed(frame_seed)

            random_indices = torch.randperm(
                num_selected,
                generator=generator,
            )[:num_missing]

            missing_mask[b, t, random_indices] = False

            joints_to_remove = [
                selected_joints[i]
                for i in random_indices.tolist()
            ]

            # Remove the whole joint:
            # x, y and detection score are all set to zero.
            corrupted_keypoints[b, :, t, joints_to_remove] = 0.0

    return corrupted_keypoints, missing_mask
