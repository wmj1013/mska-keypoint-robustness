import zlib
import torch


def generate_spatial_occlusion_mask(
    keypoints,
    selected_joints,
    occlusion_width=0.30,
    occlusion_height=0.30,
    valid_lengths=None,
    sample_names=None,
    seed=42,
):
    """
    Generate local spatial occlusion in normalized x-y space.

    Args:
        keypoints:
            Tensor with shape [B, C, T, J].
            Channel 0 = x, channel 1 = y,
            channel 2 = detection score.

        selected_joints:
            Raw joint indices of the 79 keypoints used by MSKA.

        occlusion_width:
            Width of the rectangular occlusion region
            in normalized coordinate space.

        occlusion_height:
            Height of the rectangular occlusion region
            in normalized coordinate space.

        valid_lengths:
            Number of valid frames for each sample.

        sample_names:
            Sample names used to make the occlusion
            reproducible for each video.

        seed:
            Base random seed.

    Returns:
        corrupted_keypoints:
            Keypoints after spatial occlusion.

        missing_mask:
            Binary mask with shape [B, T, N].
            True  = observed joint
            False = spatially occluded joint
    """

    if occlusion_width <= 0 or occlusion_height <= 0:
        raise ValueError(
            "Occlusion width and height must be positive."
        )

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

    for b in range(batch_size):

        if valid_lengths is None:
            valid_length = time_steps
        else:
            valid_length = int(valid_lengths[b])

        if sample_names is None:
            sample_name = f"sample-{b}"
        else:
            sample_name = str(sample_names[b])

        # --------------------------------------------------
        # Create a reproducible occlusion region
        # for this video.
        # --------------------------------------------------

        video_seed = zlib.crc32(
            f"{seed}-{sample_name}".encode()
        )

        generator = torch.Generator()
        generator.manual_seed(video_seed)

        # Pick one real frame.
        anchor_frame = torch.randint(
            low=0,
            high=valid_length,
            size=(1,),
            generator=generator,
        ).item()

        # Pick one of MSKA's 79 selected joints.
        anchor_position = torch.randint(
            low=0,
            high=num_selected,
            size=(1,),
            generator=generator,
        ).item()

        anchor_joint = selected_joints[anchor_position]

        center_x = keypoints[
            b, 0, anchor_frame, anchor_joint
        ].item()

        center_y = keypoints[
            b, 1, anchor_frame, anchor_joint
        ].item()

        half_width = occlusion_width / 2.0
        half_height = occlusion_height / 2.0

        x_min = center_x - half_width
        x_max = center_x + half_width
        y_min = center_y - half_height
        y_max = center_y + half_height

        # --------------------------------------------------
        # Apply the same spatial region to all valid frames.
        # --------------------------------------------------

        for t in range(valid_length):

            x = keypoints[
                b, 0, t, selected_joints
            ]

            y = keypoints[
                b, 1, t, selected_joints
            ]

            inside_region = (
                (x >= x_min)
                & (x <= x_max)
                & (y >= y_min)
                & (y <= y_max)
            )

            missing_mask[
                b, t, inside_region
            ] = False

            missing_positions = torch.where(
                inside_region
            )[0].tolist()

            joints_to_remove = [
                selected_joints[i]
                for i in missing_positions
            ]

            if joints_to_remove:
                # Remove x, y and detection score together.
                corrupted_keypoints[
                    b, :, t, joints_to_remove
                ] = 0.0

    return corrupted_keypoints, missing_mask
