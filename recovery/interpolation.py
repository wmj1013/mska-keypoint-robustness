import torch


def linear_interpolate_missing(
    keypoints,
    missing_mask,
    selected_joints,
    valid_lengths=None,
):
    """
    Recover missing keypoints using temporal linear interpolation.

    Args:
        keypoints:
            Tensor with shape [B, C, T, J].
            Missing joints have already been set to zero.

        missing_mask:
            Boolean tensor with shape [B, T, N].
            True  = observed joint
            False = missing joint

        selected_joints:
            Raw indices of the N MSKA-selected joints.

        valid_lengths:
            Number of valid frames for each sample.

    Returns:
        recovered_keypoints:
            Tensor with the same shape as keypoints.

    Recovery rules:
        1. Missing point between two observed frames:
           use linear interpolation.

        2. Missing point before the first observed frame:
           use the nearest future observation.

        3. Missing point after the last observed frame:
           use the nearest previous observation.

        4. If a joint is missing for the entire sequence:
           leave it unchanged because interpolation is impossible.
    """

    recovered_keypoints = keypoints.clone()

    batch_size, _, time_steps, _ = keypoints.shape

    if missing_mask.shape[0] != batch_size:
        raise ValueError(
            "Batch size of missing_mask does not match keypoints."
        )

    if missing_mask.shape[2] != len(selected_joints):
        raise ValueError(
            "missing_mask joint dimension does not match selected_joints."
        )

    for b in range(batch_size):

        if valid_lengths is None:
            valid_length = time_steps
        else:
            valid_length = int(valid_lengths[b])

        for local_joint_index, raw_joint_index in enumerate(
            selected_joints
        ):

            observed = missing_mask[
                b,
                :valid_length,
                local_joint_index,
            ]

            # Nothing is missing for this joint.
            if observed.all():
                continue

            observed_frames = torch.where(
                observed
            )[0]

            # No temporal information is available.
            if observed_frames.numel() == 0:
                continue

            missing_frames = torch.where(
                ~observed
            )[0]

            for t_tensor in missing_frames:

                t = int(t_tensor)

                previous_frames = observed_frames[
                    observed_frames < t
                ]

                next_frames = observed_frames[
                    observed_frames > t
                ]

                has_previous = (
                    previous_frames.numel() > 0
                )

                has_next = (
                    next_frames.numel() > 0
                )

                # ------------------------------------------
                # Case 1:
                # observed frame on both sides
                # -> true linear interpolation
                # ------------------------------------------

                if has_previous and has_next:

                    previous_t = int(
                        previous_frames[-1]
                    )

                    next_t = int(
                        next_frames[0]
                    )

                    alpha = (
                        (t - previous_t)
                        / (next_t - previous_t)
                    )

                    previous_value = recovered_keypoints[
                        b,
                        :,
                        previous_t,
                        raw_joint_index,
                    ]

                    next_value = recovered_keypoints[
                        b,
                        :,
                        next_t,
                        raw_joint_index,
                    ]

                    recovered_keypoints[
                        b,
                        :,
                        t,
                        raw_joint_index,
                    ] = (
                        (1.0 - alpha) * previous_value
                        + alpha * next_value
                    )

                # ------------------------------------------
                # Case 2:
                # beginning of sequence
                # -> nearest future observation
                # ------------------------------------------

                elif has_next:

                    next_t = int(
                        next_frames[0]
                    )

                    recovered_keypoints[
                        b,
                        :,
                        t,
                        raw_joint_index,
                    ] = recovered_keypoints[
                        b,
                        :,
                        next_t,
                        raw_joint_index,
                    ]

                # ------------------------------------------
                # Case 3:
                # end of sequence
                # -> nearest previous observation
                # ------------------------------------------

                elif has_previous:

                    previous_t = int(
                        previous_frames[-1]
                    )

                    recovered_keypoints[
                        b,
                        :,
                        t,
                        raw_joint_index,
                    ] = recovered_keypoints[
                        b,
                        :,
                        previous_t,
                        raw_joint_index,
                    ]

    return recovered_keypoints
