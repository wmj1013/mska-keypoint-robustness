import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from datasets import S2T_Dataset
from Tokenizer import GlossTokenizer_S2G
from train import get_args_parser

from missing.random_mask import generate_random_joint_mask
from missing.spatial_mask import generate_spatial_occlusion_mask
from missing.motion_mask import generate_motion_aware_mask

from models.temporal_dae import TemporalDAE


# --------------------------------------------------
# Unified training conditions
#
# The same Temporal DAE sees:
#   - three missingness mechanisms
#   - two missingness severities
# --------------------------------------------------

TRAIN_CONDITIONS = [
    {
        "name": "random_light",
        "type": "random",
        "missing_ratio": 0.01,
    },
    {
        "name": "random_medium",
        "type": "random",
        "missing_ratio": 0.02,
    },
    {
        "name": "spatial_light",
        "type": "spatial",
        "size": 0.10,
    },
    {
        "name": "spatial_medium",
        "type": "spatial",
        "size": 0.15,
    },
    {
        "name": "motion_light",
        "type": "motion",
        "missing_ratio": 0.01,
        "temperature": 0.5,
    },
    {
        "name": "motion_medium",
        "type": "motion",
        "missing_ratio": 0.02,
        "temperature": 0.5,
    },
]


# Validation uses the same six corruption conditions.
# This allows the best unified model to be selected according
# to performance across all missingness types and severities.
VAL_CONDITIONS = TRAIN_CONDITIONS


def apply_corruption(
    keypoints,
    src_input,
    selected_joints,
    condition,
    seed,
):
    """
    Apply one artificial missingness condition.

    Returns:
        corrupted_keypoints:
            Full 133-joint tensor after corruption.

        missing_mask:
            Boolean mask [B, T, 79].
            True  = observed
            False = missing
    """

    missing_type = condition["type"]

    if missing_type == "random":

        return generate_random_joint_mask(
            keypoints=keypoints,
            selected_joints=selected_joints,
            missing_ratio=condition["missing_ratio"],
            valid_lengths=src_input["src_length"],
            sample_names=src_input["name"],
            seed=seed,
        )

    if missing_type == "spatial":

        return generate_spatial_occlusion_mask(
            keypoints=keypoints,
            selected_joints=selected_joints,
            occlusion_width=condition["size"],
            occlusion_height=condition["size"],
            valid_lengths=src_input["src_length"],
            sample_names=src_input["name"],
            seed=seed,
        )

    if missing_type == "motion":

        return generate_motion_aware_mask(
            keypoints=keypoints,
            selected_joints=selected_joints,
            missing_ratio=condition["missing_ratio"],
            valid_lengths=src_input["src_length"],
            sample_names=src_input["name"],
            seed=seed,
            temperature=condition["temperature"],
        )

    raise ValueError(
        f"Unsupported missing type: {missing_type}"
    )


def masked_mse_loss(
    predictions,
    targets,
    missing_mask,
    valid_lengths,
):
    """
    MSE calculated only at artificially missing locations.

    Padding frames are excluded.

    Args:
        predictions:
            [B, 3, T, 79]

        targets:
            [B, 3, T, 79]

        missing_mask:
            [B, T, 79]

        valid_lengths:
            [B]
    """

    _, _, time_steps, _ = predictions.shape

    frame_index = torch.arange(
        time_steps,
        device=predictions.device,
    ).unsqueeze(0)

    valid_frame_mask = (
        frame_index
        < valid_lengths.unsqueeze(1)
    )

    # True only for locations that:
    #   1. belong to a real frame
    #   2. were artificially removed
    reconstruction_mask = (
        (~missing_mask)
        & valid_frame_mask.unsqueeze(-1)
    )

    reconstruction_mask = (
        reconstruction_mask
        .unsqueeze(1)
        .expand_as(predictions)
    )

    if not reconstruction_mask.any():
        return predictions.sum() * 0.0

    squared_error = (
        predictions - targets
    ).pow(2)

    return squared_error[
        reconstruction_mask
    ].mean()


@torch.no_grad()
def evaluate_condition(
    model,
    dataloader,
    selected_joints,
    condition,
    device,
    max_batches,
):
    """
    Measure masked reconstruction MSE on one fixed
    validation corruption condition.
    """

    model.eval()

    total_loss = 0.0
    total_batches = 0

    for step, src_input in enumerate(dataloader):

        if (
            max_batches > 0
            and step >= max_batches
        ):
            break

        clean_full = src_input["keypoint"]

        corrupted_full, missing_mask = apply_corruption(
            keypoints=clean_full,
            src_input=src_input,
            selected_joints=selected_joints,
            condition=condition,
            seed=42,
        )

        clean_selected = clean_full[
            :,
            :,
            :,
            selected_joints,
        ].to(device)

        corrupted_selected = corrupted_full[
            :,
            :,
            :,
            selected_joints,
        ].to(device)

        missing_mask = missing_mask.to(device)

        valid_lengths = torch.as_tensor(
            src_input["src_length"],
            dtype=torch.long,
            device=device,
        )

        predictions = model(
            keypoints=corrupted_selected,
            missing_mask=missing_mask,
            valid_lengths=valid_lengths,
        )

        loss = masked_mse_loss(
            predictions=predictions,
            targets=clean_selected,
            missing_mask=missing_mask,
            valid_lengths=valid_lengths,
        )

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def main(args, config):

    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------

    seed = getattr(args, "seed", 42)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Training device: {device}"
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    tokenizer = GlossTokenizer_S2G(
        config["gloss"]
    )

    train_data = S2T_Dataset(
        path=config["data"]["train_label_path"],
        tokenizer=tokenizer,
        config=config,
        args=args,
        phase="train",
        training_refurbish=True,
    )

    dev_data = S2T_Dataset(
        path=config["data"]["dev_label_path"],
        tokenizer=tokenizer,
        config=config,
        args=args,
        phase="val",
        training_refurbish=True,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_data.collate_fn,
        pin_memory=args.pin_mem,
    )

    dev_loader = DataLoader(
        dev_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dev_data.collate_fn,
        pin_memory=args.pin_mem,
    )

    selected_joints = config[
        "model"
    ]["RecognitionNetwork"]["DSTA-Net"]["body"]

    print(
        f"Training samples: {len(train_data)}"
    )

    print(
        f"Validation samples: {len(dev_data)}"
    )

    print(
        f"Selected MSKA joints: {len(selected_joints)}"
    )

    # --------------------------------------------------
    # Temporal DAE
    # --------------------------------------------------

    model = TemporalDAE(
        num_joints=79,
        num_channels=3,
        projection_dim=256,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.dae_lr,
        weight_decay=args.dae_weight_decay,
    )

    checkpoint_dir = Path(
        args.dae_checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float("inf")
    start_epoch = 0

    # --------------------------------------------------
    # Optional resume from checkpoint
    # --------------------------------------------------

    if args.dae_resume:

        checkpoint = torch.load(
            args.dae_resume,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        start_epoch = checkpoint["epoch"]

        best_val_loss = checkpoint.get(
            "mean_val_loss",
            float("inf"),
        )

        print(
            f"Resumed from: {args.dae_resume}"
        )

        print(
            f"Completed epoch: {start_epoch}"
        )

        print(
            f"Previous best/validation MSE: "
            f"{best_val_loss:.6f}"
        )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    for epoch in range(
        start_epoch,
        args.dae_epochs,
    ):

        model.train()

        running_loss = 0.0
        train_batches = 0

        for step, src_input in enumerate(
            train_loader
        ):

            if (
                args.dae_max_train_batches > 0
                and step
                >= args.dae_max_train_batches
            ):
                break

            # Cycle through all six training conditions.
            condition = TRAIN_CONDITIONS[
                (step + epoch)
                % len(TRAIN_CONDITIONS)
            ]

            # Different corruption seed every epoch/batch.
            corruption_seed = (
                seed
                + epoch * 100000
                + step
            )

            clean_full = src_input["keypoint"]

            corrupted_full, missing_mask = apply_corruption(
                keypoints=clean_full,
                src_input=src_input,
                selected_joints=selected_joints,
                condition=condition,
                seed=corruption_seed,
            )

            clean_selected = clean_full[
                :,
                :,
                :,
                selected_joints,
            ].to(device)

            corrupted_selected = corrupted_full[
                :,
                :,
                :,
                selected_joints,
            ].to(device)

            missing_mask = missing_mask.to(
                device
            )

            valid_lengths = torch.as_tensor(
                src_input["src_length"],
                dtype=torch.long,
                device=device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            predictions = model(
                keypoints=corrupted_selected,
                missing_mask=missing_mask,
                valid_lengths=valid_lengths,
            )

            loss = masked_mse_loss(
                predictions=predictions,
                targets=clean_selected,
                missing_mask=missing_mask,
                valid_lengths=valid_lengths,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            running_loss += loss.item()
            train_batches += 1

            if step % 100 == 0:

                print(
                    f"Epoch "
                    f"{epoch + 1}/"
                    f"{args.dae_epochs} | "
                    f"Step {step} | "
                    f"Condition "
                    f"{condition['name']} | "
                    f"Loss "
                    f"{loss.item():.6f}"
                )

        train_loss = (
            running_loss
            / max(train_batches, 1)
        )

        # --------------------------------------------------
        # Validation under all three missingness types.
        # --------------------------------------------------

        val_losses = {}

        for condition in VAL_CONDITIONS:

            val_loss = evaluate_condition(
                model=model,
                dataloader=dev_loader,
                selected_joints=selected_joints,
                condition=condition,
                device=device,
                max_batches=args.dae_max_val_batches,
            )

            val_losses[
                condition["name"]
            ] = val_loss

        mean_val_loss = sum(
            val_losses.values()
        ) / len(val_losses)

        print(
            "\n"
            f"Epoch {epoch + 1} summary\n"
            f"Train masked MSE: "
            f"{train_loss:.6f}\n"
            f"Random val MSE: "
            f"{val_losses['random_light']:.6f}\n"
            f"Spatial val MSE: "
            f"{val_losses['spatial_light']:.6f}\n"
            f"Motion val MSE: "
            f"{val_losses['motion_light']:.6f}\n"
            f"Mean val MSE: "
            f"{mean_val_loss:.6f}\n"
        )

        # Always save the latest checkpoint.
        latest_path = (
            checkpoint_dir
            / "latest.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_losses": val_losses,
                "mean_val_loss": mean_val_loss,
            },
            latest_path,
        )

        # Save the best validation model.
        if mean_val_loss < best_val_loss:

            best_val_loss = mean_val_loss

            best_path = (
                checkpoint_dir
                / "best.pth"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_losses": val_losses,
                    "mean_val_loss": mean_val_loss,
                },
                best_path,
            )

            print(
                f"New best checkpoint saved: "
                f"{best_path}"
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Train Unified Temporal DAE",
        parents=[get_args_parser()],
    )

    parser.add_argument(
        "--dae_epochs",
        type=int,
        default=10,
        help="Number of Temporal DAE training epochs.",
    )

    parser.add_argument(
        "--dae_lr",
        type=float,
        default=1e-3,
        help="Temporal DAE learning rate.",
    )

    parser.add_argument(
        "--dae_weight_decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )

    parser.add_argument(
        "--dae_resume",
        type=str,
        default="",
        help="Resume Temporal DAE training from a checkpoint.",
    )

    parser.add_argument(
        "--dae_checkpoint_dir",
        type=str,
        default="checkpoints/temporal_dae",
        help="Directory used to save Temporal DAE checkpoints.",
    )

    parser.add_argument(
        "--dae_max_train_batches",
        type=int,
        default=0,
        help=(
            "Limit training batches for debugging. "
            "0 means use all batches."
        ),
    )

    parser.add_argument(
        "--dae_max_val_batches",
        type=int,
        default=0,
        help=(
            "Limit validation batches for debugging. "
            "0 means use all batches."
        ),
    )

    args = parser.parse_args()

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as f:

        config = yaml.load(
            f,
            Loader=yaml.FullLoader,
        )

    main(
        args=args,
        config=config,
    )
