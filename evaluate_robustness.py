import argparse
import random
from collections import defaultdict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from torch.utils.data import DataLoader

import utils
from datasets import S2T_Dataset
from metrics import wer_list
from model import SignLanguageModel
from phoenix_cleanup import clean_phoenix_2014, clean_phoenix_2014_trans
from Tokenizer import GlossTokenizer_S2G
from train import get_args_parser

from missing.random_mask import generate_random_joint_mask
from missing.spatial_mask import generate_spatial_occlusion_mask
from missing.motion_mask import generate_motion_aware_mask
from recovery.interpolation import linear_interpolate_missing
from models.temporal_dae import (
    TemporalDAE,
    replace_missing_keypoints,
)


def apply_missingness(src_input, config, args):
    """
    Apply artificial keypoint missingness before MSKA recognition.

    Returns:
        missing_mask:
            Binary mask [B, T, 79], or None when no missingness is used.
    """

    if args.missing_type == "none":
        return None

    selected_joints = config[
        "model"
    ]["RecognitionNetwork"]["DSTA-Net"]["body"]

    if args.missing_type == "random":

        corrupted_keypoints, missing_mask = generate_random_joint_mask(
            keypoints=src_input["keypoint"],
            selected_joints=selected_joints,
            missing_ratio=args.missing_ratio,
            valid_lengths=src_input["src_length"],
            sample_names=src_input["name"],
            seed=args.mask_seed,
        )

    elif args.missing_type == "spatial":

        corrupted_keypoints, missing_mask = generate_spatial_occlusion_mask(
            keypoints=src_input["keypoint"],
            selected_joints=selected_joints,
            occlusion_width=args.occlusion_width,
            occlusion_height=args.occlusion_height,
            valid_lengths=src_input["src_length"],
            sample_names=src_input["name"],
            seed=args.mask_seed,
        )

    elif args.missing_type == "motion":

        corrupted_keypoints, missing_mask = generate_motion_aware_mask(
            keypoints=src_input["keypoint"],
            selected_joints=selected_joints,
            missing_ratio=args.missing_ratio,
            valid_lengths=src_input["src_length"],
            sample_names=src_input["name"],
            seed=args.mask_seed,
            temperature=args.temperature,
        )

    else:
        raise ValueError(
            f"Unsupported missing type: {args.missing_type}"
        )

    src_input["keypoint"] = corrupted_keypoints

    return missing_mask

def apply_recovery(
    src_input,
    missing_mask,
    config,
    args,
    temporal_dae=None,
):
    """
    Apply keypoint recovery after artificial missingness.

    Recovery is applied only to artificially missing locations.
    """

    if args.recovery == "none":
        return

    if missing_mask is None:
        raise ValueError(
            "A recovery method requires an active missingness condition."
        )

    selected_joints = config[
        "model"
    ]["RecognitionNetwork"]["DSTA-Net"]["body"]

    # --------------------------------------------------
    # Linear interpolation
    # --------------------------------------------------

    if args.recovery == "interpolation":

        recovered_keypoints = linear_interpolate_missing(
            keypoints=src_input["keypoint"],
            missing_mask=missing_mask,
            selected_joints=selected_joints,
            valid_lengths=src_input["src_length"],
        )

        src_input["keypoint"] = recovered_keypoints
        return

    # --------------------------------------------------
    # Unified Temporal DAE
    # --------------------------------------------------

    if args.recovery == "temporal_dae":

        if temporal_dae is None:
            raise ValueError(
                "Temporal DAE model has not been loaded."
            )

        dae_device = next(
            temporal_dae.parameters()
        ).device

        # Extract only the 79 joints used by MSKA.
        # Shape: [B, 3, T, 79]
        corrupted_selected = src_input[
            "keypoint"
        ][
            :,
            :,
            :,
            selected_joints,
        ].to(dae_device)

        mask_on_device = missing_mask.to(
            dae_device
        )

        valid_lengths = torch.as_tensor(
            src_input["src_length"],
            dtype=torch.long,
            device=dae_device,
        )

        # Predict all selected keypoints.
        predictions = temporal_dae(
            keypoints=corrupted_selected,
            missing_mask=mask_on_device,
            valid_lengths=valid_lengths,
        )

        # Preserve all observed values exactly.
        # Only Mask=False positions use DAE predictions.
        recovered_selected = replace_missing_keypoints(
            corrupted_keypoints=corrupted_selected,
            predictions=predictions,
            missing_mask=mask_on_device,
        )

        # Put the recovered 79 joints back into the
        # original 133-joint tensor.
        recovered_full = src_input[
            "keypoint"
        ].clone()

        recovered_full[
            :,
            :,
            :,
            selected_joints,
        ] = recovered_selected.to(
            recovered_full.device
        )

        src_input["keypoint"] = recovered_full
        return

    raise ValueError(
        f"Unsupported recovery method: {args.recovery}"
    )


def evaluate_robustness(
    args,
    config,
    dataloader,
    model,
    tokenizer,
    temporal_dae=None,
):
    """
    Evaluate MSKA under a specified keypoint missingness condition.
    """

    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Robustness evaluation:"
    print_freq = 10

    results = defaultdict(dict)

    with torch.no_grad():

        for step, src_input in enumerate(
            metric_logger.log_every(
                dataloader,
                print_freq,
                header,
            )
        ):

            # --------------------------------------------------
            # Our robustness intervention:
            # corrupt keypoints before they enter MSKA.
            # --------------------------------------------------

            missing_mask = apply_missingness(
                src_input=src_input,
                config=config,
                args=args,
            )

            # --------------------------------------------------
            # Optional keypoint recovery
            # --------------------------------------------------

            apply_recovery(
                src_input=src_input,
                missing_mask=missing_mask,
                config=config,
                args=args,
                temporal_dae=temporal_dae,
            )

            # --------------------------------------------------
            # Original MSKA recognition
            # --------------------------------------------------

            output = model(src_input)

            for key, gloss_logits in output.items():

                if "gloss_logits" not in key:
                    continue

                logits_name = key.replace(
                    "gloss_logits",
                    "",
                )

                ctc_decode_output = (
                    model.recognition_network.decode(
                        gloss_logits=gloss_logits,
                        beam_size=args.beam_size,
                        input_lengths=output["input_lengths"],
                    )
                )

                batch_pred_gls = (
                    tokenizer.convert_ids_to_tokens(
                        ctc_decode_output
                    )
                )

                for name, gls_hyp, gls_ref in zip(
                    src_input["name"],
                    batch_pred_gls,
                    src_input["gloss"],
                ):

                    if tokenizer.lower_case:
                        hypothesis = " ".join(
                            gls_hyp
                        ).upper()

                        reference = gls_ref.upper()

                    else:
                        hypothesis = " ".join(
                            gls_hyp
                        )

                        reference = gls_ref

                    results[name][
                        f"{logits_name}gls_hyp"
                    ] = hypothesis

                    results[name]["gls_ref"] = reference

            metric_logger.update(
                loss=output["total_loss"].item()
            )

    # --------------------------------------------------
    # Calculate WER using the same logic as original MSKA
    # --------------------------------------------------

    evaluation_results = {
        "wer": None
    }

    first_sample_name = next(iter(results))

    for hyp_name in results[first_sample_name].keys():

        if "gls_hyp" not in hyp_name:
            continue

        head_name = hyp_name.replace(
            "gls_hyp",
            "",
        )

        dataset_name = config[
            "data"
        ]["dataset_name"].lower()

        if dataset_name == "phoenix-2014t":

            gls_ref = [
                clean_phoenix_2014_trans(
                    results[name]["gls_ref"]
                )
                for name in results
            ]

            gls_hyp = [
                clean_phoenix_2014_trans(
                    results[name][hyp_name]
                )
                for name in results
            ]

        elif dataset_name == "phoenix-2014":

            gls_ref = [
                clean_phoenix_2014(
                    results[name]["gls_ref"]
                )
                for name in results
            ]

            gls_hyp = [
                clean_phoenix_2014(
                    results[name][hyp_name]
                )
                for name in results
            ]

        else:

            gls_ref = [
                results[name]["gls_ref"]
                for name in results
            ]

            gls_hyp = [
                results[name][hyp_name]
                for name in results
            ]

        wer_results = wer_list(
            hypotheses=gls_hyp,
            references=gls_ref,
        )

        head_wer = wer_results["wer"]

        print(
            f"WER for {head_name or 'main'} head: "
            f"{head_wer:.4f}"
        )

        # Use the same MSKA ensemble output as the primary
        # evaluation metric in every experiment.
        if head_name == "ensemble_last_":
            evaluation_results["wer"] = head_wer

    if evaluation_results["wer"] is None:
        raise RuntimeError(
            "ensemble_last_ WER was not found in the MSKA outputs."
        )

    print(
        f"Primary ensemble WER: "
        f"{evaluation_results['wer']:.4f}"
    )

    metric_logger.update(
        wer=evaluation_results["wer"]
    )

    print("* Averaged stats:", metric_logger)

    return {
        "loss": metric_logger.loss.global_avg,
        "wer": evaluation_results["wer"],
    }


def main(args, config):

    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # Reproducible evaluation.
    seed = args.seed + utils.get_rank()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = False

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    tokenizer = GlossTokenizer_S2G(
        config["gloss"]
    )

    # --------------------------------------------------
    # Select evaluation split
    # --------------------------------------------------

    if args.split == "dev":

        data_path = config[
            "data"
        ]["dev_label_path"]

        phase = "val"

    else:

        data_path = config[
            "data"
        ]["test_label_path"]

        phase = "test"

    dataset = S2T_Dataset(
        path=data_path,
        tokenizer=tokenizer,
        config=config,
        args=args,
        phase=phase,
        training_refurbish=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=args.pin_mem,
    )

    print(
        f"Evaluation split: {args.split}"
    )

    print(
        f"Missing type: {args.missing_type}"
    )

    print(
        f"Missing ratio: {args.missing_ratio}"
    )

    print(
        f"Number of samples: {len(dataset)}"
    )

    # --------------------------------------------------
    # Original MSKA model
    # --------------------------------------------------

    model = SignLanguageModel(
        cfg=config,
        args=args,
    )

    model.to(device)

    if not args.resume:

        raise ValueError(
            "Please provide the MSKA checkpoint "
            "using --resume."
        )

    checkpoint = torch.load(
        args.resume,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    print(
        f"Loaded checkpoint: {args.resume}"
    )

    # --------------------------------------------------
    # Optional Temporal DAE recovery model
    # --------------------------------------------------

    temporal_dae = None

    if args.recovery == "temporal_dae":

        if not args.dae_checkpoint:
            raise ValueError(
                "Please provide the Temporal DAE checkpoint "
                "using --dae_checkpoint."
            )

        temporal_dae = TemporalDAE(
            num_joints=79,
            num_channels=3,
            projection_dim=256,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
        ).to(device)

        dae_checkpoint = torch.load(
            args.dae_checkpoint,
            map_location=device,
        )

        temporal_dae.load_state_dict(
            dae_checkpoint["model"],
            strict=True,
        )

        temporal_dae.eval()

        print(
            f"Loaded Temporal DAE checkpoint: "
            f"{args.dae_checkpoint}"
        )

        print(
            f"Temporal DAE epoch: "
            f"{dae_checkpoint.get('epoch', 'unknown')}"
        )

        print(
            f"Temporal DAE mean val MSE: "
            f"{dae_checkpoint.get('mean_val_loss', float('nan')):.6f}"
        )

    # --------------------------------------------------
    # Robustness evaluation
    # --------------------------------------------------

    stats = evaluate_robustness(
        args=args,
        config=config,
        dataloader=dataloader,
        model=model,
        tokenizer=tokenizer,
        temporal_dae=temporal_dae,
    )

    print(
        f"\nFinal {args.split} WER: "
        f"{stats['wer']:.4f}"
    )

    print(
        f"Final {args.split} loss: "
        f"{stats['loss']:.4f}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "MSKA robustness evaluation",
        parents=[get_args_parser()],
    )

    parser.add_argument(
        "--missing_type",
        type=str,
        default="none",
        choices=["none", "random", "spatial", "motion"],
        help="Type of artificial keypoint missingness.",
    )

    parser.add_argument(
        "--missing_ratio",
        type=float,
        default=0.0,
        help="Proportion of selected MSKA joints to mask.",
    )

    parser.add_argument(
        "--mask_seed",
        type=int,
        default=42,
        help="Seed used to generate reproducible masks.",
    )

    parser.add_argument(
        "--occlusion_width",
        type=float,
        default=0.10,
        help="Width of the normalized spatial occlusion region.",
    )

    parser.add_argument(
        "--occlusion_height",
        type=float,
        default=0.10,
        help="Height of the normalized spatial occlusion region.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Softmax temperature for motion-aware masking.",
    )

    parser.add_argument(
        "--recovery",
        type=str,
        default="none",
        choices=["none", "interpolation", "temporal_dae"],
        help="Keypoint recovery method applied after missingness.",
    )

    parser.add_argument(
        "--dae_checkpoint",
        type=str,
        default="",
        help="Checkpoint used for Temporal DAE recovery.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "test"],
        help="Dataset split used for evaluation.",
    )

    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="CTC decoding beam size.",
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
