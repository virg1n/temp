#!/usr/bin/env python3
"""Idea C adapter for official pretrained DepthSplat / MVSplat-style models.

This replaces the earlier 2D-refiner interpretation of Idea C with a genuine
feed-forward 3D Gaussian path.  It exports the challenge tar into the chunked
format used by DepthSplat/MVSplat, builds a fixed evaluation index, and collects
DepthSplat-rendered PNGs back into a contest submission zip.

DepthSplat itself is kept as an external official repo under
``B/external/depthsplat``.  This file does not reimplement the model.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from geometry_common import CAMERAS, ROOT, TarDataset, alpha_from_meta, save_jpeg


TARGET_BYTES_PER_CHUNK = int(1.5e8)


def image_to_raw_tensor(img: np.ndarray, image_shape: tuple[int, int], quality: int) -> torch.Tensor:
    h, w = image_shape
    pil = Image.fromarray(img)
    if pil.size != (w, h):
        pil = pil.resize((w, h), Image.Resampling.LANCZOS)
    bio = io.BytesIO()
    pil.save(bio, format="JPEG", quality=quality, subsampling=0)
    return torch.from_numpy(np.frombuffer(bio.getvalue(), dtype=np.uint8).copy())


def normalized_camera_vector(meta: dict, timestep: str, camera: str, origin: np.ndarray) -> np.ndarray:
    if timestep == "target":
        c2w = np.asarray(meta["poses_c2w"]["target"][camera], dtype=np.float32).copy()
    else:
        c2w = np.asarray(meta["poses_c2w"][timestep][camera], dtype=np.float32).copy()
    c2w[:3, 3] -= origin.astype(np.float32)
    w2c = np.linalg.inv(c2w)

    intr = meta["intrinsics"][camera]
    width = float(intr["width"])
    height = float(intr["height"])
    return np.asarray(
        [
            float(intr["fx"]) / width,
            float(intr["fy"]) / height,
            float(intr["cx"]) / width,
            float(intr["cy"]) / height,
            0.0,
            0.0,
            *w2c[:3].reshape(-1).tolist(),
        ],
        dtype=np.float32,
    )


def context_views(meta: dict, mode: str) -> list[tuple[str, str]]:
    target_camera = meta["target_camera"]
    if mode == "target_pair":
        return [("t0", target_camera), ("t1", target_camera)]
    if mode == "all_12":
        return [(ts, cam) for ts in ("t0", "t1") for cam in CAMERAS]
    if mode == "all_12_time_interleaved":
        return [(ts, cam) for cam in CAMERAS for ts in ("t0", "t1")]
    raise ValueError(f"Unknown context mode: {mode}")


def choose_origin(meta: dict, ctx: list[tuple[str, str]]) -> np.ndarray:
    # Keep the world numerically local for neural rendering/rasterization.
    target_cam = meta["target_camera"]
    target_origin = np.asarray(meta["poses_c2w"]["target"][target_cam], dtype=np.float32)[:3, 3]
    if np.isfinite(target_origin).all():
        return target_origin.copy()
    ts, cam = ctx[0]
    return np.asarray(meta["poses_c2w"][ts][cam], dtype=np.float32)[:3, 3].copy()


def read_target_or_dummy(ds: TarDataset, split: str, sample_id: str, camera: str) -> np.ndarray:
    if split == "train":
        try:
            return ds.read_image(f"{ROOT}/{split}/{sample_id}/target/{camera}.jpg")
        except KeyError:
            pass
    # Test has no GT.  The official loaders require bytes for the target image,
    # but the tensor is ignored when test.compute_scores=false.
    return ds.read_image(f"{ROOT}/{split}/{sample_id}/input/t0/{camera}.jpg")


def build_example(
    ds: TarDataset,
    split: str,
    sample_id: str,
    context_mode: str,
    image_shape: tuple[int, int],
    quality: int,
) -> tuple[dict, dict]:
    meta = ds.meta(split, sample_id)
    target_camera = meta["target_camera"]
    ctx = context_views(meta, context_mode)
    origin = choose_origin(meta, ctx)

    images: list[torch.Tensor] = []
    cameras: list[np.ndarray] = []
    view_names: list[str] = []

    for timestep, camera in ctx:
        img = ds.read_image(f"{ROOT}/{split}/{sample_id}/input/{timestep}/{camera}.jpg")
        images.append(image_to_raw_tensor(img, image_shape, quality))
        cameras.append(normalized_camera_vector(meta, timestep, camera, origin))
        view_names.append(f"{timestep}/{camera}")

    target_img = read_target_or_dummy(ds, split, sample_id, target_camera)
    images.append(image_to_raw_tensor(target_img, image_shape, quality))
    cameras.append(normalized_camera_vector(meta, "target", target_camera, origin))
    view_names.append(f"target/{target_camera}")

    example = {
        "url": "",
        "timestamps": torch.arange(len(images), dtype=torch.int64),
        "cameras": torch.tensor(np.stack(cameras), dtype=torch.float32),
        "images": images,
        "key": sample_id,
    }
    target_intr = meta["intrinsics"][target_camera]
    index_entry = {
        "context": tuple(range(len(ctx))),
        "target": (len(ctx),),
        "target_camera": target_camera,
        "context_views": view_names[:-1],
        "alpha": alpha_from_meta(meta),
        "target_size": [int(target_intr["height"]), int(target_intr["width"])],
    }
    return example, index_entry


def save_chunk(chunk: list[dict], out_stage: Path, chunk_index: int, index: dict[str, str]) -> int:
    chunk_key = f"{chunk_index:0>6}.torch"
    out_stage.mkdir(parents=True, exist_ok=True)
    torch.save(chunk, out_stage / chunk_key)
    for example in chunk:
        index[example["key"]] = chunk_key
    return chunk_index + 1


def export_split(args: argparse.Namespace) -> None:
    image_shape = (int(args.image_height), int(args.image_width))
    out_root = Path(args.output_root)
    out_stage = out_root / args.split
    out_stage.mkdir(parents=True, exist_ok=True)

    ds = TarDataset(args.dataset_tar)
    try:
        sample_ids = ds.sample_ids(args.split)
        if args.limit:
            sample_ids = sample_ids[: args.limit]

        chunk: list[dict] = []
        chunk_bytes = 0
        chunk_index = 0
        chunk_index_map: dict[str, str] = {}
        eval_index: dict[str, dict] = {}
        manifest: dict[str, dict] = {}

        for i, sample_id in enumerate(sample_ids, 1):
            example, entry = build_example(
                ds,
                args.split,
                sample_id,
                args.context_mode,
                image_shape,
                args.jpeg_quality,
            )
            approx_bytes = sum(int(x.numel()) for x in example["images"])
            chunk.append(example)
            chunk_bytes += approx_bytes
            eval_index[sample_id] = {"context": entry["context"], "target": entry["target"]}
            manifest[sample_id] = entry

            if chunk_bytes >= TARGET_BYTES_PER_CHUNK:
                chunk_index = save_chunk(chunk, out_stage, chunk_index, chunk_index_map)
                chunk = []
                chunk_bytes = 0

            if i == 1 or i % args.log_every == 0 or i == len(sample_ids):
                print(f"export {args.split} {i}/{len(sample_ids)} {sample_id}")

        if chunk:
            save_chunk(chunk, out_stage, chunk_index, chunk_index_map)

        (out_stage / "index.json").write_text(json.dumps(chunk_index_map, indent=2), encoding="utf-8")
        eval_path = out_stage / f"evaluation_index_{args.context_mode}.json"
        eval_path.write_text(json.dumps(eval_index, indent=2), encoding="utf-8")
        manifest_path = out_stage / f"manifest_{args.context_mode}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote {out_stage}")
        print(f"Wrote {eval_path}")
    finally:
        ds.close()


def collect_outputs(args: argparse.Namespace) -> None:
    output_dir = Path(args.depthsplat_output_dir)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    sample_ids = sorted(manifest)
    missing: list[Path] = []

    for i, sample_id in enumerate(sample_ids, 1):
        entry = manifest[sample_id]
        target_index = int(entry["target"][0])
        png_path = output_dir / "images" / sample_id / "color" / f"{target_index:0>6}.png"
        if not png_path.exists():
            missing.append(png_path)
            continue
        target_size = entry.get("target_size")
        with Image.open(png_path) as img:
            pil = img.convert("RGB")
            if target_size is not None:
                target_h, target_w = int(target_size[0]), int(target_size[1])
                if pil.size != (target_w, target_h):
                    pil = pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
            pred = np.array(pil, copy=True)
        save_jpeg(out_dir / sample_id / "pred.jpg", pred, quality=args.jpeg_quality)
        if i == 1 or i % args.log_every == 0 or i == len(sample_ids):
            print(f"collect {i}/{len(sample_ids)} {sample_id}")

    if missing:
        preview = "\n".join(str(p) for p in missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} rendered outputs. First missing paths:\n{preview}")

    if args.zip_path:
        with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for sample_id in sample_ids:
                zf.write(out_dir / sample_id / "pred.jpg", arcname=f"submission/{sample_id}/pred.jpg")
        print(f"Wrote {args.zip_path}")


def write_depthsplat_commands(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    rel_root = args.depthsplat_dataset_root
    ctx = args.context_mode
    num_views = 12 if ctx.startswith("all_12") else 2

    if num_views == 12:
        experiment = "dl3dv"
        checkpoint = "pretrained/depthsplat-gs-small-re10kdl3dv-448x768-randview4-10-c08188db.pth"
        image_shape = "[512,960]"
        ori_shape = "[512,960]"
        extra_model = (
            "model.encoder.upsample_factor=8 "
            "model.encoder.lowest_feature_resolution=8 "
            "model.encoder.gaussian_adapter.gaussian_scale_max=0.1"
        )
        batch_size = 1
    else:
        experiment = "re10k"
        checkpoint = "pretrained/depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth"
        image_shape = "[256,256]"
        ori_shape = "[256,256]"
        extra_model = (
            "model.encoder.num_scales=2 "
            "model.encoder.upsample_factor=2 "
            "model.encoder.lowest_feature_resolution=4 "
            "model.encoder.monodepth_vit_type=vitb"
        )
        batch_size = 4

    train_index = f"{rel_root}/train/evaluation_index_{ctx}.json"
    test_index = f"{rel_root}/test/evaluation_index_{ctx}.json"
    out = Path(args.command_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Run this script from the repository root.
# Download the checkpoint listed below into B/external/depthsplat/pretrained first.

Push-Location B/external/depthsplat

# Fine-tune on the challenge train split using the official pretrained Gaussian model.
python -m src.main +experiment={experiment} mode=train `
  dataset.roots=[{rel_root}] `
  dataset/view_sampler=evaluation `
  dataset.view_sampler.num_context_views={num_views} `
  dataset.view_sampler.index_path={train_index} `
  dataset.image_shape={image_shape} `
  dataset.ori_image_shape={ori_shape} `
  dataset.skip_bad_shape=false `
  dataset.near=0.5 dataset.far=180.0 `
  data_loader.train.batch_size={batch_size} data_loader.train.num_workers=4 `
  trainer.max_steps=30000 trainer.val_check_interval=2000 `
  checkpointing.pretrained_model={checkpoint} `
  checkpointing.resume=false `
  {extra_model} `
  output_dir=outputs/yandex_depthsplat_finetune_{ctx}

# Render the challenge test split. Replace checkpointing.pretrained_model with
# the fine-tuned checkpoint path after training if desired.
python -m src.main +experiment={experiment} mode=test `
  dataset.roots=[{rel_root}] `
  dataset/view_sampler=evaluation `
  dataset.view_sampler.num_context_views={num_views} `
  dataset.view_sampler.index_path={test_index} `
  dataset.image_shape={image_shape} `
  dataset.ori_image_shape={ori_shape} `
  dataset.skip_bad_shape=false `
  dataset.near=0.5 dataset.far=180.0 `
  test.save_image=true test.compute_scores=false test.render_chunk_size=1 `
  checkpointing.pretrained_model={checkpoint} `
  {extra_model} `
  output_dir=outputs/yandex_depthsplat_test_{ctx}

Pop-Location

# Convert rendered PNGs to the contest zip.
python B/solution_C_depthsplat_adapter.py collect `
  --depthsplat-output-dir B/external/depthsplat/outputs/yandex_depthsplat_test_{ctx} `
  --manifest B/depthsplat_yandex/test/manifest_{ctx}.json `
  --out-dir B/submission_C `
  --zip-path B/submission_C.zip
"""
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Dataset root for DepthSplat commands: {root}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export/collect task B data for official DepthSplat Idea C")
    sub = p.add_subparsers(dest="cmd", required=True)

    export = sub.add_parser("export")
    export.add_argument("--dataset-tar", default="B/cv_dataset.tar")
    export.add_argument("--output-root", default="B/depthsplat_yandex")
    export.add_argument("--split", choices=("train", "test"), required=True)
    export.add_argument("--context-mode", choices=("target_pair", "all_12", "all_12_time_interleaved"), default="all_12")
    export.add_argument("--image-height", type=int, default=512)
    export.add_argument("--image-width", type=int, default=960)
    export.add_argument("--jpeg-quality", type=int, default=95)
    export.add_argument("--limit", type=int, default=0)
    export.add_argument("--log-every", type=int, default=25)

    commands = sub.add_parser("commands")
    commands.add_argument("--output-root", default="B/depthsplat_yandex")
    commands.add_argument("--depthsplat-dataset-root", default="../../depthsplat_yandex")
    commands.add_argument("--context-mode", choices=("target_pair", "all_12", "all_12_time_interleaved"), default="all_12")
    commands.add_argument("--command-file", default="B/run_C_depthsplat.ps1")

    collect = sub.add_parser("collect")
    collect.add_argument("--depthsplat-output-dir", required=True)
    collect.add_argument("--manifest", default="B/depthsplat_yandex/test/manifest_all_12.json")
    collect.add_argument("--out-dir", default="B/submission_C")
    collect.add_argument("--zip-path", default="B/submission_C.zip")
    collect.add_argument("--jpeg-quality", type=int, default=95)
    collect.add_argument("--log-every", type=int, default=25)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "export":
        export_split(args)
    elif args.cmd == "commands":
        write_depthsplat_commands(args)
    elif args.cmd == "collect":
        collect_outputs(args)
    else:
        raise AssertionError(args.cmd)


if __name__ == "__main__":
    main()
