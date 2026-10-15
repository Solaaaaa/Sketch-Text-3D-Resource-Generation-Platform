#!/usr/bin/env python3
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("SPCONV_ALGO", "native")

PROJECT_ROOT = Path(__file__).resolve().parent
configured_trellis_root = Path(
    os.environ.get("SKETCH_PLATFORM_TRELLIS_ROOT", "external/TRELLIS")
).expanduser()
TRELLIS_PROJECT_ROOT = (
    configured_trellis_root
    if configured_trellis_root.is_absolute()
    else PROJECT_ROOT / configured_trellis_root
)
RESULT_PREFIX = "__TRELLIS_RESULT__="
PIPELINE = None
PIPELINE_SOURCE = None


def emit(payload: dict) -> None:
    print(f"{RESULT_PREFIX}{json.dumps(payload, ensure_ascii=False)}", flush=True)


def import_trellis_runtime() -> dict:
    if str(TRELLIS_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(TRELLIS_PROJECT_ROOT))

    import imageio
    import numpy as np
    import torch
    from PIL import Image
    from easydict import EasyDict as edict
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.representations import Gaussian
    from trellis.utils import postprocessing_utils, render_utils

    return {
        "imageio": imageio,
        "np": np,
        "torch": torch,
        "Image": Image,
        "edict": edict,
        "TrellisImageTo3DPipeline": TrellisImageTo3DPipeline,
        "Gaussian": Gaussian,
        "postprocessing_utils": postprocessing_utils,
        "render_utils": render_utils,
    }


def validate_local_model_source(model_source: str) -> str:
    model_path = Path(model_source).expanduser()
    pipeline_path = model_path / "pipeline.json"
    if not model_path.is_dir():
        raise RuntimeError(
            f"TRELLIS 模型目录不存在：{model_path}。"
            "第四阶段只接受本地 TRELLIS-image-large 目录。"
        )
    model_root = model_path.resolve()
    try:
        pipeline_path.resolve().relative_to(model_root)
    except ValueError as exc:
        raise RuntimeError(
            f"TRELLIS pipeline.json 路径越界：{pipeline_path}。"
        ) from exc
    if not pipeline_path.is_file():
        raise RuntimeError(
            f"TRELLIS 模型目录缺少 pipeline.json：{pipeline_path}。"
            "不会从远程仓库或缓存补充模型。"
        )

    try:
        with pipeline_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"TRELLIS pipeline.json 无法读取或解析：{pipeline_path}。"
        ) from exc

    args = config.get("args") if isinstance(config, dict) else None
    model_names = args.get("models") if isinstance(args, dict) else None
    if not isinstance(model_names, dict) or not model_names:
        raise RuntimeError(
            f"TRELLIS pipeline.json 未包含有效的 models 配置：{pipeline_path}。"
        )

    missing = []
    for model_name in model_names.values():
        if not isinstance(model_name, str) or Path(model_name).is_absolute():
            missing.append(str(model_name))
            continue
        model_prefix = (model_path / model_name).resolve()
        try:
            model_prefix.relative_to(model_root)
        except ValueError:
            missing.append(f"路径越界：{model_name}")
            continue
        for suffix in (".json", ".safetensors"):
            model_file = model_prefix.with_name(f"{model_prefix.name}{suffix}")
            try:
                model_file.resolve().relative_to(model_root)
            except ValueError:
                missing.append(f"路径越界：{model_file}")
                continue
            if not model_file.is_file():
                missing.append(str(model_file))
    if missing:
        raise RuntimeError(
            "TRELLIS 本地模型文件不完整；不会从远程仓库或缓存补充：\n"
            + "\n".join(missing)
        )

    return str(model_path.resolve())


def load_pipeline(runtime: dict, model_source: str):
    global PIPELINE, PIPELINE_SOURCE
    if PIPELINE is None or PIPELINE_SOURCE != model_source:
        pipeline_cls = runtime["TrellisImageTo3DPipeline"]
        PIPELINE = pipeline_cls.from_pretrained(model_source)
        PIPELINE.cuda()
        PIPELINE_SOURCE = model_source
    return PIPELINE


def pack_state(runtime: dict, gaussian, mesh) -> dict:
    return {
        "gaussian": {
            **gaussian.init_params,
            "_xyz": gaussian._xyz.detach().cpu().numpy(),
            "_features_dc": gaussian._features_dc.detach().cpu().numpy(),
            "_scaling": gaussian._scaling.detach().cpu().numpy(),
            "_rotation": gaussian._rotation.detach().cpu().numpy(),
            "_opacity": gaussian._opacity.detach().cpu().numpy(),
        },
        "mesh": {
            "vertices": mesh.vertices.detach().cpu().numpy(),
            "faces": mesh.faces.detach().cpu().numpy(),
        },
    }


def unpack_state(runtime: dict, state: dict):
    torch = runtime["torch"]
    Gaussian = runtime["Gaussian"]
    edict = runtime["edict"]

    gaussian = Gaussian(
        aabb=state["gaussian"]["aabb"],
        sh_degree=state["gaussian"]["sh_degree"],
        mininum_kernel_size=state["gaussian"]["mininum_kernel_size"],
        scaling_bias=state["gaussian"]["scaling_bias"],
        opacity_bias=state["gaussian"]["opacity_bias"],
        scaling_activation=state["gaussian"]["scaling_activation"],
    )
    gaussian._xyz = torch.tensor(state["gaussian"]["_xyz"], device="cuda")
    gaussian._features_dc = torch.tensor(state["gaussian"]["_features_dc"], device="cuda")
    gaussian._scaling = torch.tensor(state["gaussian"]["_scaling"], device="cuda")
    gaussian._rotation = torch.tensor(state["gaussian"]["_rotation"], device="cuda")
    gaussian._opacity = torch.tensor(state["gaussian"]["_opacity"], device="cuda")

    mesh = edict(
        vertices=torch.tensor(state["mesh"]["vertices"], device="cuda"),
        faces=torch.tensor(state["mesh"]["faces"], device="cuda"),
    )
    return gaussian, mesh


def generate(args: argparse.Namespace) -> dict:
    model_source = validate_local_model_source(args.model_source)
    runtime = import_trellis_runtime()
    torch = runtime["torch"]
    np = runtime["np"]
    imageio = runtime["imageio"]
    Image = runtime["Image"]
    render_utils = runtime["render_utils"]

    job_dir = Path(args.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for image_path in args.images:
        with Image.open(image_path) as image:
            images.append(image.convert("RGBA").copy())

    pipeline = load_pipeline(runtime, model_source)
    outputs = pipeline.run_multi_image(
        images,
        seed=args.seed,
        formats=["gaussian", "mesh"],
        preprocess_image=True,
        sparse_structure_sampler_params={
            "steps": args.ss_steps,
            "cfg_strength": args.ss_guidance,
        },
        slat_sampler_params={
            "steps": args.slat_steps,
            "cfg_strength": args.slat_guidance,
        },
        mode=args.multiimage_algo,
    )

    video_gs = render_utils.render_video(outputs["gaussian"][0], num_frames=120)["color"]
    video_mesh = render_utils.render_video(outputs["mesh"][0], num_frames=120)["normal"]
    video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]

    video_path = job_dir / "preview.mp4"
    state_path = job_dir / "state.pt"
    imageio.mimsave(video_path, video, format="FFMPEG", fps=15, codec="libx264")
    torch.save(pack_state(runtime, outputs["gaussian"][0], outputs["mesh"][0]), state_path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "video_path": str(video_path),
        "state_path": str(state_path),
    }


def extract_glb(args: argparse.Namespace) -> dict:
    runtime = import_trellis_runtime()
    torch = runtime["torch"]
    postprocessing_utils = runtime["postprocessing_utils"]

    state = torch.load(args.state_path, map_location="cpu", weights_only=False)
    gaussian, mesh = unpack_state(runtime, state)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    glb = postprocessing_utils.to_glb(
        gaussian,
        mesh,
        simplify=args.mesh_simplify,
        texture_size=args.texture_size,
        verbose=False,
    )
    glb.export(output_path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"glb_path": str(output_path)}


def extract_gaussian(args: argparse.Namespace) -> dict:
    runtime = import_trellis_runtime()
    torch = runtime["torch"]

    state = torch.load(args.state_path, map_location="cpu", weights_only=False)
    gaussian, _ = unpack_state(runtime, state)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gaussian.save_ply(str(output_path))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"gaussian_path": str(output_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TRELLIS Step 4 helper runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--job-dir", required=True)
    generate_parser.add_argument("--model-source", required=True)
    generate_parser.add_argument("--image", dest="images", action="append", required=True)
    generate_parser.add_argument("--seed", type=int, required=True)
    generate_parser.add_argument("--ss-guidance", type=float, required=True)
    generate_parser.add_argument("--ss-steps", type=int, required=True)
    generate_parser.add_argument("--slat-guidance", type=float, required=True)
    generate_parser.add_argument("--slat-steps", type=int, required=True)
    generate_parser.add_argument("--multiimage-algo", choices=["stochastic", "multidiffusion"], required=True)

    extract_glb_parser = subparsers.add_parser("extract-glb")
    extract_glb_parser.add_argument("--state-path", required=True)
    extract_glb_parser.add_argument("--output-path", required=True)
    extract_glb_parser.add_argument("--mesh-simplify", type=float, required=True)
    extract_glb_parser.add_argument("--texture-size", type=int, required=True)

    extract_gaussian_parser = subparsers.add_parser("extract-gaussian")
    extract_gaussian_parser.add_argument("--state-path", required=True)
    extract_gaussian_parser.add_argument("--output-path", required=True)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "generate":
            payload = generate(args)
        elif args.command == "extract-glb":
            payload = extract_glb(args)
        else:
            payload = extract_gaussian(args)
        emit({"ok": True, **payload})
        return 0
    except Exception as exc:
        emit({
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
