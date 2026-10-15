#!/usr/bin/env python3
"""Check local resource files without model imports, network access or GPU calls."""
import argparse
import ast
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT_ROOT))
from trellis_step4_runner import validate_local_model_source


def configured(name, default):
    path = Path(os.environ.get(name) or default).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'flux', 'qwen', 'trellis'], default='all')
    parser.add_argument('--qwen-backend', choices=['GGUF', 'Diffusion'], default='GGUF')
    args = parser.parse_args()
    failures = []

    def require(path):
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f'Missing or empty file: {path}')
            return False
        return True

    def contains(path, symbols):
        if require(path):
            text = path.read_text(encoding='utf-8')
            for symbol in symbols:
                if symbol not in text:
                    failures.append(f'Missing source symbol {symbol}: {path}')

    if args.stage in ('all', 'flux'):
        for variable, default in [('SKETCH_PLATFORM_BASE_MODEL', 'models/FLUX.1-dev'),
                                  ('SKETCH_PLATFORM_FILL_MODEL', 'models/FLUX.1-Fill-dev')]:
            root = configured(variable, default)
            require(root / 'model_index.json')
            require(root / 'scheduler/scheduler_config.json')
            for folder in ('tokenizer', 'tokenizer_2'):
                require(root / folder / 'tokenizer_config.json')
            require(root / 'tokenizer/vocab.json')
            require(root / 'tokenizer/merges.txt')
            require(root / 'tokenizer_2/spiece.model')
            for folder in ('transformer', 'vae', 'text_encoder', 'text_encoder_2'):
                require(root / folder / 'config.json')
                weights = list((root / folder).glob('*.safetensors'))
                if not weights:
                    failures.append(f'No safetensors in {root / folder}')
                for weight in weights:
                    require(weight)
                for index in (root / folder).glob('*.safetensors.index.json'):
                    for filename in set(json.loads(index.read_text())['weight_map'].values()):
                        require(index.parent / filename)
        controlnet = configured('SKETCH_PLATFORM_CONTROLNET_MODEL',
                               'models/flux1-dev-visdrone-urban-sketch-controlnet')
        require(controlnet / 'config.json')
        require(controlnet / 'diffusion_pytorch_model.safetensors')

    if args.stage in ('all', 'qwen'):
        # Read workflow filenames from app source, without importing the Gradio app.
        values = {}
        for node in ast.parse((PROJECT_ROOT / 'app.py').read_text()).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith('MULTIVIEW_'):
                        values[target.id] = ast.literal_eval(node.value)
        root = configured('SKETCH_PLATFORM_QWEN_ROOT', 'external/ComfyUI_Qwen-Multiple-Angles') / 'ComfyUI'
        require(root / 'main.py')
        require(root / 'requirements.txt')
        for folder, key in [('text_encoders', 'MULTIVIEW_CLIP_MODEL'), ('vae', 'MULTIVIEW_VAE_MODEL'),
                            ('loras', 'MULTIVIEW_ANGLE_LORA'), ('loras', 'MULTIVIEW_LIGHTNING_LORA')]:
            require(root / 'models' / folder / values[key])
        folder = 'unet' if args.qwen_backend == 'GGUF' else 'diffusion_models'
        require(root / 'models' / folder / values['MULTIVIEW_MODEL_CHOICES'][args.qwen_backend][0])
        contains(root / 'nodes.py', ['LoraLoaderModelOnly', 'CLIPLoader', 'VAELoader', 'LoadImage',
                                     'UNETLoader', 'VAEEncode', 'KSampler', 'VAEDecode', 'SaveImage'])
        for filename, symbol in [('nodes_qwen.py', 'TextEncodeQwenImageEditPlus'),
                                 ('nodes_flux.py', 'FluxKontextMultiReferenceLatentMethod'),
                                 ('nodes_cfg.py', 'CFGNorm'),
                                 ('nodes_model_advanced.py', 'ModelSamplingAuraFlow')]:
            contains(root / 'comfy_extras' / filename, [symbol])
        if args.qwen_backend == 'GGUF':
            contains(root / 'custom_nodes/ComfyUI-GGUF/nodes.py', ['UnetLoaderGGUF'])
            require(root / 'custom_nodes/ComfyUI-GGUF/requirements.txt')

    if args.stage in ('all', 'trellis'):
        runtime = configured('SKETCH_PLATFORM_TRELLIS_ROOT', 'external/TRELLIS')
        for path in ('trellis/pipelines/trellis_image_to_3d.py', 'trellis/models/__init__.py',
                     'trellis/representations/mesh/flexicubes/flexicubes.py',
                     'trellis/representations/mesh/flexicubes/LICENSE.txt'):
            require(runtime / path)
        require(configured('SKETCH_PLATFORM_TRELLIS_RUNNER', 'trellis_step4_runner.py'))
        model = configured('SKETCH_PLATFORM_TRELLIS_MODEL', 'models/TRELLIS-image-large')
        try:
            validate_local_model_source(str(model))
        except (RuntimeError, OSError) as exc:
            failures.append(str(exc))
        cache = configured('SKETCH_PLATFORM_CACHE_ROOT', 'cache')
        require(cache / 'torch/hub/facebookresearch_dinov2_main/hubconf.py')
        require(cache / 'torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth')
        require(cache / 'u2net/u2net.onnx')

    for failure in failures:
        print('FAIL:', failure)
    if failures:
        print(f'{len(failures)} file checks failed. No model was loaded.')
        return 1
    print(f'PASS: {args.stage} resource files ({args.qwen_backend}); no model imports/GPU/network.')
    print('File presence does not validate weight contents, environment compatibility or GPU behavior.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
