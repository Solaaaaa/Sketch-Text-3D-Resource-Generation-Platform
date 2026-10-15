# Sketch to Urban Scene and 3D Resource Generation

A four-stage prototype for turning an architectural sketch and text into urban
scene images, local edits, multiple views and 3D resources:

1. **Sketch + text → image:** FLUX.1-dev and the author's VisDrone ControlNet.
2. **Local editing:** FLUX.1-Fill-dev with a mask.
3. **Multiple views:** Qwen Image Edit through ComfyUI.
4. **Image → 3D:** TRELLIS-image-large, Gaussian/mesh preview and GLB/PLY export.

`app.py` is the interface; `trellis_step4_runner.py` is the project's adapter.
The historical thesis title is retained in [project context](docs/PROJECT_CONTEXT.md).
Data/training utilities are under `scripts/`; datasets are not needed for inference.

## New-server setup

Follow [deploy/README.md](deploy/README.md) for the Linux Python/CUDA environment,
exact source revisions, dependency installation, auxiliary resources, `.env`,
file checks and launch commands. It distinguishes observed working versions
from proposed installation steps that still require new-server GPU validation.

Obtain primary weights under their upstream terms. Public ModelScope metadata,
model cards and required directory layouts were checked on 2026-10-15; the
following commands are for the new server, after installing the ModelScope CLI:

```bash
modelscope download --model black-forest-labs/FLUX.1-dev \
  --local_dir ./models/FLUX.1-dev
modelscope download --model Solathl/flux1-dev-visdrone-urban-sketch-controlnet \
  --local_dir ./models/flux1-dev-visdrone-urban-sketch-controlnet
modelscope download --model AI-ModelScope/FLUX.1-Fill-dev \
  --local_dir ./models/FLUX.1-Fill-dev
modelscope download --model AI-ModelScope/TRELLIS-image-large \
  --local_dir ./models/TRELLIS-image-large

export SKETCH_PLATFORM_TRELLIS_MODEL=./models/TRELLIS-image-large
```

FLUX needs complete Diffusers directories, including both text encoders,
tokenizers, transformer, VAE and scheduler. ControlNet requires `config.json`
and `diffusion_pytorch_model.safetensors`; ModelScope's `configuration.json`
metadata does not replace the Diffusers configuration.
[MODEL_MANIFEST.md](docs/MODEL_MANIFEST.md) maps resources to files, variables and
sources. Exact downloaded model revisions and weight SHA256 remain unrecorded.

## Qwen / ComfyUI

Obtain the third-party main program and selected model resources using the
introduction page <https://deepface.cc/thread-925-1-1.html> and its UC disk entries.
Extract so the final path is:

```text
external/ComfyUI_Qwen-Multiple-Angles/ComfyUI/main.py
```

`SKETCH_PLATFORM_QWEN_ROOT` points to the directory containing `ComfyUI/`, not to
`ComfyUI/` itself. For an extra archive nesting level, locate `ComfyUI/main.py`
and configure that parent; absolute paths are supported. Use a Linux environment,
not a bundled Windows Python. The app starts ComfyUI on demand with the app's
Python, or reuses an already ready service; normally no separate launch is needed.

Choose **GGUF or Diffusion**, not both. Both choices need the specified text
encoder, VAE, angle LoRA and 8-step Lightning LoRA. GGUF additionally requires
ComfyUI-GGUF. See [the exact file/node checklist](docs/THIRD_PARTY_QWEN_BUNDLE.md),
including a file-only check. This project does not mirror or upload the bundle or
custom nodes. Archive contents and each component's provenance/terms must be
checked by the recipient; a share link is not redistribution permission.

## TRELLIS runtime and weights

`external/TRELLIS/trellis/` is an attributed snapshot of third-party
[microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) source at `eeacb0b`, with
local patches documented in [THIRD_PARTY_TRELLIS.md](docs/THIRD_PARTY_TRELLIS.md).
It includes the complete Python package and FlexiCubes runtime. The TRELLIS
root MIT license, FlexiCubes NVIDIA license, Gaussian-derived code licenses and
inline notices are retained separately. FlexiCubes and Gaussian-derived code
carry non-commercial research/evaluation restrictions.

TRELLIS-image-large weights are not included. Stage 4 strictly uses the local
`models/TRELLIS-image-large` directory or `SKETCH_PLATFORM_TRELLIS_MODEL` override.
It validates `pipeline.json` and all referenced JSON/safetensors files with path
containment checks. It does not use Hugging Face, ModelScope, or other caches as
the primary model source. DINOv2/U2Net are separate auxiliary resources: prepare
them on the new server following the deployment guide. Their caches below
`SKETCH_PLATFORM_CACHE_ROOT` are not a primary TRELLIS model source.

CUDA, PyTorch, spconv, nvdiffrast, Mip-Splatting's Gaussian rasterizer, utils3d,
Kaolin and rembg/ONNX Runtime remain external dependencies. `vox2seq` is optional
for serialized attention, not needed by the audited default model; its source is
excluded because the upstream extension's license coverage remains unresolved.
This project does not claim every optional TRELLIS mode is supported.

## Validation status and publication boundary

October 12 GPU testing completed Stages 1–3 and wrote Stage 4 video/state before
Gradio rejected an external output directory. The candidate now preserves the
`allowed_paths` repair, which passed the October 15 CPU file-return regression.
Complete repaired GPU/browser validation, including new GLB/PLY exports, remains
outstanding on the new server.

The agreed order is preparation → parent review → user manual GitHub upload →
new GPU server clone/resource preparation → full four-stage validation. The
current machine's occupied GPU does not block preparation or manual publication.
Only the candidate repository contents belong in GitHub; surrounding staging
audit directories, backups and test fixtures do not.

No weights, datasets, captions, generated media, caches, checkpoints, virtual
environments or credentials are included. `.gitignore` excludes the default
local resource layout while retaining the TRELLIS source snapshot.

## License

Author-owned code and documentation use [PolyForm Noncommercial 1.0.0](LICENSE).
It does not relicense TRELLIS, FlexiCubes, Gaussian-derived code, FLUX, Qwen,
ComfyUI, Diffusers, custom nodes, weights or datasets. Each resource's terms
remain applicable; see the third-party notices and model manifest.
