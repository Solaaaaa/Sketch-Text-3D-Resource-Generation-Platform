# Model and runtime manifest

Checked 2026-10-15 using small public metadata/model cards and read-only local
source/config inspection. No weights were downloaded. Exact installed model
revision and weight SHA256: **Not recorded yet** for all rows below; public
metadata must not be mistaken for proof that local weight bytes match it.
Paths are relative to the project root unless overridden by an absolute path.
No dataset is needed for four-stage inference.

| Stage | Resource and public source | Default target | Environment variable | Required files |
|---|---|---|---|---|
| 1 | [FLUX.1-dev](https://modelscope.cn/models/black-forest-labs/FLUX.1-dev), `black-forest-labs/FLUX.1-dev` | `models/FLUX.1-dev` | `SKETCH_PLATFORM_BASE_MODEL` | `model_index.json`, scheduler, tokenizer/tokenizer_2, text_encoder/text_encoder_2, transformer and vae directories with configs and full weights |
| 1 | [Author ControlNet](https://modelscope.cn/models/Solathl/flux1-dev-visdrone-urban-sketch-controlnet), `Solathl/flux1-dev-visdrone-urban-sketch-controlnet` | `models/flux1-dev-visdrone-urban-sketch-controlnet` | `SKETCH_PLATFORM_CONTROLNET_MODEL` | `config.json`, `diffusion_pytorch_model.safetensors` |
| 2 | [FLUX.1-Fill-dev](https://modelscope.cn/models/AI-ModelScope/FLUX.1-Fill-dev), `AI-ModelScope/FLUX.1-Fill-dev` | `models/FLUX.1-Fill-dev` | `SKETCH_PLATFORM_FILL_MODEL` | Complete Diffusers layout as above; a single top-level checkpoint is insufficient |
| 3 | Qwen Image Edit, VAE, encoder, angle/Lightning LoRAs; [bundle preparation](THIRD_PARTY_QWEN_BUNDLE.md) | `external/ComfyUI_Qwen-Multiple-Angles/ComfyUI/models` | `SKETCH_PLATFORM_QWEN_ROOT` points to the directory containing `ComfyUI/` | Select GGUF **or** Diffusion; common encoder, VAE and two LoRAs; exact names in Qwen guide |
| 4 | [TRELLIS-image-large](https://modelscope.cn/models/AI-ModelScope/TRELLIS-image-large), `AI-ModelScope/TRELLIS-image-large` | `models/TRELLIS-image-large` | `SKETCH_PLATFORM_TRELLIS_MODEL` | `pipeline.json` and every referenced `.json`/`.safetensors` pair within this directory |
| 4 auxiliary | [DINOv2](https://github.com/facebookresearch/dinov2), `dinov2_vitl14_reg` | `cache/torch/hub/facebookresearch_dinov2_main` and `cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth` | `SKETCH_PLATFORM_CACHE_ROOT` → child `TORCH_HOME` | `hubconf.py` and full source checkout plus weight file |
| 4 auxiliary | [rembg U2Net](https://github.com/danielgatis/rembg/blob/main/rembg/sessions/u2net.py) | `cache/u2net/u2net.onnx` | `SKETCH_PLATFORM_CACHE_ROOT` → child `U2NET_HOME` | `u2net.onnx`; rembg 2.0.72 provider MD5 `60024c5c889badc19c04ad937298a77b` |

Download commands (run from the clone on the new server):

```bash
modelscope download --model black-forest-labs/FLUX.1-dev --local_dir ./models/FLUX.1-dev
modelscope download --model Solathl/flux1-dev-visdrone-urban-sketch-controlnet --local_dir ./models/flux1-dev-visdrone-urban-sketch-controlnet
modelscope download --model AI-ModelScope/FLUX.1-Fill-dev --local_dir ./models/FLUX.1-Fill-dev
modelscope download --model AI-ModelScope/TRELLIS-image-large --local_dir ./models/TRELLIS-image-large
export SKETCH_PLATFORM_TRELLIS_MODEL=./models/TRELLIS-image-large
```

The public ModelScope API returned the named FLUX repositories and full Diffusers
file layouts; model cards point to Black Forest Labs' original Hugging Face
repositories and FLUX non-commercial license/AUP. Direct Hugging Face metadata
requests were unavailable from this audit host. ModelScope is a distribution
source, not a substitute license. Read and accept the provider's terms before
use; no redistribution of weights is authorized by this project's license.
The ControlNet is an author-trained derivative subject to applicable base-model
terms, not an official Black Forest Labs model. `configuration.json` is platform
metadata and does not replace ControlNet's `config.json`.

TRELLIS ModelScope metadata declares MIT; consult the downloaded model card and
component terms rather than inferring weight terms from the runtime license.
Qwen component revisions, original weight sources and licenses are not verified
from the third-party bundle's filenames; see its notice for this remaining limit.
DINOv2 and U2Net retain their own upstream terms and are downloaded separately.
Auxiliary caches are never primary TRELLIS-image-large fallback sources.

## Runtime source (not weights)

| Component | Included? | Source/version | License/provenance |
|---|---|---|---|
| TRELLIS Python package | Yes, `external/TRELLIS/trellis/` | microsoft/TRELLIS `eeacb0bf6a7d25058232d746bef4e5e880b130ff` with recorded patches | MIT for majority; component exceptions in [TRELLIS notice](THIRD_PARTY_TRELLIS.md) |
| FlexiCubes runtime | Yes | MaxtirError/FlexiCubes `f97beb0dd3c6c68f3ab5696b6dcaf9af69f0514e` | Complete NVIDIA LICENSE.txt and attribution retained; non-commercial research/evaluation |
| Gaussian-derived utilities/renderer | Yes | 3DGS/Mip-Splatting provenance documented in TRELLIS notice | Both applicable complete license texts retained under `external/TRELLIS/third_party_licenses`; non-commercial research/evaluation |
| vox2seq | No | fixed TRELLIS extension; only serialized attention needs it | Missing/conflicting license evidence unresolved; optional source moved outside upload candidate |
| ComfyUI / required custom nodes | No | inspected ComfyUI source 0.6.0, GGUF 2.0.0 without exact local revisions | Local core LICENSE is GPL-3.0; ComfyUI-GGUF LICENSE is Apache-2.0; preserve separately obtained licenses; no bundle redistribution grant |
| Diffusers / CUDA extensions / Python dependencies | No | observed versions and pinned proposed sources in [deployment](../deploy/README.md) | External dependencies with individual terms; not project-original code |

## Training-only resources

VisDrone images, sketches, captions and checkpoints, FS-COCO and research
artifacts remain excluded. Dataset source/terms must be reviewed separately if
using `scripts/data/` or `scripts/training/`. They are not listed as inference
installation requirements. Historical observed model sizes (not new downloads):
FLUX base about 54G, Fill 55G, ControlNet 8G, Qwen models 43G, TRELLIS 3.1G;
provide additional disk space for dependencies, auxiliary weights and outputs.
