# New Linux GPU server deployment

These are preparation instructions for a **new isolated environment**. The
repository includes integration code and the attributed TRELLIS Python runtime.
Weights, ComfyUI, custom nodes, CUDA extensions and caches are obtained separately.
Datasets and training tools are not prerequisites for four-stage inference.

Release order decided 2026-10-15: candidate preparation → parent review → user
manually uploads only `candidate_repo/` → new GPU server clones/prepares resources
→ complete four-stage GPU/browser validation. The repaired full GPU chain has
not yet passed. The occupied original machine is not a prerequisite testing host.

## 1. Observed environment versus the proposed new installation

The read-only audit of the working main interpreter on 2026-10-15 observed
Python 3.12.3 on Linux x86_64/glibc 2.35, Torch 2.8.0+cu128, torchvision
0.23.0+cu128, Gradio 6.9.0 and gradio-litmodel3d 0.0.1. The October 12 run
succeeded through Stage 3 and generated Stage 4 video/state before a Gradio
output path failure. October 15 CPU regression verified the allowed_paths fix.
It did not verify new GPU exports or browser rendering.

| Dependency | Observed | Public source / new-server choice |
|---|---|---|
| Diffusers | editable source `0.37.0.dev0`; stale installed distribution metadata says `0.36.0.dev0`; checkout has no Git revision | [huggingface/diffusers](https://github.com/huggingface/diffusers), public v0.37.0 commit `d791c5c024014784df4a8dac2601e19fb4d300fc`; proposed reproducible replacement, not proven identical |
| ComfyUI | source version 0.6.0; no Git metadata; frontend installed 1.39.19 although bundle requirements request 1.34.9 | [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI), v0.6.0 commit `e4c61d75555036fa28b6bb34e5fd67b007c9f391`; source/node and GPU checks still required |
| ComfyUI-GGUF | source 2.0.0 (GitHub main), no revision; gguf 0.17.1 | [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF), checked public commit `6ea2651e7df66d7585f6ffee804b20e92fb38b8a`; not proven identical to the bundle |
| spconv | spconv-cu126 2.3.8 | [traveller59/spconv](https://github.com/traveller59/spconv); observed CUDA 12.6 wheel with Torch cu128 is a tested historical environment fact, not universal compatibility |
| nvdiffrast | 0.4.0 from a main.zip; original commit unknown | [NVlabs/nvdiffrast](https://github.com/NVlabs/nvdiffrast), v0.4.0 commit `253ac4fcea7de5f396371124af597e6cc957bfae` |
| Gaussian rasterizer | 0.0.0 from local mip-splatting checkout; only generated egg-info is dirty | [autonomousvision/mip-splatting](https://github.com/autonomousvision/mip-splatting), commit `dda02ab5ecf45d6edb8c540d9bb65c7e451345a9`, `submodules/diff-gaussian-rasterization` |
| utils3d | 0.0.2, direct_url records exact commit | [EasternJournalist/utils3d](https://github.com/EasternJournalist/utils3d), `9a4eb15e4021b67b12c460c7057d642626897ec8` |
| Kaolin | 0.18.0, binary CPython 3.12 wheel; install origin not recorded | official [Torch 2.8/cu128 wheel index](https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html) has a CPython 3.12 Linux wheel |
| rembg / ONNX Runtime | 2.0.72 / onnxruntime-gpu 1.24.2 | propose rembg 2.0.67 with the same U2Net API/weight; 2.0.72's declared NumPy/Pillow requirements conflict with this stack; GPU behavior of the replacement is pending |

`requirements-runtime.txt` records selected relevant versions and the proposed
rembg replacement, not a whole-machine pip freeze or a complete dependency lock.
The installed environment contains metadata conflicts: rembg 2.0.72 requires
NumPy >=2.3 and Pillow >=12.1, while this stack uses NumPy 2.2.6/Pillow 10.4.0 and
OpenCV 4.12 requires NumPy <2.3. The proposed rembg 2.0.67 supports Python 3.12
and these pins according to its [published metadata](https://pypi.org/pypi/rembg/2.0.67/json).
Its U2Net session factory and download URL/hash were checked in upstream source;
the replacement has not been installed or run on this host.

gradio-litmodel3d 0.0.1 declares `gradio>=4,<5`, conflicting with Gradio 6.9.0.
It is therefore installed separately after the main dependencies, as an explicit
exception described below. Historical app startup and the CPU preview/download
regression exercised this combination; they do not establish complete browser
compatibility. Do not downgrade Gradio to 4: this app uses Gradio 6 launch options.

New-server resolution,
CUDA extension builds and all GPU operations remain unverified. The old TRELLIS
upstream installer defaults to Python 3.10/Torch 2.4/CUDA 11.8 and Gradio 4.44.1;
do not combine that full installer with this observed stack. The old dedicated
TRELLIS Python paths mentioned in historical notes are not verified environments.

## 2. Create the app environment and install dependencies

Choose a GPU image with Python 3.12, a driver supporting CUDA 12.8, CUDA 12.8
**development toolkit including nvcc**, a compatible C++ compiler, Git, and the
normal Linux build tools. PyTorch wheels alone do not supply the compiler for
extensions. On other GPU/driver/OS combinations choose a compatible stack first;
the commands below target the observed Linux x86_64 stack. Execute them only on
the new server. First clone your own uploaded repository and `cd` to its root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r deploy/requirements-runtime.txt -c deploy/constraints-core.txt

mkdir -p external/dependencies
git clone https://github.com/huggingface/diffusers.git external/dependencies/diffusers
git -C external/dependencies/diffusers checkout d791c5c024014784df4a8dac2601e19fb4d300fc
python -m pip install -e ./external/dependencies/diffusers -c deploy/constraints-core.txt
python -m pip install spconv-cu126==2.3.8
python -m pip install kaolin==0.18.0 \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
python -m pip install \
  'git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8'

git clone https://github.com/NVlabs/nvdiffrast.git external/dependencies/nvdiffrast
git -C external/dependencies/nvdiffrast checkout 253ac4fcea7de5f396371124af597e6cc957bfae
python -m pip install --no-build-isolation ./external/dependencies/nvdiffrast
git clone https://github.com/autonomousvision/mip-splatting.git external/dependencies/mip-splatting
git -C external/dependencies/mip-splatting checkout dda02ab5ecf45d6edb8c540d9bb65c7e451345a9
git -C external/dependencies/mip-splatting submodule update --init --recursive
python -m pip install --no-build-isolation \
  ./external/dependencies/mip-splatting/submodules/diff-gaussian-rasterization
```

The rasterizer must support `kernel_size` and `subpixel_offset`, as used by the
vendored renderer. Replacing it with another package of the same import name
can fail at runtime. CUDA compiler/architecture compatibility, spconv kernel
selection and ONNX Runtime CUDA/cuDNN libraries must be validated on the chosen
GPU. Follow each upstream's build instructions if its wheel/build is unsuitable;
do not suppress installation failures or assume importing metadata proves ABI
compatibility. Kaolin is needed by FlexiCubes. `vox2seq` is optional and excluded
because its license is unresolved; no default install is required.
`diffoctreerast`, xformers and flash-attn are not needed for this runner's audited
Gaussian/mesh route with SDPA. See [TRELLIS notice](../docs/THIRD_PARTY_TRELLIS.md).

## 3. Qwen / ComfyUI

Follow [the acquisition and extraction guide](../docs/THIRD_PARTY_QWEN_BUNDLE.md)
for the third-party page/UC entries and the exact model/node list. Extract to
`external/ComfyUI_Qwen-Multiple-Angles/ComfyUI/main.py`, or configure an absolute
parent path. Do not run the Windows launcher/environment on Linux.

If assembling ComfyUI from public source instead of using bundle code, this is
the pinned starting point (run only into a new, absent target directory):

```bash
git clone https://github.com/Comfy-Org/ComfyUI.git \
  external/ComfyUI_Qwen-Multiple-Angles/ComfyUI
git -C external/ComfyUI_Qwen-Multiple-Angles/ComfyUI checkout \
  e4c61d75555036fa28b6bb34e5fd67b007c9f391
# GGUF backend only; skip this clone for Diffusion-only deployment.
git clone https://github.com/city96/ComfyUI-GGUF.git \
  external/ComfyUI_Qwen-Multiple-Angles/ComfyUI/custom_nodes/ComfyUI-GGUF
git -C external/ComfyUI_Qwen-Multiple-Angles/ComfyUI/custom_nodes/ComfyUI-GGUF \
  checkout 6ea2651e7df66d7585f6ffee804b20e92fb38b8a
```

The public revisions are verified to exist, but their combination is a proposed
new-server installation, not a recreation of undocumented bundle revisions.
Place the selected Qwen weights into the exact directories in the Qwen guide.
In the **app's Linux environment**, install core and required node dependencies:

```bash
python -m pip install -r external/ComfyUI_Qwen-Multiple-Angles/ComfyUI/requirements.txt \
  -c deploy/constraints-core.txt
# GGUF only:
python -m pip install -r external/ComfyUI_Qwen-Multiple-Angles/ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt \
  -c deploy/constraints-core.txt
python -m pip check
```

At this point `pip check` must pass. Then install the one custom component with
dependency resolution disabled so it cannot replace Gradio:

```bash
python -m pip install --no-deps gradio-litmodel3d==0.0.1
python -m pip check
```

The second check is expected to exit 1 with exactly this known declaration
conflict: `gradio-litmodel3d 0.0.1 has requirement gradio<5.0,>=4.0, but you have
gradio 6.9.0`. Any additional missing dependency or conflict must be resolved
before launch. Keep the diagnostic output; do not suppress all errors or use
`--no-deps` for the other packages. This is a documented compatibility exception,
not a claim of a fully consistent dependency lock. Verify the component's model
preview and downloads in the new server's browser acceptance test.

For a different root use that path in these install commands. Review or disable
unused bundled custom nodes in your private deployment to avoid unnecessary
startup dependencies. Do not install every bundle plugin just because it exists.
The app checks `/history`, then starts ComfyUI as needed with its own interpreter
and `--disable-xformers`. Keep localhost URL/host/port consistent. Do not start
a duplicate service. See the Qwen guide for separately managed services.

## 4. Obtain primary models

The IDs, metadata and Diffusers directory layouts were checked through public
ModelScope metadata and small model cards on 2026-10-15; no weight download was
performed during preparation. Accept provider/model terms, including access
approval if requested. Commands below download full directories into the clone:

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

FLUX requires the full Diffusers component directories, not only top-level
`flux1-*.safetensors`. ControlNet must include `config.json` and
`diffusion_pytorch_model.safetensors`; platform `configuration.json` is not a
replacement. TRELLIS requires `pipeline.json` and each referenced JSON/weight
pair, all inside its configured local directory. Primary TRELLIS fallback to
Hub IDs or caches remains disabled. See [model manifest](../docs/MODEL_MANIFEST.md).

## 5. Auxiliary DINOv2 and U2Net

Stage 4 uses `dinov2_vitl14_reg` and `rembg.new_session('u2net')`. They are separate
from TRELLIS-image-large. App sets these child-process locations from
`SKETCH_PLATFORM_CACHE_ROOT` (project-relative by default):

- Torch Hub source: `cache/torch/hub/facebookresearch_dinov2_main/`
- DINOv2 weight: `cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth`
- U2Net weight: `cache/u2net/u2net.onnx`
- HF auxiliary cache: `cache/huggingface/`

Prepare them explicitly on the new server. Example for the default cache root;
substitute your absolute cache root if customized:

```bash
AUX_CACHE="$PWD/cache"
mkdir -p "$AUX_CACHE/torch/hub/checkpoints" "$AUX_CACHE/u2net"
git clone https://github.com/facebookresearch/dinov2.git \
  "$AUX_CACHE/torch/hub/facebookresearch_dinov2_main"
git -C "$AUX_CACHE/torch/hub/facebookresearch_dinov2_main" checkout \
  7764ea0f912e53c92e82eb78a2a1631e92725fc8
curl --fail --location \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth \
  --output "$AUX_CACHE/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
curl --fail --location \
  https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx \
  --output "$AUX_CACHE/u2net/u2net.onnx"
```

DINOv2 URL/naming follows the [upstream Hub loader](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/hub/backbones.py)
and locally audited Hub utils. Both the observed rembg 2.0.72 loader and the
[proposed 2.0.67 loader](https://github.com/danielgatis/rembg/blob/v2.0.67/rembg/sessions/u2net.py)
specify the U2Net URL
and MD5 `60024c5c889badc19c04ad937298a77b`; compare `md5sum` after download (a
provider integrity check, not a security guarantee). No DINOv2 weight SHA256 is
claimed. The local DINOv2 checkout's historical revision was unrecorded; this
public revision is a new-server candidate. Respect DINOv2/U2Net upstream terms.

Without preparation, the loaders may attempt network downloads for these
auxiliary resources on first use. This does not permit a primary TRELLIS model
fallback. Setting `TORCH_HOME` or `U2NET_HOME` externally does not override the
app's per-run settings; set `SKETCH_PLATFORM_CACHE_ROOT` and populate its layout.

## 6. Configure, check files, then validate on the new GPU

```bash
cp .env.example .env
# Edit .env to match this server. It is a trusted Bash file, never committed.
set -a
source ./.env
set +a
python deploy/check_resources.py --stage all --qwen-backend GGUF
bash deploy/start_app.example.sh
```

Complete the dependency check in section 3 first; its one documented custom
component exception must not hide other failures.

Select `Diffusion` in the file check and UI if that is the backend you obtained.
`check_resources.py` only reads paths, JSON and source text; it imports no model
framework, opens no weight tensor, calls no CUDA API and makes no network requests.
It cannot prove model contents, node registration or extension ABI compatibility.

Run the launch script from the repository root with `.venv` activated. It loads
`.env` via Bash `source`, exports assignments (`set -a`), then executes `python
app.py`. `ENV_FILE` and `APP_PATH` overrides are interpreted relative to the shell
working directory. App/runner resource paths in `.env` are interpreted relative
to their script directory; absolute overrides are allowed. Direct `python app.py`
does not read `.env` automatically. `TRELLIS_PYTHON_BIN=.venv/bin/python` uses the
same prepared environment; a separate working environment can be specified.

Keep `SKETCH_PLATFORM_RUNTIME_ROOT` dedicated to application outputs. The app
launch preserves `allowed_paths=[str(TRELLIS_RUNTIME_ROOT.resolve())]`, allowing
Gradio 6.9 to return video/GLB/PLY from an external runtime directory. A Gradio
cache setting alone does not authorize that directory.

On the new server, verify the four stages in the browser, including Stage 4 video
playback, **new** GLB/PLY exports, previews and downloads with runtime outside the
clone. Preserve logs and record dependency versions/GPU/driver plus model source
revisions. If Gradio rejects a path after generation, inspect existing video/state
before repeating GPU work. Upload readiness and a successful static file check
are not claims that these GPU/browser checks have passed.
