# Qwen / ComfyUI preparation

## Acquisition and extraction

The third-party introduction page is <https://deepface.cc/thread-925-1-1.html>.
On 2026-10-15 it returned HTTP 200 and displayed separate UC disk entries for
**main program** and **models**. These are third-party share links, not this
project's releases. Follow the entries on that page; obtain the program and the
weights for your chosen backend. Archive names, extraction passwords, share
availability and archive contents have not been verified by downloading them.
This project provides no bundle mirror and uploads no bundle to GitHub/ModelScope.
A public share link does not establish redistribution permission.

Extract into your private resources directory. The expected final layout is:

```text
external/ComfyUI_Qwen-Multiple-Angles/
└── ComfyUI/
    ├── main.py
    ├── requirements.txt
    ├── nodes.py
    ├── comfy_extras/
    ├── custom_nodes/
    │   └── ComfyUI-GGUF/           # required only for GGUF
    └── models/
        ├── unet/                 # GGUF choice
        ├── diffusion_models/     # Diffusion choice
        ├── text_encoders/
        ├── vae/
        └── loras/
```

Set `SKETCH_PLATFORM_QWEN_ROOT=external/ComfyUI_Qwen-Multiple-Angles` in `.env`.
This variable points to the directory **containing `ComfyUI/`**. An absolute
path is also accepted. If extraction creates an extra outer directory, locate
`ComfyUI/main.py` and set the variable to the parent of that `ComfyUI` directory;
check `test -f "$SKETCH_PLATFORM_QWEN_ROOT/ComfyUI/main.py"` from the project root.
Do not assume an archive's topmost directory is the right root.

The inspected local bundle has Windows launcher files and a `deepface/`
runtime. Windows `.bat` files and bundled Python binaries are not a Linux
environment. Use a fresh Linux Python environment as described in
[deployment instructions](../deploy/README.md).

## Exact workflow resources

Paths below are relative to `ComfyUI/`. Download **one** diffusion backend;
both backends need all four common files.

| Choice | Required file | Workflow loader |
|---|---|---|
| GGUF (UI default) | `models/unet/qwen-image-edit-2511-Q4_K_S.gguf` | `UnetLoaderGGUF` from ComfyUI-GGUF |
| Diffusion alternative | `models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors` | core `UNETLoader` |
| Common text encoder | `models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` | `CLIPLoader`, type `qwen_image` |
| Common VAE | `models/vae/qwen_image_vae.safetensors` | `VAELoader` |
| Common angle LoRA | `models/loras/qwen-image-edit-2511-multiple-angles-lora.safetensors` | `LoraLoaderModelOnly` |
| Common acceleration LoRA | `models/loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors` | `LoraLoaderModelOnly` |

The 4-step Lightning, AnyPose and realism LoRAs present in the inspected bundle
are not used by this application's workflow. Do not substitute them for the
8-step file. Individual weight upstreams/revisions/licenses are not established
from filenames; review them via the provider before downloading or using them.

Core nodes required by both backends: `LoraLoaderModelOnly`, `CLIPLoader`,
`VAELoader`, `LoadImage`, `ModelSamplingAuraFlow`, `VAEEncode`, `KSampler`,
`VAEDecode`, `SaveImage`, plus:

| Node | Source in the inspected ComfyUI tree |
|---|---|
| `TextEncodeQwenImageEditPlus` | `comfy_extras/nodes_qwen.py` |
| `FluxKontextMultiReferenceLatentMethod` | `comfy_extras/nodes_flux.py` |
| `CFGNorm` | `comfy_extras/nodes_cfg.py` |

GGUF additionally requires [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
and its `requirements.txt` (`gguf`, `sentencepiece`, `protobuf`). The local node
identifies itself as 2.0.0 (GitHub main), but has no Git revision metadata.
`ComfyUI-qwenmultiangle`, `ComfyUI-Custom-Scripts` and
`ComfyUI-to-Python-Extension` are in the inspected bundle but no node from those
packages is referenced in `build_multiview_comfy_prompt()`. They are not required
by this app; if retained in a private installation, ComfyUI may still import them
at startup, so review their dependencies/licenses or disable them there.

File-only check, after loading your `.env` from the project root:

```bash
set -a
source ./.env
set +a
python deploy/check_resources.py --stage qwen --qwen-backend GGUF
# Or, for the alternative backend:
python deploy/check_resources.py --stage qwen --qwen-backend Diffusion
```

This reads filenames and node source text. It does not load models, import
ComfyUI, or prove that nodes register successfully; the new server must verify
node registration and complete a real workflow.

## Connection and Python behavior

`app.py` probes `SKETCH_PLATFORM_COMFY_URL/history`. A successful response is
reused. Otherwise it starts `ComfyUI/main.py` using **`sys.executable`**, the
same interpreter as the app, with `--listen SKETCH_PLATFORM_COMFY_BIND_HOST`,
`--port SKETCH_PLATFORM_COMFY_PORT`, and `--disable-xformers`. Its working
directory is `ComfyUI/`; startup logs go to `SKETCH_PLATFORM_COMFY_LOG`.
The default is localhost:8188. There is no separate Comfy Python variable.
Install ComfyUI and required node dependencies into the app environment for
this default mode. You do not need to launch a second service yourself.

An already running separately managed service can be reused. Keep its URL and
local resources consistent. If a custom/remote URL is unavailable, the app still
attempts a local start; it has no remote-service-only switch. Before Stage 4 the
app stops its own ComfyUI child to free VRAM; for a reused service it requests
resource release. Avoid sharing the same service with unrelated jobs.
