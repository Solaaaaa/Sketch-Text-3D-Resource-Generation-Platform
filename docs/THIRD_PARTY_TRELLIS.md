# Third-party TRELLIS runtime

## Provenance and boundaries

`external/TRELLIS/trellis/` contains the complete Python package (models,
modules, pipelines, representations, renderers and utils) from
[microsoft/TRELLIS](https://github.com/microsoft/TRELLIS/tree/eeacb0bf6a7d25058232d746bef4e5e880b130ff),
commit `eeacb0bf6a7d25058232d746bef4e5e880b130ff`. The upstream root MIT license is
retained in `external/TRELLIS/LICENSE`. Upstream describes **the majority** of its
code as MIT; component notices and restrictions below also apply. This project
supplies an adapter and a documented source copy, without claiming authorship
of TRELLIS. PolyForm Noncommercial applies only to author-owned project files,
not TRELLIS, FlexiCubes, Gaussian-derived files, model weights or dependencies.

The upstream README is retained as historical provenance. Its examples using
Hub model IDs and its broad installer are not this app's deployment instructions;
use [our deployment guide](../deploy/README.md). Upstream app/examples, assets,
dataset toolkits, temporary files, Git metadata, caches, compiled extensions and
all weights are excluded.

## FlexiCubes

The submodule is [MaxtirError/FlexiCubes](https://github.com/MaxtirError/FlexiCubes/tree/f97beb0dd3c6c68f3ab5696b6dcaf9af69f0514e),
commit `f97beb0dd3c6c68f3ab5696b6dcaf9af69f0514e`. Runtime files, attribution README,
and complete NVIDIA `LICENSE.txt` are retained under
`trellis/representations/mesh/flexicubes/`. That license limits use to
non-commercial research/evaluation and requires retaining notices and the
license. FlexiCubes imports `kaolin.utils.testing.check_tensor`; Kaolin is an
external dependency even though this app does not call its training tools.

## Gaussian code attribution

Two runtime files retain Inria/GRAPHDECO copyright headers referring to
`LICENSE.md`. The 2026-10-15 source audit established the following provenance
and added complete, unmodified upstream license texts in
`external/TRELLIS/third_party_licenses/`:

- `trellis/representations/gaussian/general_utils.py` matches all whitespace-
  separated source tokens in [3D Gaussian Splatting utils/general_utils.py](https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/utils/general_utils.py).
  The exact [LICENSE.md at that commit](https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/LICENSE.md)
  is retained as `GAUSSIAN_SPLATTING_LICENSE.md`.
- `trellis/renderers/gaussian_render.py` adapts the
  [Mip-Splatting renderer](https://github.com/autonomousvision/mip-splatting/blob/dda02ab5ecf45d6edb8c540d9bb65c7e451345a9/gaussian_renderer/__init__.py):
  shared render implementation and `kernel_size`/`subpixel_offset` rasterizer
  interface (85 matching complete lines out of 105 upstream lines), with TRELLIS
  camera handling and a wrapper added. Mip-Splatting's pinned README states
  "This project is built upon 3DGS. Please follow the license of 3DGS."
  Its exact [LICENSE.md](https://github.com/autonomousvision/mip-splatting/blob/dda02ab5ecf45d6edb8c540d9bb65c7e451345a9/LICENSE.md)
  is retained as `MIP_SPLATTING_LICENSE.md`.

These licenses limit use to non-commercial research/evaluation and require
retaining the complete license and copyright/attribution notices. Copyright
headers are preserved. The 3DGS license includes a section about `loss_utils.py`
not present in the Mip-Splatting copy; both complete texts are retained without
merging them. `renderers/sh_utils.py` retains its full PlenOctree BSD notice inline.
The external Mip-Splatting CUDA rasterizer also has its own license; it is not
included in this repository.

## vox2seq: optional, excluded pending license resolution

The fixed TRELLIS tree's `extensions/vox2seq/setup.py` states:

> Copyright (C) 2023, Inria
> GRAPHDECO research group, https://team.inria.fr/graphdeco
> All rights reserved.
> This software is free for non-commercial, research and evaluation use
> under the terms of the LICENSE.md file.

Evidence: [fixed setup.py](https://github.com/microsoft/TRELLIS/blob/eeacb0bf6a7d25058232d746bef4e5e880b130ff/extensions/vox2seq/setup.py),
[fixed tree](https://api.github.com/repos/microsoft/TRELLIS/git/trees/eeacb0bf6a7d25058232d746bef4e5e880b130ff?recursive=1),
and [path history](https://api.github.com/repos/microsoft/TRELLIS/commits?path=extensions/vox2seq&sha=eeacb0bf6a7d25058232d746bef4e5e880b130ff).
The tree has no `extensions/vox2seq/LICENSE.md`; that raw URL returns 404.
History through this commit has one release entry,
`334d3b2fa1212477230ea1be989447273f389618`, with the same unresolved notice.
`src/api.h` additionally names Jianfeng XIANG; `pytorch/z_order.py` names
Peng-Shuai Wang and an MIT license file; `pytorch/hilbert.py` references
PrincetonLIPS/numpy-hilbert-curve. This mix does not establish a complete,
unambiguous license for the extension. A same-named file from another project
would not resolve it. The Gaussian licenses above are **not** assigned to vox2seq.

Dependency trace, independently checked against the six JSON configs referenced
by the local TRELLIS-image-large `pipeline.json`:

| Pipeline model | Attention route |
|---|---|
| sparse structure decoder | convolutional decoder, no vox2seq |
| sparse structure flow | `SparseStructureFlowModel`: dense `full` attention |
| structured latent flow | `SLatFlowModel`: sparse `full` attention |
| Gaussian / radiance field / mesh decoders | `attn_mode="swin"`, window size 8 |

`structured_latent_vae/base.py::block_attn_config()` maps `swin` to `windowed`.
`modules/sparse/attention/modules.py` calls the full/windowed functions for these
routes. Only `serialized_attn.py::calc_serialization()` lazily imports vox2seq
and calls `encode()`. Importing that Python module alone does not import vox2seq.
`shift_window`, `shift_sequence`, `shift_order` and direct serialized attention
remain optional modes requiring it. The app's stochastic/multidiffusion input
combination choice does not change these model attention modes.

Accordingly the 16 extension files were backed up outside the candidate and
removed from `external/TRELLIS/extensions/vox2seq`. The complete `trellis/`
package, including its lazy optional code, remains. The default audited model
configuration needs no vox2seq install. To use a different configuration that
requires serialization, first resolve licenses with the
[pinned upstream extension](https://github.com/microsoft/TRELLIS/tree/eeacb0bf6a7d25058232d746bef4e5e880b130ff/extensions/vox2seq),
then obtain/build it privately. This repository does not claim all optional
TRELLIS modes are ready to run.

## Recorded patches and external requirements

Preserved local patches: `modules/sparse/__init__.py` accepts `sdpa`/`naive`;
full, serialized and windowed sparse attention implement the corresponding
PyTorch paths. `pipelines/trellis_image_to_3d.py` checks the local
`facebookresearch_dinov2_main` Torch Hub checkout before upstream Hub loading.
This is auxiliary DINOv2 loading, not a TRELLIS-image-large checkpoint source.
The upstream app's command-line port change is excluded. Staging also removes
remote checkpoint fallbacks from `models/__init__.py`, `pipelines/base.py` and
`pipelines/__init__.py`; trailing whitespace was cleaned in copied source files.

TRELLIS still needs CUDA/PyTorch, spconv, nvdiffrast, the **Mip-Splatting** variant
of diff_gaussian_rasterization, utils3d, Kaolin, rembg/ONNX Runtime, DINOv2/U2Net,
and supporting Python packages. See [deployment](../deploy/README.md) for pinned
sources, observed versions and preparation commands. They are external code,
not project-original implementations. `diffoctreerast` is only needed for the
optional octree/radiance-field rendering route; this runner requests Gaussian
and mesh output and does not render radiance fields.

## Strict local primary model

TRELLIS-image-large weights are excluded. App and runner require a complete local
`pipeline.json` and referenced JSON/safetensors files under
`SKETCH_PLATFORM_TRELLIS_MODEL` (default `models/TRELLIS-image-large`), including
path containment checks. They never fall back to a Hub or cached primary model.
The app sets `HF_HOME`, `TORCH_HOME` and `U2NET_HOME` below
`SKETCH_PLATFORM_CACHE_ROOT` for auxiliary resources. A new server must explicitly
prepare those resources; it cannot rely on the author's cache being present.
