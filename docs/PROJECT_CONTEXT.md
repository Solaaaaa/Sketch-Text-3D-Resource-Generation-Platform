# Project Context

The historical graduation thesis title was `面向战术情境推演的草图—文本协同场景生成平台`.
The public software described by this repository is positioned for
architectural and urban scene creation: a user supplies a hand-drawn building
or city sketch and text, then refines the image, requests multiple views, and
produces a 3D resource.

VisDrone was used as public aerial imagery containing urban and suburban
structures, roads, vehicles, and people. The author prepared paired target
images, Canny-derived hand-drawn conditioning images, and captions for a
FLUX ControlNet training experiment. The final preprocessing concepts are
represented by the configurable script in `scripts/data/`; the original
working files are not copied into this repository.

The current implementation is a local prototype. A historical full-chain
test exercised the four stages and left private outputs under the local
runtime directory. Those outputs, model files, and datasets are intentionally
excluded, and the test does not establish public production readiness.
