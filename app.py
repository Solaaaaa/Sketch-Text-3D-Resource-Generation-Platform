import atexit
import gc
import io
import itertools
import json
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import requests
import torch
from gradio_litmodel3d import LitModel3D
from PIL import Image
from diffusers import FluxControlNetModel, FluxControlNetPipeline, FluxFillPipeline


PROJECT_ROOT = Path(__file__).resolve().parent


def _env_path(name: str, default: str) -> Path:
    """Resolve absolute overrides or paths relative to this project copy."""
    configured = os.environ.get(name)
    path = Path(configured if configured else default).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


BASE_MODEL_PATH = str(_env_path("SKETCH_PLATFORM_BASE_MODEL", "models/FLUX.1-dev"))
CONTROLNET_MODEL_PATH = str(
    _env_path(
        "SKETCH_PLATFORM_CONTROLNET_MODEL",
        "models/flux1-dev-visdrone-urban-sketch-controlnet",
    )
)
FILL_MODEL_PATH = str(_env_path("SKETCH_PLATFORM_FILL_MODEL", "models/FLUX.1-Fill-dev"))
DEFAULT_CONTROLNET_CONDITIONING_SCALE = 0.7
PARAMETER_GRID = list(itertools.product([0.7, 0.9], [20, 30], [3.0, 3.5, 4.0]))

QWEN_PROJECT_ROOT = _env_path(
    "SKETCH_PLATFORM_QWEN_ROOT", "external/ComfyUI_Qwen-Multiple-Angles"
)
COMFY_ROOT = QWEN_PROJECT_ROOT / "ComfyUI"
COMFY_MAIN_PATH = COMFY_ROOT / "main.py"
COMFY_BIND_HOST = os.environ.get("SKETCH_PLATFORM_COMFY_BIND_HOST", "127.0.0.1")
COMFY_PORT = int(os.environ.get("SKETCH_PLATFORM_COMFY_PORT", "8188"))
COMFY_SERVER_URL = os.environ.get(
    "SKETCH_PLATFORM_COMFY_URL", f"http://{COMFY_BIND_HOST}:{COMFY_PORT}"
).rstrip("/")
COMFY_HISTORY_URL = f"{COMFY_SERVER_URL}/history"
COMFY_PROMPT_URL = f"{COMFY_SERVER_URL}/prompt"
COMFY_VIEW_URL = f"{COMFY_SERVER_URL}/view"
COMFY_UPLOAD_URL = f"{COMFY_SERVER_URL}/upload/image"
COMFY_STARTUP_LOG = _env_path(
    "SKETCH_PLATFORM_COMFY_LOG", "runtime/qwen_multiangle_comfy.log"
)

MULTIVIEW_MODEL_CHOICES = {
    "GGUF": ["qwen-image-edit-2511-Q4_K_S.gguf"],
    "Diffusion": ["qwen_image_edit_2511_fp8mixed.safetensors"],
}
MULTIVIEW_CLIP_MODEL = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
MULTIVIEW_VAE_MODEL = "qwen_image_vae.safetensors"
MULTIVIEW_ANGLE_LORA = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
MULTIVIEW_LIGHTNING_LORA = "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"

TRELLIS_PROJECT_ROOT = _env_path("SKETCH_PLATFORM_TRELLIS_ROOT", "external/TRELLIS")
TRELLIS_RUNNER_PATH = _env_path(
    "SKETCH_PLATFORM_TRELLIS_RUNNER", "trellis_step4_runner.py"
)
TRELLIS_ENV_PYTHON = _env_path("TRELLIS_PYTHON_BIN", ".venv/bin/python")
TRELLIS_LOCAL_MODEL_PATH = _env_path(
    "SKETCH_PLATFORM_TRELLIS_MODEL", "models/TRELLIS-image-large"
)
TRELLIS_RUNTIME_ROOT = _env_path(
    "SKETCH_PLATFORM_RUNTIME_ROOT", "runtime"
)
TRELLIS_CACHE_ROOT = _env_path(
    "SKETCH_PLATFORM_CACHE_ROOT", "cache"
)
TRELLIS_WORKDIR = _env_path("SKETCH_PLATFORM_TRELLIS_WORKDIR", str(TRELLIS_PROJECT_ROOT))
TRELLIS_RESULT_PREFIX = "__TRELLIS_RESULT__="

controlnet_pipe = None
fill_pipe = None
comfy_process = None
comfy_log_handle = None
comfy_started_by_app = False

APP_CSS = """
.mv-reset-btn {
    max-width: 84px;
    min-width: 84px !important;
}

.mv-reset-btn button {
    height: 42px;
}

#mv-widget-host {
    padding: 0;
}

#mv-widget-host .mv-widget-shell {
    position: relative;
    width: 100%;
    height: 420px;
    overflow: hidden;
    border-radius: 18px;
    border: 1px solid rgba(46, 56, 75, 0.9);
    background: linear-gradient(180deg, #1b1b1f 0%, #17171a 100%);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}

#mv-widget-host .mv-widget-canvas {
    width: 100%;
    height: 100%;
    display: block;
}

#mv-widget-host .mv-widget-canvas canvas {
    width: 100%;
    height: 100%;
    display: block;
}

#mv-widget-host .mv-widget-loading {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #9fb0c8;
    font-size: 14px;
    letter-spacing: 0.02em;
    background: radial-gradient(circle at center, rgba(35, 40, 50, 0.22), rgba(24, 24, 28, 0.0) 58%);
    transition: opacity 0.2s ease;
    pointer-events: none;
    z-index: 2;
}

#mv-widget-host .mv-widget-shell[data-ready="1"] .mv-widget-loading {
    opacity: 0;
}

#mv-widget-host .mv-widget-status {
    position: absolute;
    left: 50%;
    bottom: 12px;
    transform: translateX(-50%);
    padding: 8px 16px;
    border-radius: 10px;
    border: 1px solid rgba(35, 35, 35, 0.96);
    background: rgba(4, 8, 12, 0.92);
    color: #86f7dd;
    font-size: 13px;
    line-height: 1.2;
    letter-spacing: 0.01em;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    white-space: nowrap;
    backdrop-filter: blur(8px);
    box-shadow: 0 8px 26px rgba(0, 0, 0, 0.35);
    pointer-events: none;
    z-index: 3;
}

@media (max-width: 768px) {
    #mv-widget-host .mv-widget-shell {
        height: 360px;
    }

    #mv-widget-host .mv-widget-status {
        font-size: 12px;
        padding: 7px 12px;
    }
}
"""

MULTIANGLE_WIDGET_HTML = """
<div class="mv-widget-shell">
  <div class="mv-widget-canvas"></div>
  <div class="mv-widget-loading">正在加载 3D 可视化控制球...</div>
  <div class="mv-widget-status">方位角：90° ｜ 仰角：0° ｜ 距离：5.0</div>
</div>
"""

MULTIANGLE_WIDGET_HEAD = """
<script>
(() => {
  if (window.__multiAngleWidgetRegistered) {
    return;
  }
  window.__multiAngleWidgetRegistered = true;

  const ids = {
    widgetHost: 'mv-widget-host',
    sourcePreview: 'mv-source-preview',
    azimuth: 'mv-azimuth',
    elevation: 'mv-elevation',
    distance: 'mv-distance',
  };

  function getRoot(id) {
    return document.getElementById(id);
  }

  function getRangeInput(root) {
    return root ? root.querySelector('input[type="range"]') : null;
  }

  function getNumberInput(root) {
    return root ? root.querySelector('input[type="number"]') : null;
  }

  function getValue(id) {
    const root = getRoot(id);
    const input = getNumberInput(root) || getRangeInput(root) || (root ? root.querySelector('input') : null);
    return input ? Number(input.value || 0) : 0;
  }

  function fire(el) {
    if (!el) {
      return;
    }
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function setSliderValue(id, value) {
    const root = getRoot(id);
    if (!root) {
      return;
    }

    const range = getRangeInput(root);
    const number = getNumberInput(root);
    const text = String(value);

    if (range) {
      range.value = text;
    }
    if (number) {
      number.value = text;
    }

    fire(number || range);
    if (number && range && number !== range) {
      fire(range);
    }
  }

  function getSourceImageSrc() {
    const root = getRoot(ids.sourcePreview);
    if (!root) {
      return '';
    }
    const img = root.querySelector('img');
    return img ? img.src || '' : '';
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function bucketAzimuth(value) {
    const angle = ((value % 360) + 360) % 360;
    if (angle < 22.5 || angle >= 337.5) return 'front view';
    if (angle < 67.5) return 'front-right quarter view';
    if (angle < 112.5) return 'right side view';
    if (angle < 157.5) return 'back-right quarter view';
    if (angle < 202.5) return 'back view';
    if (angle < 247.5) return 'back-left quarter view';
    if (angle < 292.5) return 'left side view';
    return 'front-left quarter view';
  }

  function bucketElevation(value) {
    if (value < -15) return 'low-angle shot';
    if (value < 15) return 'eye-level shot';
    if (value < 45) return 'elevated shot';
    return 'high-angle shot';
  }

  function bucketDistance(value) {
    if (value < 2) return 'wide shot';
    if (value < 6) return 'medium shot';
    return 'close-up';
  }

  function formatPrompt(azimuth, elevation, distance) {
    return `<sks> ${bucketAzimuth(azimuth)} ${bucketElevation(elevation)} ${bucketDistance(distance)}`;
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const existing = document.querySelector(`script[data-mv-three-src="${src}"]`);
      if (existing) {
        if (window.THREE) {
          resolve(window.THREE);
          return;
        }
        existing.addEventListener('load', () => resolve(window.THREE), { once: true });
        existing.addEventListener('error', reject, { once: true });
        return;
      }

      const script = document.createElement('script');
      script.src = src;
      script.async = true;
      script.dataset.mvThreeSrc = src;
      script.onload = () => resolve(window.THREE);
      script.onerror = () => reject(new Error(`Failed to load ${src}`));
      document.head.appendChild(script);
    });
  }

  async function loadThree() {
    if (window.THREE) {
      return window.THREE;
    }
    if (window.__multiAngleThreePromise) {
      return window.__multiAngleThreePromise;
    }

    const sources = [
      'https://unpkg.com/three@0.160.0/build/three.min.js',
      'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js',
      'https://cdnjs.cloudflare.com/ajax/libs/three.js/r160/three.min.js',
    ];

    window.__multiAngleThreePromise = (async () => {
      let lastError = null;
      for (const src of sources) {
        try {
          const three = await loadScript(src);
          if (three) {
            return three;
          }
        } catch (error) {
          lastError = error;
        }
      }
      throw lastError || new Error('Unable to load Three.js');
    })();

    return window.__multiAngleThreePromise;
  }

  class MultiAngleWidget {
    constructor(shell, THREE) {
      this.THREE = THREE;
      this.shell = shell;
      this.canvasContainer = shell.querySelector('.mv-widget-canvas');
      this.loadingEl = shell.querySelector('.mv-widget-loading');
      this.statusEl = shell.querySelector('.mv-widget-status');

      this.CENTER = new THREE.Vector3(0, 0.6, 0);
      this.AZIMUTH_RADIUS = 1.8;
      this.ELEVATION_RADIUS = 1.42;
      this.ELEV_ARC_X = -0.88;
      this.COLORS = {
        azimuth: 0x19e6c3,
        elevation: 0xff7bd6,
        distance: 0xffcb3d,
        camera: 0x5a79b8,
      };

      this.state = {
        azimuth: clamp(getValue(ids.azimuth), 0, 315),
        elevation: clamp(getValue(ids.elevation), -30, 60),
        distance: clamp(getValue(ids.distance), 0, 10),
      };
      this.liveAzimuth = this.state.azimuth;
      this.liveElevation = this.state.elevation;
      this.liveDistance = this.state.distance;

      this.isDragging = false;
      this.dragTarget = null;
      this.hoveredHandle = null;
      this.lastImageSrc = '';
      this.animationId = null;
      this.time = 0;

      this.raycaster = new THREE.Raycaster();
      this.mouse = new THREE.Vector2();

      this.initScene();
      this.bindEvents();
      this.syncFromDom(true);
      this.shell.dataset.ready = '1';
      if (this.loadingEl) {
        this.loadingEl.textContent = '';
      }
      this.animate();
    }

    initScene() {
      const THREE = this.THREE;
      const width = Math.max(this.canvasContainer.clientWidth || 0, 320);
      const height = Math.max(this.canvasContainer.clientHeight || 0, 320);

      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color(0x1b1b1f);

      this.camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
      this.camera.position.set(4.2, 3.0, 4.6);
      this.camera.lookAt(0, 0.8, 0);

      this.renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: false,
        powerPreference: 'high-performance',
      });
      this.renderer.setSize(width, height);
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      if ('outputColorSpace' in this.renderer && THREE.SRGBColorSpace) {
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
      } else if ('outputEncoding' in this.renderer && THREE.sRGBEncoding) {
        this.renderer.outputEncoding = THREE.sRGBEncoding;
      }
      this.renderer.domElement.style.touchAction = 'none';
      this.canvasContainer.innerHTML = '';
      this.canvasContainer.appendChild(this.renderer.domElement);

      const ambient = new THREE.AmbientLight(0xffffff, 0.62);
      this.scene.add(ambient);

      const hemisphere = new THREE.HemisphereLight(0xffffff, 0x111114, 0.45);
      this.scene.add(hemisphere);

      const mainLight = new THREE.DirectionalLight(0xffffff, 0.85);
      mainLight.position.set(5.5, 7.5, 5.0);
      this.scene.add(mainLight);

      const fillLight = new THREE.DirectionalLight(0x527eff, 0.3);
      fillLight.position.set(-5.5, 4.0, -4.5);
      this.scene.add(fillLight);

      this.gridHelper = new THREE.GridHelper(10, 24, 0x272733, 0x22222c);
      this.gridHelper.position.y = 0;
      const gridMaterial = this.gridHelper.material;
      if (Array.isArray(gridMaterial)) {
        gridMaterial.forEach((material) => {
          material.transparent = true;
          material.opacity = 0.7;
        });
      } else {
        gridMaterial.transparent = true;
        gridMaterial.opacity = 0.7;
      }
      this.scene.add(this.gridHelper);

      this.createAxes();
      this.createSubject();
      this.createCameraIndicator();
      this.createAzimuthRing();
      this.createElevationArc();
      this.createDistanceHandle();
      this.updateVisuals();
      this.updateStatus();
    }

    createAxes() {
      const THREE = this.THREE;
      const axisMaterial = new THREE.LineBasicMaterial({
        color: 0x324056,
        transparent: true,
        opacity: 0.9,
      });
      const points = [
        new THREE.Vector3(-2.5, 0.001, 0),
        new THREE.Vector3(2.5, 0.001, 0),
        new THREE.Vector3(0, 0.001, -2.5),
        new THREE.Vector3(0, 0.001, 2.5),
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(0, 1.7, 0),
      ];
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      this.axes = new THREE.LineSegments(geometry, axisMaterial);
      this.scene.add(this.axes);
    }

    createGridTexture() {
      const THREE = this.THREE;
      const canvas = document.createElement('canvas');
      const size = 256;
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext('2d');

      ctx.fillStyle = '#1c2130';
      ctx.fillRect(0, 0, size, size);
      ctx.strokeStyle = '#2a3242';
      ctx.lineWidth = 1;
      for (let i = 0; i <= size; i += 16) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, size);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(size, i);
        ctx.stroke();
      }

      const texture = new THREE.CanvasTexture(canvas);
      texture.wrapS = THREE.RepeatWrapping;
      texture.wrapT = THREE.RepeatWrapping;
      texture.repeat.set(4, 4);
      return texture;
    }

    createSubject() {
      const THREE = this.THREE;
      const cardGeo = new THREE.BoxGeometry(1.15, 1.2, 0.03);
      const frontMat = new THREE.MeshBasicMaterial({ color: 0x495360 });
      const backMat = new THREE.MeshBasicMaterial({ map: this.createGridTexture() });
      const edgeMat = new THREE.MeshBasicMaterial({ color: 0x202631 });
      this.planeMat = frontMat;
      this.imagePlane = new THREE.Mesh(cardGeo, [edgeMat, edgeMat, edgeMat, edgeMat, frontMat, backMat]);
      this.imagePlane.position.copy(this.CENTER);
      this.scene.add(this.imagePlane);

      const frameGeo = new THREE.EdgesGeometry(cardGeo);
      const frameMat = new THREE.LineBasicMaterial({
        color: 0xe7eefb,
        transparent: true,
        opacity: 0.22,
      });
      this.imageFrame = new THREE.LineSegments(frameGeo, frameMat);
      this.imageFrame.position.copy(this.CENTER);
      this.scene.add(this.imageFrame);
    }

    createCameraIndicator() {
      const THREE = this.THREE;
      this.cameraIndicator = new THREE.Group();

      const body = new THREE.Mesh(
        new THREE.BoxGeometry(0.42, 0.22, 0.22),
        new THREE.MeshStandardMaterial({
          color: this.COLORS.camera,
          emissive: 0x1b3157,
          emissiveIntensity: 0.35,
          metalness: 0.3,
          roughness: 0.5,
        })
      );
      this.cameraIndicator.add(body);

      const top = new THREE.Mesh(
        new THREE.BoxGeometry(0.16, 0.06, 0.1),
        new THREE.MeshStandardMaterial({
          color: 0x7c95ca,
          emissive: 0x263a68,
          emissiveIntensity: 0.25,
          metalness: 0.25,
          roughness: 0.55,
        })
      );
      top.position.set(-0.04, 0.14, 0);
      this.cameraIndicator.add(top);

      const lens = new THREE.Mesh(
        new THREE.CylinderGeometry(0.055, 0.055, 0.14, 20),
        new THREE.MeshStandardMaterial({
          color: 0xdfe6ff,
          emissive: 0x4b65a0,
          emissiveIntensity: 0.18,
          metalness: 0.65,
          roughness: 0.32,
        })
      );
      lens.rotation.x = Math.PI / 2;
      lens.position.set(0.21, 0, 0);
      this.cameraIndicator.add(lens);

      this.scene.add(this.cameraIndicator);
    }

    createAzimuthRing() {
      const THREE = this.THREE;
      this.azimuthRing = new THREE.Mesh(
        new THREE.TorusGeometry(this.AZIMUTH_RADIUS, 0.035, 18, 120),
        new THREE.MeshBasicMaterial({
          color: this.COLORS.azimuth,
          transparent: true,
          opacity: 0.92,
        })
      );
      this.azimuthRing.rotation.x = Math.PI / 2;
      this.azimuthRing.position.y = 0.02;
      this.scene.add(this.azimuthRing);

      this.azimuthHandle = new THREE.Mesh(
        new THREE.SphereGeometry(0.16, 32, 32),
        new THREE.MeshStandardMaterial({
          color: this.COLORS.azimuth,
          emissive: this.COLORS.azimuth,
          emissiveIntensity: 0.4,
          metalness: 0.18,
          roughness: 0.38,
        })
      );
      this.scene.add(this.azimuthHandle);
    }

    createElevationArc() {
      const THREE = this.THREE;
      const arcPoints = [];
      for (let i = 0; i <= 40; i += 1) {
        const angle = (-30 + (90 * i / 40)) * Math.PI / 180;
        arcPoints.push(new THREE.Vector3(
          this.ELEV_ARC_X,
          this.ELEVATION_RADIUS * Math.sin(angle) + this.CENTER.y,
          this.ELEVATION_RADIUS * Math.cos(angle)
        ));
      }
      const arcCurve = new THREE.CatmullRomCurve3(arcPoints);
      this.elevationArc = new THREE.Mesh(
        new THREE.TubeGeometry(arcCurve, 48, 0.032, 10, false),
        new THREE.MeshBasicMaterial({
          color: this.COLORS.elevation,
          transparent: true,
          opacity: 0.94,
        })
      );
      this.scene.add(this.elevationArc);

      this.elevationHandle = new THREE.Mesh(
        new THREE.SphereGeometry(0.16, 32, 32),
        new THREE.MeshStandardMaterial({
          color: this.COLORS.elevation,
          emissive: this.COLORS.elevation,
          emissiveIntensity: 0.4,
          metalness: 0.18,
          roughness: 0.38,
        })
      );
      this.scene.add(this.elevationHandle);
    }

    createDistanceHandle() {
      const THREE = this.THREE;
      this.distanceHandle = new THREE.Mesh(
        new THREE.SphereGeometry(0.16, 32, 32),
        new THREE.MeshStandardMaterial({
          color: this.COLORS.distance,
          emissive: this.COLORS.distance,
          emissiveIntensity: 0.35,
          metalness: 0.24,
          roughness: 0.38,
        })
      );
      this.scene.add(this.distanceHandle);
    }

    updateDistanceLine(start, end) {
      const THREE = this.THREE;
      if (this.distanceTube) {
        this.scene.remove(this.distanceTube);
        this.distanceTube.geometry.dispose();
        this.distanceTube.material.dispose();
      }
      const path = new THREE.LineCurve3(start, end);
      this.distanceTube = new THREE.Mesh(
        new THREE.TubeGeometry(path, 1, 0.01, 8, false),
        new THREE.MeshBasicMaterial({
          color: this.COLORS.distance,
          transparent: true,
          opacity: 0.92,
        })
      );
      this.scene.add(this.distanceTube);
    }

    getCameraPosition(azimuth, elevation, distance) {
      const azRad = (azimuth * Math.PI) / 180;
      const elRad = (elevation * Math.PI) / 180;
      const visualDistance = 2.6 - (distance / 10) * 2.0;
      return new this.THREE.Vector3(
        visualDistance * Math.sin(azRad) * Math.cos(elRad),
        this.CENTER.y + (visualDistance * Math.sin(elRad)),
        visualDistance * Math.cos(azRad) * Math.cos(elRad)
      );
    }

    getDistanceHandlePosition(cameraPosition, distance) {
      const t = 0.15 + ((10 - distance) / 10) * 0.7;
      return this.CENTER.clone().lerp(cameraPosition, t);
    }

    projectToScreen(vector) {
      const projected = vector.clone().project(this.camera);
      const width = this.renderer.domElement.clientWidth || this.renderer.domElement.width || 1;
      const height = this.renderer.domElement.clientHeight || this.renderer.domElement.height || 1;
      return {
        x: (projected.x * 0.5 + 0.5) * width,
        y: (-projected.y * 0.5 + 0.5) * height,
      };
    }

    getClosestDistanceValue(pointer) {
      let bestValue = this.liveDistance;
      let bestDistance = Infinity;
      const azimuth = this.liveAzimuth;
      const elevation = this.liveElevation;
      for (let value = 0; value <= 10.0001; value += 0.1) {
        const sample = Number(value.toFixed(1));
        const position = this.getDistanceHandlePosition(this.getCameraPosition(azimuth, elevation, sample), sample);
        const screen = this.projectToScreen(position);
        const dx = pointer.x - screen.x;
        const dy = pointer.y - screen.y;
        const distance = (dx * dx) + (dy * dy);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestValue = sample;
        }
      }
      return bestValue;
    }

    updateVisuals() {
      const cameraPosition = this.getCameraPosition(this.liveAzimuth, this.liveElevation, this.liveDistance);
      this.cameraIndicator.position.copy(cameraPosition);
      this.cameraIndicator.lookAt(this.CENTER);
      this.cameraIndicator.rotateY(Math.PI / 2);

      const azRad = (this.liveAzimuth * Math.PI) / 180;
      this.azimuthHandle.position.set(
        this.AZIMUTH_RADIUS * Math.sin(azRad),
        0.08,
        this.AZIMUTH_RADIUS * Math.cos(azRad)
      );

      const elRad = (this.liveElevation * Math.PI) / 180;
      this.elevationHandle.position.set(
        this.ELEV_ARC_X,
        this.CENTER.y + this.ELEVATION_RADIUS * Math.sin(elRad),
        this.ELEVATION_RADIUS * Math.cos(elRad)
      );

      const distancePosition = this.getDistanceHandlePosition(cameraPosition, this.liveDistance);
      this.distanceHandle.position.copy(distancePosition);
      this.updateDistanceLine(this.CENTER.clone(), cameraPosition.clone());
    }

    updateStatus() {
      const prompt = formatPrompt(this.liveAzimuth, this.liveElevation, this.liveDistance);
      if (this.statusEl) {
        this.statusEl.textContent = `方位角：${Math.round(this.liveAzimuth)}° ｜ 仰角：${Math.round(this.liveElevation)}° ｜ 距离：${this.liveDistance.toFixed(1)}`;
        this.statusEl.title = prompt;
      }
    }

    updateImage(url) {
      if (url === this.lastImageSrc) {
        return;
      }
      this.lastImageSrc = url;

      if (!url) {
        this.planeMat.map = null;
        this.planeMat.color.set(0x495360);
        this.planeMat.needsUpdate = true;
        this.imagePlane.scale.set(1, 1, 1);
        this.imageFrame.scale.set(1, 1, 1);
        return;
      }

      const img = new Image();
      if (!url.startsWith('data:')) {
        img.crossOrigin = 'anonymous';
      }

      img.onload = () => {
        const texture = new this.THREE.Texture(img);
        if (this.THREE.SRGBColorSpace) {
          texture.colorSpace = this.THREE.SRGBColorSpace;
        } else if (this.THREE.sRGBEncoding) {
          texture.encoding = this.THREE.sRGBEncoding;
        }
        texture.needsUpdate = true;
        this.planeMat.map = texture;
        this.planeMat.color.set(0xffffff);
        this.planeMat.needsUpdate = true;

        const aspectRatio = img.width / img.height;
        const maxSize = 1.35;
        let scaleX;
        let scaleY;
        if (aspectRatio > 1) {
          scaleX = maxSize;
          scaleY = maxSize / aspectRatio;
        } else {
          scaleY = maxSize;
          scaleX = maxSize * aspectRatio;
        }
        this.imagePlane.scale.set(scaleX, scaleY, 1);
        this.imageFrame.scale.set(scaleX, scaleY, 1);
      };

      img.onerror = () => {
        this.planeMat.map = null;
        this.planeMat.color.set(0x495360);
        this.planeMat.needsUpdate = true;
      };

      img.src = url;
    }

    syncFromDom(force = false) {
      const nextState = {
        azimuth: clamp(getValue(ids.azimuth), 0, 315),
        elevation: clamp(getValue(ids.elevation), -30, 60),
        distance: clamp(getValue(ids.distance), 0, 10),
      };

      if (
        force ||
        nextState.azimuth !== this.state.azimuth ||
        nextState.elevation !== this.state.elevation ||
        nextState.distance !== this.state.distance
      ) {
        this.state = nextState;
        this.liveAzimuth = nextState.azimuth;
        this.liveElevation = nextState.elevation;
        this.liveDistance = nextState.distance;
        this.updateVisuals();
        this.updateStatus();
      }

      this.updateImage(getSourceImageSrc());
      this.onResize();
    }

    bindEvents() {
      this.boundPointerDown = this.onPointerDown.bind(this);
      this.boundPointerMove = this.onPointerMove.bind(this);
      this.boundPointerUp = this.onPointerUp.bind(this);
      this.boundResize = this.onResize.bind(this);

      const canvas = this.renderer.domElement;
      canvas.addEventListener('pointerdown', this.boundPointerDown);
      canvas.addEventListener('pointermove', this.boundPointerMove);
      canvas.addEventListener('pointerup', this.boundPointerUp);
      canvas.addEventListener('pointerleave', this.boundPointerUp);
      canvas.addEventListener('pointercancel', this.boundPointerUp);

      this.resizeObserver = new ResizeObserver(this.boundResize);
      this.resizeObserver.observe(this.shell);
      window.addEventListener('resize', this.boundResize);
    }

    getMousePos(event) {
      const rect = this.renderer.domElement.getBoundingClientRect();
      this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      };
    }

    setHandleScale(handle, scale) {
      handle.scale.setScalar(scale);
    }

    onPointerDown(event) {
      this.getMousePos(event);
      this.raycaster.setFromCamera(this.mouse, this.camera);

      const handles = [
        { mesh: this.azimuthHandle, name: 'azimuth' },
        { mesh: this.elevationHandle, name: 'elevation' },
        { mesh: this.distanceHandle, name: 'distance' },
      ];

      for (const handle of handles) {
        if (this.raycaster.intersectObject(handle.mesh).length > 0) {
          this.isDragging = true;
          this.dragTarget = handle.name;
          this.setHandleScale(handle.mesh, 1.25);
          this.renderer.domElement.style.cursor = 'grabbing';
          this.renderer.domElement.setPointerCapture(event.pointerId);
          return;
        }
      }
    }

    onPointerMove(event) {
      const pointer = this.getMousePos(event);
      this.raycaster.setFromCamera(this.mouse, this.camera);

      if (!this.isDragging) {
        const handles = [this.azimuthHandle, this.elevationHandle, this.distanceHandle];
        const hovered = handles.find((handle) => this.raycaster.intersectObject(handle).length > 0) || null;
        if (this.hoveredHandle && this.hoveredHandle !== hovered) {
          this.setHandleScale(this.hoveredHandle, 1);
        }
        if (hovered) {
          this.setHandleScale(hovered, 1.12);
          this.renderer.domElement.style.cursor = 'grab';
          this.hoveredHandle = hovered;
        } else {
          this.renderer.domElement.style.cursor = 'default';
          this.hoveredHandle = null;
        }
        return;
      }

      const intersect = new this.THREE.Vector3();
      if (this.dragTarget === 'azimuth') {
        const plane = new this.THREE.Plane(new this.THREE.Vector3(0, 1, 0), 0);
        if (this.raycaster.ray.intersectPlane(plane, intersect)) {
          let angle = Math.atan2(intersect.x, intersect.z) * (180 / Math.PI);
          if (angle < 0) {
            angle += 360;
          }
          angle = clamp(angle, 0, 315);
          this.liveAzimuth = angle;
          setSliderValue(ids.azimuth, Math.round(angle));
        }
      } else if (this.dragTarget === 'elevation') {
        const plane = new this.THREE.Plane(new this.THREE.Vector3(1, 0, 0), -this.ELEV_ARC_X);
        if (this.raycaster.ray.intersectPlane(plane, intersect)) {
          let angle = Math.atan2(intersect.y - this.CENTER.y, intersect.z) * (180 / Math.PI);
          angle = clamp(angle, -30, 60);
          this.liveElevation = angle;
          setSliderValue(ids.elevation, Math.round(angle));
        }
      } else if (this.dragTarget === 'distance') {
        const value = this.getClosestDistanceValue(pointer);
        this.liveDistance = value;
        setSliderValue(ids.distance, value.toFixed(1));
      }

      this.state = {
        azimuth: this.liveAzimuth,
        elevation: this.liveElevation,
        distance: this.liveDistance,
      };
      this.updateVisuals();
      this.updateStatus();
    }

    onPointerUp() {
      if (this.azimuthHandle) {
        this.setHandleScale(this.azimuthHandle, 1);
      }
      if (this.elevationHandle) {
        this.setHandleScale(this.elevationHandle, 1);
      }
      if (this.distanceHandle) {
        this.setHandleScale(this.distanceHandle, 1);
      }
      this.isDragging = false;
      this.dragTarget = null;
      this.renderer.domElement.style.cursor = this.hoveredHandle ? 'grab' : 'default';
    }

    onResize() {
      const width = this.canvasContainer.clientWidth;
      const height = this.canvasContainer.clientHeight;
      if (!width || !height) {
        return;
      }
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(width, height, false);
    }

    animate() {
      this.animationId = window.requestAnimationFrame(() => this.animate());
      this.time += 0.015;
      const pulse = 1 + (Math.sin(this.time * 2) * 0.03);
      if (!this.isDragging) {
        this.distanceHandle.scale.setScalar(pulse);
      }
      this.renderer.render(this.scene, this.camera);
    }

    dispose() {
      if (this.animationId) {
        window.cancelAnimationFrame(this.animationId);
      }
      if (this.resizeObserver) {
        this.resizeObserver.disconnect();
      }
      window.removeEventListener('resize', this.boundResize);
      if (this.renderer) {
        this.renderer.dispose();
      }
      this.shell.dataset.ready = '0';
    }
  }

  async function boot() {
    const host = document.querySelector(`#${ids.widgetHost} .mv-widget-shell`);
    if (!host) {
      return;
    }
    if (host.clientWidth < 120 || host.clientHeight < 120) {
      return;
    }

    try {
      const THREE = await loadThree();
      if (window.__multiAngleWidgetHost !== host) {
        if (window.__multiAngleWidgetInstance) {
          window.__multiAngleWidgetInstance.dispose();
        }
        window.__multiAngleWidgetHost = host;
        window.__multiAngleWidgetInstance = new MultiAngleWidget(host, THREE);
      }
      window.__multiAngleWidgetInstance.syncFromDom();
    } catch (error) {
      const loading = host.querySelector('.mv-widget-loading');
      if (loading) {
        loading.textContent = '3D 控制球资源加载失败，仍可使用右侧参数设置。';
      }
      console.error(error);
    }
  }

  function startBootLoop() {
    if (window.__multiAngleWidgetBootLoopStarted) {
      return;
    }
    window.__multiAngleWidgetBootLoopStarted = true;
    boot();
    setInterval(boot, 400);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startBootLoop, { once: true });
  } else {
    startBootLoop();
  }
})();
</script>
"""


def ensure_cuda_available() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "当前实例处于无卡模式，前端界面可以打开，但模型推理需要切换到有卡模式后再执行。"
        )



def load_controlnet_pipeline() -> FluxControlNetPipeline:
    global controlnet_pipe
    if controlnet_pipe is None:
        ensure_cuda_available()
        print("正在加载 FLUX.1-dev + ControlNet 模型，请稍候...")
        controlnet = FluxControlNetModel.from_pretrained(
            CONTROLNET_MODEL_PATH,
            torch_dtype=torch.bfloat16,
        )
        controlnet_pipe = FluxControlNetPipeline.from_pretrained(
            BASE_MODEL_PATH,
            controlnet=controlnet,
            torch_dtype=torch.bfloat16,
        )
        controlnet_pipe.to("cuda")
        print("ControlNet 管线加载完成。")
    return controlnet_pipe



def load_fill_pipeline() -> FluxFillPipeline:
    global fill_pipe
    if fill_pipe is None:
        ensure_cuda_available()
        print("正在加载 FLUX.1-Fill-dev 模型，请稍候...")
        fill_pipe = FluxFillPipeline.from_pretrained(
            FILL_MODEL_PATH,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        print("局部重绘管线加载完成。")
    return fill_pipe



def release_upstream_gpu_resources() -> None:
    global controlnet_pipe, fill_pipe

    released: list[str] = []

    if controlnet_pipe is not None:
        try:
            controlnet_pipe.to("cpu")
        except Exception:
            pass
        controlnet_pipe = None
        released.append("第一步场景生成模型")

    if fill_pipe is not None:
        try:
            fill_pipe.to("cpu")
        except Exception:
            pass
        fill_pipe = None
        released.append("第二步局部重绘模型")

    comfy_was_started_by_app = (
        comfy_started_by_app
        and comfy_process is not None
        and comfy_process.poll() is None
    )
    if comfy_was_started_by_app:
        shutdown_comfy_process()
        released.append("第三步 ComfyUI 进程")

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    if released:
        print("已为第四色步释放上游显存占用：" + "、".join(released))
        time.sleep(2)
    elif is_comfy_server_ready():
        print("检测到仍有外部 ComfyUI 服务在运行；如果第四色步仍然显存不足，请先手动关闭该进程后再试。")


def prepare_seed(seed: float | int | None, fallback: int = 42) -> int:
    raw_value = fallback if seed is None else int(seed)
    if raw_value < 0:
        return random.SystemRandom().randint(0, 2**31 - 1)
    return raw_value



def make_editor_value(image: Image.Image | None) -> dict[str, Any] | None:
    if image is None:
        return None
    rgba_image = image.convert("RGBA")
    return {
        "background": rgba_image,
        "layers": [],
        "composite": rgba_image,
    }



def merge_mask_layers(editor_value: dict[str, Any] | None, size: tuple[int, int]) -> Image.Image:
    width, height = size
    merged = np.zeros((height, width), dtype=np.uint8)

    if not editor_value:
        return Image.fromarray(merged, mode="L")

    for layer in editor_value.get("layers", []) or []:
        if layer is None:
            continue
        layer_rgba = layer.convert("RGBA") if isinstance(layer, Image.Image) else Image.fromarray(layer).convert("RGBA")
        alpha = np.array(layer_rgba.getchannel("A"), dtype=np.uint8)
        merged = np.maximum(merged, alpha)

    merged = np.where(merged > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(merged, mode="L")


def resolve_stage_input_image(
    default_image: Image.Image | None,
    manual_image: Image.Image | None,
) -> Image.Image | None:
    return manual_image if manual_image is not None else default_image


def load_uploaded_trellis_entries(uploaded_files: Any) -> list[tuple[Image.Image, str]]:
    if not uploaded_files:
        return []

    files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    entries: list[tuple[Image.Image, str]] = []

    for index, file_item in enumerate(files):
        raw_path = getattr(file_item, "name", file_item)
        file_path = Path(str(raw_path))
        with Image.open(file_path) as uploaded_image:
            entries.append(
                (
                    uploaded_image.convert("RGB").copy(),
                    f"手动上传输入 {index + 1} | {file_path.name}",
                )
            )

    return entries



def multiview_placeholder_status() -> str:
    return "请先在第二步确认当前结果，第三步才会解锁；解锁后也可手动上传图像。"


def reset_multiview_stage(status: str | None = None):
    return (
        gr.update(visible=False),
        None,
        None,
        None,
        None,
        [],
        [],
        status or multiview_placeholder_status(),
        None,
    )


def unlock_multiview_stage(source_image: Image.Image | None, status: str | None = None):
    if source_image is not None:
        resolved_status = status or "第三步已解锁。你现在可以调整相机参数并开始生成多视角图像。"
    else:
        resolved_status = (
            status
            or "第三步已解锁。当前没有默认输入，你可以手动上传第三步输入图像后继续；"
            "若只想使用第四色步，也可直接点击确认按钮继续。"
        )

    return (
        gr.update(visible=True),
        source_image,
        source_image,
        None,
        None,
        [],
        [],
        resolved_status,
        None,
    )


def format_multiview_prompt(azimuth: float | int, elevation: float | int, distance: float | int) -> str:
    angle = int(azimuth) % 360
    vertical = int(elevation)
    zoom = float(distance)

    if angle < 22.5 or angle >= 337.5:
        azimuth_text = "front view"
    elif angle < 67.5:
        azimuth_text = "front-right quarter view"
    elif angle < 112.5:
        azimuth_text = "right side view"
    elif angle < 157.5:
        azimuth_text = "back-right quarter view"
    elif angle < 202.5:
        azimuth_text = "back view"
    elif angle < 247.5:
        azimuth_text = "back-left quarter view"
    elif angle < 292.5:
        azimuth_text = "left side view"
    else:
        azimuth_text = "front-left quarter view"

    if vertical < -15:
        elevation_text = "low-angle shot"
    elif vertical < 15:
        elevation_text = "eye-level shot"
    elif vertical < 45:
        elevation_text = "elevated shot"
    else:
        elevation_text = "high-angle shot"

    if zoom < 2:
        distance_text = "wide shot"
    elif zoom < 6:
        distance_text = "medium shot"
    else:
        distance_text = "close-up"

    return f"<sks> {azimuth_text} {elevation_text} {distance_text}"



def available_model_files(model_type: str) -> list[str]:
    return MULTIVIEW_MODEL_CHOICES.get(model_type, MULTIVIEW_MODEL_CHOICES["GGUF"])



def update_multiview_model_file(model_type: str):
    choices = available_model_files(model_type)
    return gr.update(choices=choices, value=choices[0])



def ensure_multiview_assets(model_type: str, model_file: str) -> None:
    required_paths = [
        COMFY_ROOT / "models" / "text_encoders" / MULTIVIEW_CLIP_MODEL,
        COMFY_ROOT / "models" / "vae" / MULTIVIEW_VAE_MODEL,
        COMFY_ROOT / "models" / "loras" / MULTIVIEW_ANGLE_LORA,
        COMFY_ROOT / "models" / "loras" / MULTIVIEW_LIGHTNING_LORA,
    ]

    if model_type == "GGUF":
        required_paths.append(COMFY_ROOT / "models" / "unet" / model_file)
    elif model_type == "Diffusion":
        required_paths.append(COMFY_ROOT / "models" / "diffusion_models" / model_file)
    else:
        raise RuntimeError(f"不支持的模型类型：{model_type}")

    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError("缺少第三步所需模型文件：\n" + "\n".join(missing))



def is_comfy_server_ready() -> bool:
    try:
        response = requests.get(COMFY_HISTORY_URL, timeout=2)
        return response.ok
    except requests.RequestException:
        return False



def shutdown_comfy_process() -> None:
    global comfy_process, comfy_log_handle, comfy_started_by_app

    if comfy_started_by_app and comfy_process is not None and comfy_process.poll() is None:
        comfy_process.terminate()
        try:
            comfy_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            comfy_process.kill()
            comfy_process.wait(timeout=5)

    if comfy_log_handle is not None and not comfy_log_handle.closed:
        comfy_log_handle.close()

    comfy_process = None
    comfy_log_handle = None
    comfy_started_by_app = False


atexit.register(shutdown_comfy_process)



def ensure_comfy_server(timeout_seconds: int = 180) -> None:
    global comfy_process, comfy_log_handle, comfy_started_by_app

    if is_comfy_server_ready():
        return

    if comfy_process is None or comfy_process.poll() is not None:
        COMFY_STARTUP_LOG.parent.mkdir(parents=True, exist_ok=True)
        if comfy_log_handle is not None and not comfy_log_handle.closed:
            comfy_log_handle.close()
        comfy_log_handle = COMFY_STARTUP_LOG.open("a", encoding="utf-8")

        env = os.environ.copy()
        if os.environ.get("HF_ENDPOINT"):
            env["HF_ENDPOINT"] = os.environ["HF_ENDPOINT"]
        comfy_process = subprocess.Popen(
            [
                sys.executable,
                str(COMFY_MAIN_PATH),
                "--listen",
                COMFY_BIND_HOST,
                "--port",
                str(COMFY_PORT),
                "--disable-xformers",
            ],
            cwd=str(COMFY_ROOT),
            stdout=comfy_log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        comfy_started_by_app = True

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if is_comfy_server_ready():
            return
        if comfy_process is not None and comfy_process.poll() is not None:
            raise RuntimeError(f"ComfyUI 启动失败，日志见：{COMFY_STARTUP_LOG}")
        time.sleep(2)

    raise RuntimeError(f"ComfyUI 启动超时，日志见：{COMFY_STARTUP_LOG}")



def upload_image_to_comfy(image: Image.Image) -> str:
    image_to_upload = image.convert("RGB")
    buffer = io.BytesIO()
    image_to_upload.save(buffer, format="PNG")
    buffer.seek(0)
    filename = f"gradio_multiangle_{uuid.uuid4().hex}.png"

    response = requests.post(
        COMFY_UPLOAD_URL,
        files={"image": (filename, buffer.getvalue(), "image/png")},
        data={"type": "input", "overwrite": "true"},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["name"]



def build_multiview_comfy_prompt(
    uploaded_filename: str,
    model_type: str,
    model_file: str,
    prompt_text: str,
    steps: int,
    cfg: float,
    seed: int,
) -> dict[str, Any]:
    if model_type == "GGUF":
        loader_class = "UnetLoaderGGUF"
        loader_inputs: dict[str, Any] = {"unet_name": model_file}
    else:
        loader_class = "UNETLoader"
        loader_inputs = {
            "unet_name": model_file,
            "weight_dtype": "default",
        }

    return {
        "1": {
            "class_type": loader_class,
            "inputs": loader_inputs,
        },
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": MULTIVIEW_ANGLE_LORA,
                "strength_model": 1,
            },
        },
        "3": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["2", 0],
                "lora_name": MULTIVIEW_LIGHTNING_LORA,
                "strength_model": 1,
            },
        },
        "4": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": MULTIVIEW_CLIP_MODEL,
                "type": "qwen_image",
                "device": "default",
            },
        },
        "5": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": MULTIVIEW_VAE_MODEL,
            },
        },
        "6": {
            "class_type": "LoadImage",
            "inputs": {
                "image": uploaded_filename,
            },
        },
        "7": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "clip": ["4", 0],
                "vae": ["5", 0],
                "image1": ["6", 0],
                "prompt": "",
            },
        },
        "8": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "clip": ["4", 0],
                "vae": ["5", 0],
                "image1": ["6", 0],
                "prompt": prompt_text,
            },
        },
        "9": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "model": ["3", 0],
                "shift": 3.1,
            },
        },
        "10": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "conditioning": ["8", 0],
                "reference_latents_method": "index_timestep_zero",
            },
        },
        "11": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "conditioning": ["7", 0],
                "reference_latents_method": "index_timestep_zero",
            },
        },
        "12": {
            "class_type": "CFGNorm",
            "inputs": {
                "model": ["9", 0],
                "strength": 1,
            },
        },
        "13": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["6", 0],
                "vae": ["5", 0],
            },
        },
        "14": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["12", 0],
                "positive": ["10", 0],
                "negative": ["11", 0],
                "latent_image": ["13", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1,
            },
        },
        "15": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["14", 0],
                "vae": ["5", 0],
            },
        },
        "16": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["15", 0],
                "filename_prefix": "Qwen_Edit_2511",
            },
        },
    }



def queue_comfy_prompt(prompt: dict[str, Any]) -> str:
    prompt_id = str(uuid.uuid4())
    response = requests.post(
        COMFY_PROMPT_URL,
        json={"prompt": prompt, "prompt_id": prompt_id},
        timeout=60,
    )

    if not response.ok:
        try:
            payload = response.json()
        except ValueError as exc:
            response.raise_for_status()
            raise RuntimeError("ComfyUI 返回了无法解析的错误响应。") from exc

        error = payload.get("error")
        node_errors = payload.get("node_errors")
        raise RuntimeError(f"ComfyUI 提示词校验失败：{error}；节点错误：{node_errors}")

    payload = response.json()
    return payload["prompt_id"]



def wait_for_comfy_completion(prompt_id: str, timeout_seconds: int = 900) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = requests.get(f"{COMFY_HISTORY_URL}/{prompt_id}", timeout=20)
        response.raise_for_status()
        payload = response.json()
        if prompt_id in payload:
            return payload[prompt_id]
        time.sleep(1.5)

    raise RuntimeError("等待 ComfyUI 生成结果超时。")



def fetch_comfy_images(history_item: dict[str, Any]) -> list[Image.Image]:
    images: list[Image.Image] = []
    outputs = history_item.get("outputs", {})
    for node_id in sorted(outputs.keys(), key=lambda item: int(item)):
        node_output = outputs[node_id]
        for image_info in node_output.get("images", []):
            response = requests.get(COMFY_VIEW_URL, params=image_info, timeout=120)
            response.raise_for_status()
            images.append(Image.open(io.BytesIO(response.content)).convert("RGB"))
    return images



def run_multiview_generation(
    source_image: Image.Image,
    model_type: str,
    model_file: str,
    azimuth: float | int,
    elevation: float | int,
    distance: float | int,
    steps: float | int,
    cfg: float | int,
    seed: float | int | None,
) -> tuple[Image.Image, int, str]:
    ensure_cuda_available()
    ensure_multiview_assets(model_type, model_file)
    ensure_comfy_server()

    prompt_text = format_multiview_prompt(azimuth, elevation, distance)
    seed_value = prepare_seed(seed, fallback=-1)
    uploaded_filename = upload_image_to_comfy(source_image)
    prompt = build_multiview_comfy_prompt(
        uploaded_filename=uploaded_filename,
        model_type=model_type,
        model_file=model_file,
        prompt_text=prompt_text,
        steps=int(steps),
        cfg=float(cfg),
        seed=seed_value,
    )
    prompt_id = queue_comfy_prompt(prompt)
    history_item = wait_for_comfy_completion(prompt_id)
    images = fetch_comfy_images(history_item)
    if not images:
        raise RuntimeError("ComfyUI 已执行完成，但未返回任何图像输出。")
    return images[-1], seed_value, prompt_text



def generate_scene(sketch_image: Image.Image, prompt: str, seed: float | int | None):
    if sketch_image is None:
        raise gr.Error("请先上传草图。")
    if not prompt or not prompt.strip():
        raise gr.Error("请先输入提示词。")

    pipe = load_controlnet_pipeline()
    seed_value = prepare_seed(seed, fallback=42)
    width, height = sketch_image.size

    gallery_items = []
    generated_images = []
    print(f"开始批量生成，共计 {len(PARAMETER_GRID)} 张图像...")

    for control_guidance_end, num_inference_steps, guidance_scale in PARAMETER_GRID:
        print(
            "正在生成:",
            f"control_guidance_end={control_guidance_end},",
            f"num_inference_steps={num_inference_steps},",
            f"guidance_scale={guidance_scale}",
        )
        generator = torch.Generator(device="cuda").manual_seed(seed_value)
        image = pipe(
            prompt=prompt,
            control_image=sketch_image,
            width=width,
            height=height,
            controlnet_conditioning_scale=DEFAULT_CONTROLNET_CONDITIONING_SCALE,
            control_guidance_end=control_guidance_end,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        label = (
            f"End:{control_guidance_end} | Steps:{num_inference_steps} | "
            f"Scale:{guidance_scale}"
        )
        gallery_items.append((image, label))
        generated_images.append(image)

    selected_message = (
        f"候选场景图生成完成，共 {len(PARAMETER_GRID)} 张。"
        f"本轮实际种子：`{seed_value}`。"
        "请先从下方结果中点击一张，再点击“确认当前所选场景图，进入第二步”；"
        "如果你想跳过第一步，也可以直接点击确认按钮进入第二步后手动上传图像。"
    )
    return (
        gallery_items,
        generated_images,
        None,
        None,
        None,
        None,
        None,
        gr.update(visible=False),
        selected_message,
        prompt,
        seed_value,
        None,
        "请先在第一步从候选图中选择一张场景图，再点击确认按钮进入第二步。",
        *reset_multiview_stage(),
        gr.update(value=seed_value),
        gr.update(interactive=True),
    )


def select_generated_image(
    generated_images: list[Image.Image] | None,
    generation_seed: int | None,
    evt: gr.SelectData,
):
    if not generated_images:
        raise gr.Error("请先生成候选场景图。")

    index = evt.index if isinstance(evt.index, int) else evt.index[0]
    selected_image = generated_images[index]
    editor_value = make_editor_value(selected_image)
    message = f"已选择第 {index + 1} 张图像。请点击“确认当前所选场景图，进入第二步”。"

    return (
        selected_image,
        selected_image,
        None,
        editor_value,
        gr.update(visible=False),
        message,
        None,
        selected_image,
        "当前选择已更新。确认后才会进入第二步；如果重新选图，需要再次确认。",
        *reset_multiview_stage(),
        gr.update(value=generation_seed if generation_seed is not None else 42),
        gr.update(interactive=True),
    )


def confirm_scene_selection(
    selected_image: Image.Image | None,
    generation_seed: int | None,
):
    if selected_image is None:
        return (
            None,
            None,
            None,
            None,
            gr.update(visible=True),
            "已直接进入第二步。你现在可以在第二步手动上传图像，或返回第一步先生成并选择一张场景图。",
            None,
            "当前第二步暂无默认输入。你可以在此处手动上传图像后直接确认进入第三步，"
            "或先涂抹并执行局部重绘；如果你只想使用第三步，也可以稍后直接点击确认按钮进入第三步。",
            *reset_multiview_stage(),
            gr.update(value=generation_seed if generation_seed is not None else 42),
        )

    editor_value = make_editor_value(selected_image)
    return (
        selected_image,
        None,
        editor_value,
        selected_image,
        gr.update(visible=True),
        "已确认当前所选场景图，第二步已解锁。",
        None,
        "当前待确认的第二步结果已载入为所选场景图。若无需局部重绘，可直接点击“确认当前第二步结果，进入第三步”；若需要修改，请先涂抹并执行局部重绘。",
        *reset_multiview_stage(),
        gr.update(value=generation_seed if generation_seed is not None else 42),
    )


def load_manual_inpaint_input(
    manual_image: Image.Image | None,
    selected_image: Image.Image | None,
):
    active_image = resolve_stage_input_image(selected_image, manual_image)

    if manual_image is not None:
        status = (
            "已载入手动上传的第二步输入图像。若无需局部重绘，可直接点击“确认当前第二步结果，进入第三步”；"
            "若需要修改，请先涂抹并执行局部重绘。"
        )
    elif selected_image is not None:
        status = (
            "已清除手动上传图像，第二步已恢复为使用第一步确认的场景图。"
            "若无需局部重绘，可直接点击“确认当前第二步结果，进入第三步”。"
        )
    else:
        status = "当前第二步暂无可用输入。请手动上传图像，或返回第一步确认一张场景图。"

    return (
        make_editor_value(active_image),
        active_image,
        None,
        status,
        *reset_multiview_stage(),
    )


def use_original_image(
    selected_image: Image.Image | None,
    manual_image: Image.Image | None,
):
    source_image = resolve_stage_input_image(selected_image, manual_image)
    if source_image is None:
        raise gr.Error("请先在第二步手动上传图像，或在第一步选择一张场景图。")

    if manual_image is not None:
        status = "已将手动上传图像设为第二步结果。若确认使用它进入第三步，请点击“确认当前第二步结果，进入第三步”。"
    else:
        status = "已将当前选图设为第二步结果。若确认使用它进入第三步，请点击“确认当前第二步结果，进入第三步”。"
    return (
        source_image,
        None,
        status,
        *reset_multiview_stage(),
    )


def run_inpainting(
    editor_value: dict[str, Any] | None,
    inpaint_prompt: str,
    inpaint_seed: float | int | None,
    selected_image: Image.Image | None,
    manual_input_image: Image.Image | None,
    generation_prompt: str,
    generation_seed: int | None,
):
    default_image = resolve_stage_input_image(selected_image, manual_input_image)
    if default_image is None:
        raise gr.Error("请先在第二步准备输入图像：要么使用第一步结果，要么手动上传一张图像。")

    if editor_value and editor_value.get("background") is not None:
        source_image = editor_value["background"].convert("RGB")
    else:
        source_image = default_image.convert("RGB")

    mask_image = merge_mask_layers(editor_value, source_image.size)
    mask_pixels = np.array(mask_image, dtype=np.uint8)
    has_mask = bool(np.any(mask_pixels > 0))

    if not has_mask:
        return (
            source_image,
            mask_image,
            "未检测到涂抹区域。请先涂抹后重试，或直接使用当前输入作为第二步结果，再点击确认按钮进入第三步。",
            *reset_multiview_stage(),
        )

    prompt_to_use = (inpaint_prompt or "").strip() or (generation_prompt or "").strip()
    if not prompt_to_use:
        raise gr.Error("当前没有可回退的第一步提示词，请先输入局部重绘提示词。")

    status_prefix = ""
    if not (inpaint_prompt or "").strip() and (generation_prompt or "").strip():
        status_prefix = "未输入局部重绘提示词，已回退为第一步的场景提示词。"

    pipe = load_fill_pipeline()
    seed_value = prepare_seed(inpaint_seed, fallback=generation_seed if generation_seed is not None else 42)
    mask_rgb = mask_image.convert("RGB")
    generator = torch.Generator("cpu").manual_seed(seed_value)

    result = pipe(
        prompt=prompt_to_use,
        image=source_image,
        mask_image=mask_rgb,
        height=source_image.height,
        width=source_image.width,
        guidance_scale=30,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=generator,
    ).images[0]

    status = (
        status_prefix
        + f"局部重绘完成，本次实际种子：`{seed_value}`。若结果满意，请点击“确认当前第二步结果，进入第三步”。"
    )
    return (
        result,
        mask_image,
        status,
        *reset_multiview_stage(),
    )


def confirm_inpaint_result(final_image: Image.Image | None):
    if final_image is None:
        return unlock_multiview_stage(
            None,
            "已直接进入第三步。当前没有默认输入，你可以手动上传第三步输入图像后再生成多视角图像；"
            "如果你只想使用第四色步，也可以直接点击确认按钮继续。",
        )

    return unlock_multiview_stage(final_image)


def generate_multiview_image(
    source_image: Image.Image | None,
    manual_source_image: Image.Image | None,
    model_type: str,
    model_file: str,
    azimuth: float | int,
    elevation: float | int,
    distance: float | int,
    steps: float | int,
    cfg: float | int,
    seed: float | int | None,
    gallery_entries: list[tuple[Image.Image, str]] | None,
):
    active_source_image = resolve_stage_input_image(source_image, manual_source_image)
    if active_source_image is None:
        raise gr.Error("请先在第三步准备输入图像：要么使用第二步结果，要么手动上传一张图像。")

    try:
        result_image, used_seed, prompt_text = run_multiview_generation(
            source_image=active_source_image,
            model_type=model_type,
            model_file=model_file,
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            steps=steps,
            cfg=cfg,
            seed=seed,
        )
    except Exception as exc:
        raise gr.Error(f"第三步多视角生成失败：{exc}") from exc

    entries = list(gallery_entries or [])
    label = (
        f"{model_type} | Az:{int(azimuth)} | El:{int(elevation)} | Dist:{float(distance):.1f} | "
        f"Steps:{int(steps)} | CFG:{float(cfg):.1f} | Seed:{used_seed}"
    )
    entries.append((result_image, label))
    input_source_text = "手动上传的第三步输入图像" if manual_source_image is not None else "第二步结果"
    status = (
        f"多视角图像生成完成。当前使用：`{input_source_text}`；角度提示词：`{prompt_text}`；"
        f"本次实际种子：`{used_seed}`。"
        "若要进入第四色步，请点击“确认当前第三步结果，进入第四色步”。"
    )
    return result_image, entries, entries, status, len(entries) - 1


def load_manual_multiview_input(
    manual_image: Image.Image | None,
    source_image: Image.Image | None,
):
    if manual_image is not None:
        status = (
            "已载入手动上传的第三步输入图像。后续多视角生成与确认将优先使用这张图像；"
            "请重新生成多视角图像后再进入第四色步。"
        )
    elif source_image is not None:
        status = "已清除手动上传图像，第三步已恢复为使用第二步结果。请重新生成多视角图像后再进入第四色步。"
    else:
        status = (
            "当前第三步暂无默认输入。你可以手动上传一张图像后开始生成多视角图像；"
            "如果你只想使用第四色步，也可以直接点击确认按钮继续。"
        )

    return None, [], [], status, None


def select_multiview_image(
    gallery_entries: list[tuple[Image.Image, str]] | None,
    evt: gr.SelectData,
):
    if not gallery_entries:
        raise gr.Error("当前没有可选的多视角结果。")

    index = evt.index if isinstance(evt.index, int) else evt.index[0]
    image, label = gallery_entries[index]
    return index, image, f"已选中第 {index + 1} 张多视角图像：`{label}`。"



def delete_multiview_image(
    gallery_entries: list[tuple[Image.Image, str]] | None,
    selected_index: int | None,
):
    if not gallery_entries:
        raise gr.Error("当前没有可删除的多视角结果。")
    if selected_index is None:
        raise gr.Error("请先在多视角预览模块中点击一张图像，再执行删除。")

    entries = list(gallery_entries)
    entries.pop(selected_index)
    latest_image = entries[-1][0] if entries else None
    status = (
        "已删除所选多视角图像。若要进入第四色步，请重新点击确认按钮。"
        if entries
        else "已删除所选多视角图像，当前预览区已清空。若要直接使用正视角进入第四色步，请重新点击确认按钮。"
    )
    return entries, entries, latest_image, status, None



def clear_multiview_gallery():
    return [], [], None, "已清空当前多视角预览模块中的全部图像。若要直接使用正视角进入第四色步，请重新点击确认按钮。", None



def trellis_placeholder_status() -> str:
    return "请先在第三步确认当前结果，第四色步才会解锁；解锁后也可手动上传图像。"


def reset_trellis_result_outputs(status: str):
    return (
        None,
        None,
        None,
        status,
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(value=None, interactive=False),
        gr.update(value=None, interactive=False),
    )


def reset_trellis_stage(status: str | None = None):
    return (
        gr.update(visible=False),
        [],
        [],
        None,
        [],
        [],
        *reset_trellis_result_outputs(status or trellis_placeholder_status()),
    )


def build_trellis_input_entries(
    source_image: Image.Image | None,
    gallery_entries: list[tuple[Image.Image, str]] | None,
) -> list[tuple[Image.Image, str]]:
    entries: list[tuple[Image.Image, str]] = []
    if source_image is not None:
        entries.append((source_image, "正视角输入（固定包含）"))
    elif gallery_entries:
        raise gr.Error("第三步输入状态异常：存在多视角结果，但缺少对应的基准输入图像。")

    for index, (image, label) in enumerate(gallery_entries or []):
        entries.append((image, f"多视角输入 {index + 1} | {label}"))
    return entries


def confirm_multiview_result(
    source_image: Image.Image | None,
    manual_source_image: Image.Image | None,
    gallery_entries: list[tuple[Image.Image, str]] | None,
):
    effective_source_image = resolve_stage_input_image(source_image, manual_source_image)
    entries = build_trellis_input_entries(effective_source_image, gallery_entries)
    if not entries:
        status = "已直接进入第四色步。当前没有默认输入，请在第四色步手动上传一张或多张图像后再生成 3D 资源。"
    elif len(entries) > 1:
        extra_count = len(entries) - 1
        status = (
            f"已确认当前第三步结果，第四色步已解锁。"
            f"本次将使用 1 张正视角图像和 {extra_count} 张多视角图像作为 TRELLIS 输入。"
        )
    else:
        status = "已确认当前第三步结果，第四色步已解锁。本次将只使用正视角图像作为 TRELLIS 输入。"

    return (
        gr.update(visible=True),
        entries,
        entries,
        None,
        entries,
        entries,
        *reset_trellis_result_outputs(status),
    )


def update_trellis_manual_inputs(
    uploaded_files: Any,
    default_entries: list[tuple[Image.Image, str]] | None,
):
    manual_entries = load_uploaded_trellis_entries(uploaded_files)
    if manual_entries:
        status = (
            f"已载入 {len(manual_entries)} 张手动上传的第四色步输入图像。"
            "当前 TRELLIS 将只使用这些手动上传图像；第三步默认输入仍保留在参考区展示。"
        )
        return manual_entries, manual_entries, *reset_trellis_result_outputs(status)

    restored_entries = list(default_entries or [])
    if restored_entries:
        extra_count = max(len(restored_entries) - 1, 0)
        if extra_count > 0:
            status = (
                f"已清空手动上传图像，第四色步已恢复为默认输入："
                f"1 张正视角图像和 {extra_count} 张多视角图像。"
            )
        else:
            status = "已清空手动上传图像，第四色步已恢复为默认输入：1 张正视角图像。"
        return restored_entries, restored_entries, *reset_trellis_result_outputs(status)

    return [], [], *reset_trellis_result_outputs("当前第四色步没有默认输入。请手动上传一张或多张图像后再生成 3D 资源。")


def ensure_trellis_dirs() -> None:
    for trellis_path in (TRELLIS_RUNTIME_ROOT, TRELLIS_CACHE_ROOT):
        trellis_path.mkdir(parents=True, exist_ok=True)



def resolve_trellis_python() -> Path:
    override = os.environ.get("TRELLIS_PYTHON_BIN")
    if override:
        path = _env_path("TRELLIS_PYTHON_BIN", ".venv/bin/python")
        if path.exists():
            return path
        raise RuntimeError(f"TRELLIS_PYTHON_BIN 指向的 Python 不存在：{path}")

    if TRELLIS_ENV_PYTHON.exists():
        return TRELLIS_ENV_PYTHON

    current_python = Path(sys.executable)
    if current_python.exists():
        return current_python

    raise RuntimeError(
        "未找到可用的 TRELLIS Python 解释器。请设置环境变量 `TRELLIS_PYTHON_BIN`。"
    )



def resolve_trellis_model_path() -> str:
    model_path = TRELLIS_LOCAL_MODEL_PATH
    pipeline_path = model_path / "pipeline.json"
    if not model_path.is_dir():
        raise RuntimeError(
            "未找到本地 TRELLIS-image-large 模型目录："
            f"{model_path}。请先准备该目录，或设置 SKETCH_PLATFORM_TRELLIS_MODEL。"
        )
    model_root = model_path.resolve()
    try:
        pipeline_path.resolve().relative_to(model_root)
    except ValueError as exc:
        raise RuntimeError(
            f"本地 TRELLIS pipeline.json 路径越界：{pipeline_path}。"
        ) from exc
    if not pipeline_path.is_file():
        raise RuntimeError(
            "本地 TRELLIS-image-large 模型缺少 pipeline.json："
            f"{pipeline_path}。不会从远程仓库或缓存补充模型。"
        )

    try:
        with pipeline_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"本地 TRELLIS pipeline.json 无法读取或解析：{pipeline_path}。"
        ) from exc

    args = config.get("args") if isinstance(config, dict) else None
    model_names = args.get("models") if isinstance(args, dict) else None
    if not isinstance(model_names, dict) or not model_names:
        raise RuntimeError(
            f"本地 TRELLIS pipeline.json 未包含有效的 models 配置：{pipeline_path}。"
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
            "本地 TRELLIS-image-large 模型文件不完整；不会从远程仓库或缓存补充：\n"
            + "\n".join(missing)
        )

    return str(model_path.resolve())



def build_trellis_env() -> dict[str, str]:
    ensure_trellis_dirs()

    env = os.environ.copy()
    hf_home = TRELLIS_CACHE_ROOT / "huggingface"
    torch_home = TRELLIS_CACHE_ROOT / "torch"
    xdg_cache_home = TRELLIS_CACHE_ROOT / "xdg"
    u2net_home = TRELLIS_CACHE_ROOT / "u2net"

    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{TRELLIS_PROJECT_ROOT}{os.pathsep}{pythonpath}"
        if pythonpath
        else str(TRELLIS_PROJECT_ROOT)
    )
    env["SPCONV_ALGO"] = "native"
    env["OMP_NUM_THREADS"] = "1"
    env["ATTN_BACKEND"] = os.environ.get("TRELLIS_ATTN_BACKEND", "sdpa")
    env["SPARSE_ATTN_BACKEND"] = env["ATTN_BACKEND"]
    env["HF_HOME"] = str(hf_home)
    env["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    env["TORCH_HOME"] = str(torch_home)
    env["XDG_CACHE_HOME"] = str(xdg_cache_home)
    env["U2NET_HOME"] = str(u2net_home)
    if os.environ.get("HF_ENDPOINT"):
        env["HF_ENDPOINT"] = os.environ["HF_ENDPOINT"]
    env["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )
    return env



def save_trellis_inputs(
    job_dir: Path,
    input_entries: list[tuple[Image.Image, str]] | None,
) -> list[Path]:
    input_dir = job_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for index, (image, _) in enumerate(input_entries or []):
        if image is None:
            continue
        image_to_save = image if isinstance(image, Image.Image) else Image.fromarray(image)
        path = input_dir / f"input_{index:02d}.png"
        image_to_save.convert("RGBA").save(path)
        saved_paths.append(path)

    if not saved_paths:
        raise RuntimeError("未能整理出任何可用于 TRELLIS 的输入图像。")
    return saved_paths



def run_trellis_runner(arguments: list[str], timeout_seconds: int) -> dict[str, Any]:
    python_bin = resolve_trellis_python()
    if not TRELLIS_RUNNER_PATH.exists():
        raise RuntimeError(f"缺少 TRELLIS 辅助脚本：{TRELLIS_RUNNER_PATH}")

    try:
        process = subprocess.run(
            [str(python_bin), str(TRELLIS_RUNNER_PATH), *arguments],
            cwd=str(TRELLIS_WORKDIR),
            env=build_trellis_env(),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"TRELLIS 运行超时（>{timeout_seconds} 秒）。") from exc

    combined_output = "\n".join(part for part in (process.stdout, process.stderr) if part)
    payload = None
    for line in reversed(combined_output.splitlines()):
        if line.startswith(TRELLIS_RESULT_PREFIX):
            payload = json.loads(line[len(TRELLIS_RESULT_PREFIX):])
            break

    if payload is None:
        log_excerpt = "\n".join(combined_output.splitlines()[-20:])
        raise RuntimeError("TRELLIS 返回结果不可解析。最近日志如下：\n" + log_excerpt)

    if process.returncode != 0 or not payload.get("ok"):
        error_message = payload.get("error") or "TRELLIS 运行失败。"
        traceback_text = payload.get("traceback") or "\n".join(combined_output.splitlines()[-20:])
        raise RuntimeError(f"{error_message}\n{traceback_text}")

    return payload



def generate_trellis_asset(
    input_entries: list[tuple[Image.Image, str]] | None,
    seed: float | int | None,
    ss_guidance_strength: float | int,
    ss_sampling_steps: float | int,
    slat_guidance_strength: float | int,
    slat_sampling_steps: float | int,
    multiimage_algo: str,
    request: gr.Request,
):
    if not input_entries:
        raise gr.Error("请先在第四色步准备输入图像。若默认输入为空，请先手动上传一张或多张图像。")

    ensure_cuda_available()
    release_upstream_gpu_resources()
    ensure_trellis_dirs()

    session_hash = getattr(request, "session_hash", None) or "anonymous"
    job_dir = TRELLIS_RUNTIME_ROOT / session_hash / f"job_{uuid.uuid4().hex}"
    job_dir.mkdir(parents=True, exist_ok=True)

    input_paths = save_trellis_inputs(job_dir, input_entries)
    used_seed = prepare_seed(seed, fallback=-1)

    runner_args = [
        "generate",
        "--job-dir",
        str(job_dir),
        "--model-source",
        resolve_trellis_model_path(),
        "--seed",
        str(used_seed),
        "--ss-guidance",
        str(float(ss_guidance_strength)),
        "--ss-steps",
        str(int(ss_sampling_steps)),
        "--slat-guidance",
        str(float(slat_guidance_strength)),
        "--slat-steps",
        str(int(slat_sampling_steps)),
        "--multiimage-algo",
        multiimage_algo,
    ]
    for input_path in input_paths:
        runner_args.extend(["--image", str(input_path)])

    try:
        payload = run_trellis_runner(runner_args, timeout_seconds=7200)
    except Exception as exc:
        raise gr.Error(f"TRELLIS 3D 资源生成失败：{exc}") from exc

    result_state = {
        "job_dir": str(job_dir),
        "state_path": payload["state_path"],
        "video_path": payload["video_path"],
        "glb_path": None,
        "gaussian_path": None,
        "seed": used_seed,
        "input_count": len(input_paths),
    }
    status = (
        f"TRELLIS 3D 资源生成完成，本次实际种子：`{used_seed}`；"
        f"输入图像数：`{len(input_paths)}`。现在可以继续提取 GLB 或高斯结果。"
    )
    return (
        result_state,
        payload["video_path"],
        None,
        status,
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(value=None, interactive=False),
        gr.update(value=None, interactive=False),
    )



def extract_trellis_glb(
    result_state: dict[str, Any] | None,
    mesh_simplify: float | int,
    texture_size: float | int,
):
    if not result_state or not result_state.get("state_path"):
        raise gr.Error("请先生成 TRELLIS 3D 资源，再提取 GLB。")

    state_path = Path(result_state["state_path"])
    output_path = state_path.with_name(
        f"trellis_mesh_s{float(mesh_simplify):.2f}_t{int(texture_size)}.glb"
    )

    try:
        payload = run_trellis_runner(
            [
                "extract-glb",
                "--state-path",
                str(state_path),
                "--output-path",
                str(output_path),
                "--mesh-simplify",
                str(float(mesh_simplify)),
                "--texture-size",
                str(int(texture_size)),
            ],
            timeout_seconds=3600,
        )
    except Exception as exc:
        raise gr.Error(f"TRELLIS GLB 提取失败：{exc}") from exc

    updated_state = dict(result_state)
    updated_state["glb_path"] = payload["glb_path"]
    status = (
        f"GLB 提取完成。简化比例：`{float(mesh_simplify):.2f}`；"
        f"纹理尺寸：`{int(texture_size)}`。"
    )
    return updated_state, payload["glb_path"], gr.update(value=payload["glb_path"], interactive=True), status



def extract_trellis_gaussian(result_state: dict[str, Any] | None):
    if not result_state or not result_state.get("state_path"):
        raise gr.Error("请先生成 TRELLIS 3D 资源，再提取高斯结果。")

    state_path = Path(result_state["state_path"])
    output_path = state_path.with_name("trellis_gaussian.ply")

    try:
        payload = run_trellis_runner(
            [
                "extract-gaussian",
                "--state-path",
                str(state_path),
                "--output-path",
                str(output_path),
            ],
            timeout_seconds=1800,
        )
    except Exception as exc:
        raise gr.Error(f"TRELLIS 高斯提取失败：{exc}") from exc

    updated_state = dict(result_state)
    updated_state["gaussian_path"] = payload["gaussian_path"]
    status = "高斯结果提取完成。高斯文件通常较大，预览和下载都可能稍慢。"
    return updated_state, payload["gaussian_path"], gr.update(value=payload["gaussian_path"], interactive=True), status



def reset_slider(default_value: float | int):
    return default_value


with gr.Blocks(title="建筑与城市场景生成平台") as demo:
    gr.Markdown(
        "# 建筑与城市场景生成平台：手绘草图—文本协同创作\n"
        "当前页面已支持：第一步草图生成场景图、第二步局部重绘、第三步多视角图像生成，以及第四色步 TRELLIS 3D 资源生成。"
    )

    generated_images_state = gr.State([])
    selected_image_state = gr.State(None)
    generation_prompt_state = gr.State("")
    generation_seed_state = gr.State(42)
    multiview_source_state = gr.State(None)
    multiview_gallery_state = gr.State([])
    multiview_selected_index_state = gr.State(None)
    trellis_default_input_state = gr.State([])
    trellis_input_state = gr.State([])
    trellis_result_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="上传建筑或城市场景草图")
            input_prompt = gr.Textbox(
                label="场景提示词 (Prompt)",
                lines=6,
                placeholder="输入第一步场景生成的提示词...",
            )
            input_seed = gr.Number(value=42, label="第一步随机种子 (Seed，-1 为随机)", precision=0)
            generate_btn = gr.Button("一键生成多参数场景图", variant="primary")
            page_status = gr.Markdown(
                "可先上传草图并生成候选场景图；如果想单独使用后续步骤，也可以直接点击确认按钮进入第二步后再手动上传图像。"
            )
            confirm_scene_btn = gr.Button("确认当前所选场景图，进入第二步", interactive=True)

        with gr.Column(scale=2):
            output_gallery = gr.Gallery(
                label="第一步：参数遍历生成结果（点击一张后，确认进入第二步）",
                columns=3,
                height="auto",
                object_fit="contain",
            )

    with gr.Column(visible=False) as inpaint_panel:
        gr.Markdown("## 第二步：局部重绘（可选）")
        with gr.Row():
            with gr.Column(scale=1):
                selected_preview = gr.Image(type="pil", label="第一步输出参考图像", interactive=False)
                step2_manual_input = gr.Image(
                    type="pil",
                    label="或手动上传第二步输入图像（上传后优先使用）",
                )
                inpaint_prompt = gr.Textbox(
                    label="局部重绘提示词（可选）",
                    lines=4,
                    placeholder="例如：把被涂抹区域改成现代玻璃幕墙办公楼。留空时默认使用第一步提示词。",
                )
                inpaint_seed = gr.Number(
                    value=42,
                    label="局部重绘种子 (Seed，-1 为随机)",
                    precision=0,
                )
                inpaint_btn = gr.Button("执行局部重绘", variant="primary")
                skip_btn = gr.Button("使用当前输入作为第二步结果（不做局部重绘）")
                confirm_inpaint_btn = gr.Button("确认当前第二步结果，进入第三步")
                inpaint_status = gr.Markdown(
                    "请先在第一步确认一张场景图，或直接在这里手动上传图像；结果满意后，再点击确认按钮进入第三步。"
                )
            with gr.Column(scale=2):
                mask_editor = gr.ImageEditor(
                    type="pil",
                    label="在图上涂抹需要重绘的区域",
                    brush=gr.Brush(colors=["#FFFFFF"], default_size=32),
                    eraser=gr.Eraser(default_size=24),
                    transforms=(),
                    layers=True,
                    image_mode="RGBA",
                    width=768,
                    height=768,
                )
                with gr.Row():
                    generated_mask = gr.Image(type="pil", label="自动生成的 Mask（黑底白色涂抹区）", interactive=False)
                    final_image = gr.Image(type="pil", label="第二步输出图像", interactive=False)

    with gr.Column(visible=False) as multiview_panel:
        gr.Markdown("## 第三步：生成多视角图像")
        with gr.Row():
            with gr.Column(scale=1):
                multiview_source_preview = gr.Image(
                    type="pil",
                    label="第三步默认输入图像（来自第二步）",
                    interactive=False,
                    elem_id="mv-source-preview",
                )
                multiview_manual_input = gr.Image(
                    type="pil",
                    label="或手动上传第三步输入图像（上传后优先使用）",
                )
                multiview_model_type = gr.Radio(
                    choices=list(MULTIVIEW_MODEL_CHOICES.keys()),
                    value="GGUF",
                    label="模型类型",
                )
                multiview_model_file = gr.Dropdown(
                    choices=available_model_files("GGUF"),
                    value=available_model_files("GGUF")[0],
                    label="模型文件",
                )
            with gr.Column(scale=1):
                latest_multiview_preview = gr.Image(
                    type="pil",
                    label="第三步最新生成结果",
                    interactive=False,
                )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 可视化控制球")
                gr.HTML(MULTIANGLE_WIDGET_HTML, elem_id="mv-widget-host")
            with gr.Column(scale=1):
                gr.Markdown("### 参数设置")
                with gr.Row():
                    multiview_azimuth = gr.Slider(
                        minimum=0,
                        maximum=315,
                        value=90,
                        step=1,
                        label="方位角（水平旋转）",
                        elem_id="mv-azimuth",
                    )
                    azimuth_reset = gr.Button("重置", elem_classes=["mv-reset-btn"])
                with gr.Row():
                    multiview_elevation = gr.Slider(
                        minimum=-30,
                        maximum=60,
                        value=0,
                        step=1,
                        label="仰角（垂直角）",
                        elem_id="mv-elevation",
                    )
                    elevation_reset = gr.Button("重置", elem_classes=["mv-reset-btn"])
                with gr.Row():
                    multiview_distance = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=5,
                        step=0.1,
                        label="距离",
                        elem_id="mv-distance",
                    )
                    distance_reset = gr.Button("重置", elem_classes=["mv-reset-btn"])
                with gr.Row():
                    multiview_steps = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=8,
                        step=1,
                        label="步数",
                    )
                    steps_reset = gr.Button("重置", elem_classes=["mv-reset-btn"])
                with gr.Row():
                    multiview_cfg = gr.Slider(
                        minimum=0,
                        maximum=5,
                        value=1,
                        step=0.1,
                        label="CFG",
                    )
                    cfg_reset = gr.Button("重置", elem_classes=["mv-reset-btn"])
                with gr.Row():
                    multiview_seed = gr.Number(
                        value=-1,
                        precision=0,
                        label="种子（-1 为随机）",
                    )
                    seed_reset = gr.Button("重置", elem_classes=["mv-reset-btn"])

        with gr.Row():
            multiview_generate_btn = gr.Button("生成多视角图像", variant="primary")
            multiview_delete_btn = gr.Button("删除所选预览图")
            multiview_clear_btn = gr.Button("清空多视角预览")
            multiview_confirm_btn = gr.Button("确认当前第三步结果，进入第四色步")

        multiview_status = gr.Markdown(multiview_placeholder_status())
        multiview_gallery = gr.Gallery(
            label="多视角输出预览模块（点击图片后可删除）",
            columns=3,
            height="auto",
            object_fit="contain",
        )

    with gr.Column(visible=False) as trellis_panel:
        gr.Markdown("## 第四步：基于 TRELLIS 生成 3D 资源")
        gr.Markdown(
            "第四色步会固定使用第三步的正视角输入图像；如果第三步还生成了其它视角图像，也会一并送入 TRELLIS。"
            "如果第三步没有生成额外视角图像，TRELLIS 也可以只使用这 1 张图像继续生成 3D 资源。"
            "如果你在这里手动上传图像，第四步将完全改用手动上传内容，不再使用第三步默认输入。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                trellis_default_input_gallery = gr.Gallery(
                    label="第四色步默认输入参考（来自第三步，始终展示）",
                    columns=3,
                    height="auto",
                    object_fit="contain",
                )
                trellis_manual_input_files = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                    label="或手动上传第四色步输入图像（上传后将完全替代默认输入）",
                )
                trellis_input_gallery = gr.Gallery(
                    label="第四色步当前实际输入图像",
                    columns=3,
                    height="auto",
                    object_fit="contain",
                )
                with gr.Accordion(label="生成设置", open=False):
                    trellis_seed = gr.Number(
                        value=-1,
                        precision=0,
                        label="种子（-1 为随机）",
                    )
                    gr.Markdown("Stage 1: Sparse Structure Generation")
                    with gr.Row():
                        trellis_ss_guidance_strength = gr.Slider(
                            minimum=0.0,
                            maximum=10.0,
                            value=7.5,
                            step=0.1,
                            label="Guidance Strength",
                        )
                        trellis_ss_sampling_steps = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=12,
                            step=1,
                            label="Sampling Steps",
                        )
                    gr.Markdown("Stage 2: Structured Latent Generation")
                    with gr.Row():
                        trellis_slat_guidance_strength = gr.Slider(
                            minimum=0.0,
                            maximum=10.0,
                            value=3.0,
                            step=0.1,
                            label="Guidance Strength",
                        )
                        trellis_slat_sampling_steps = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=12,
                            step=1,
                            label="Sampling Steps",
                        )
                    trellis_multiimage_algo = gr.Radio(
                        choices=["stochastic", "multidiffusion"],
                        value="stochastic",
                        label="Multi-image Algorithm",
                    )
                trellis_generate_btn = gr.Button("生成3D资源", variant="primary")
                with gr.Accordion(label="GLB提取设置", open=False):
                    trellis_mesh_simplify = gr.Slider(
                        minimum=0.90,
                        maximum=0.98,
                        value=0.95,
                        step=0.01,
                        label="Simplify",
                    )
                    trellis_texture_size = gr.Slider(
                        minimum=512,
                        maximum=2048,
                        value=1024,
                        step=512,
                        label="Texture Size",
                    )
                with gr.Row():
                    trellis_extract_glb_btn = gr.Button("提取GLB", interactive=False)
                    trellis_extract_gaussian_btn = gr.Button("高斯抽取", interactive=False)
                gr.Markdown("*注意：高斯文件通常较大，预览与下载都可能稍慢。*")
            with gr.Column(scale=1):
                trellis_video_output = gr.Video(
                    label="生成的3D资源预览",
                    autoplay=True,
                    loop=True,
                    height=360,
                )
                trellis_model_output = LitModel3D(
                    label="提取的GLB/高斯结果",
                    exposure=10.0,
                    height=360,
                )
                with gr.Row():
                    trellis_download_glb = gr.DownloadButton(label="下载GLB", interactive=False)
                    trellis_download_gaussian = gr.DownloadButton(label="下载高斯", interactive=False)
        trellis_status = gr.Markdown(trellis_placeholder_status())

    trellis_stage_outputs = [
        trellis_panel,
        trellis_default_input_state,
        trellis_default_input_gallery,
        trellis_manual_input_files,
        trellis_input_state,
        trellis_input_gallery,
        trellis_result_state,
        trellis_video_output,
        trellis_model_output,
        trellis_status,
        trellis_extract_glb_btn,
        trellis_extract_gaussian_btn,
        trellis_download_glb,
        trellis_download_gaussian,
    ]

    generate_event = generate_btn.click(
        fn=generate_scene,
        inputs=[input_image, input_prompt, input_seed],
        outputs=[
            output_gallery,
            generated_images_state,
            selected_image_state,
            selected_preview,
            step2_manual_input,
            mask_editor,
            final_image,
            inpaint_panel,
            page_status,
            generation_prompt_state,
            generation_seed_state,
            generated_mask,
            inpaint_status,
            multiview_panel,
            multiview_source_state,
            multiview_source_preview,
            multiview_manual_input,
            latest_multiview_preview,
            multiview_gallery_state,
            multiview_gallery,
            multiview_status,
            multiview_selected_index_state,
            inpaint_seed,
            confirm_scene_btn,
        ],
    )
    generate_event.then(fn=reset_trellis_stage, outputs=trellis_stage_outputs)

    select_scene_event = output_gallery.select(
        fn=select_generated_image,
        inputs=[generated_images_state, generation_seed_state],
        outputs=[
            selected_image_state,
            selected_preview,
            step2_manual_input,
            mask_editor,
            inpaint_panel,
            page_status,
            generated_mask,
            final_image,
            inpaint_status,
            multiview_panel,
            multiview_source_state,
            multiview_source_preview,
            multiview_manual_input,
            latest_multiview_preview,
            multiview_gallery_state,
            multiview_gallery,
            multiview_status,
            multiview_selected_index_state,
            inpaint_seed,
            confirm_scene_btn,
        ],
    )
    select_scene_event.then(fn=reset_trellis_stage, outputs=trellis_stage_outputs)

    confirm_scene_event = confirm_scene_btn.click(
        fn=confirm_scene_selection,
        inputs=[selected_image_state, generation_seed_state],
        outputs=[
            selected_preview,
            step2_manual_input,
            mask_editor,
            final_image,
            inpaint_panel,
            page_status,
            generated_mask,
            inpaint_status,
            multiview_panel,
            multiview_source_state,
            multiview_source_preview,
            multiview_manual_input,
            latest_multiview_preview,
            multiview_gallery_state,
            multiview_gallery,
            multiview_status,
            multiview_selected_index_state,
            inpaint_seed,
        ],
    )
    confirm_scene_event.then(fn=reset_trellis_stage, outputs=trellis_stage_outputs)

    inpaint_event = inpaint_btn.click(
        fn=run_inpainting,
        inputs=[
            mask_editor,
            inpaint_prompt,
            inpaint_seed,
            selected_image_state,
            step2_manual_input,
            generation_prompt_state,
            generation_seed_state,
        ],
        outputs=[
            final_image,
            generated_mask,
            inpaint_status,
            multiview_panel,
            multiview_source_state,
            multiview_source_preview,
            multiview_manual_input,
            latest_multiview_preview,
            multiview_gallery_state,
            multiview_gallery,
            multiview_status,
            multiview_selected_index_state,
        ],
    )
    inpaint_event.then(fn=reset_trellis_stage, outputs=trellis_stage_outputs)

    skip_event = skip_btn.click(
        fn=use_original_image,
        inputs=[selected_image_state, step2_manual_input],
        outputs=[
            final_image,
            generated_mask,
            inpaint_status,
            multiview_panel,
            multiview_source_state,
            multiview_source_preview,
            multiview_manual_input,
            latest_multiview_preview,
            multiview_gallery_state,
            multiview_gallery,
            multiview_status,
            multiview_selected_index_state,
        ],
    )
    skip_event.then(fn=reset_trellis_stage, outputs=trellis_stage_outputs)

    step2_manual_input_event = step2_manual_input.change(
        fn=load_manual_inpaint_input,
        inputs=[step2_manual_input, selected_image_state],
        outputs=[
            mask_editor,
            final_image,
            generated_mask,
            inpaint_status,
            multiview_panel,
            multiview_source_state,
            multiview_source_preview,
            multiview_manual_input,
            latest_multiview_preview,
            multiview_gallery_state,
            multiview_gallery,
            multiview_status,
            multiview_selected_index_state,
        ],
    )
    step2_manual_input_event.then(fn=reset_trellis_stage, outputs=trellis_stage_outputs)

    confirm_inpaint_event = confirm_inpaint_btn.click(
        fn=confirm_inpaint_result,
        inputs=[final_image],
        outputs=[
            multiview_panel,
            multiview_source_state,
            multiview_source_preview,
            multiview_manual_input,
            latest_multiview_preview,
            multiview_gallery_state,
            multiview_gallery,
            multiview_status,
            multiview_selected_index_state,
        ],
    )
    confirm_inpaint_event.then(fn=reset_trellis_stage, outputs=trellis_stage_outputs)

    multiview_model_type.change(
        fn=update_multiview_model_file,
        inputs=[multiview_model_type],
        outputs=[multiview_model_file],
    )

    azimuth_reset.click(fn=lambda: reset_slider(90), outputs=[multiview_azimuth])
    elevation_reset.click(fn=lambda: reset_slider(0), outputs=[multiview_elevation])
    distance_reset.click(fn=lambda: reset_slider(5), outputs=[multiview_distance])
    steps_reset.click(fn=lambda: reset_slider(8), outputs=[multiview_steps])
    cfg_reset.click(fn=lambda: reset_slider(1), outputs=[multiview_cfg])
    seed_reset.click(fn=lambda: -1, outputs=[multiview_seed])

    multiview_generate_event = multiview_generate_btn.click(
        fn=generate_multiview_image,
        inputs=[
            multiview_source_state,
            multiview_manual_input,
            multiview_model_type,
            multiview_model_file,
            multiview_azimuth,
            multiview_elevation,
            multiview_distance,
            multiview_steps,
            multiview_cfg,
            multiview_seed,
            multiview_gallery_state,
        ],
        outputs=[
            latest_multiview_preview,
            multiview_gallery_state,
            multiview_gallery,
            multiview_status,
            multiview_selected_index_state,
        ],
    )
    multiview_generate_event.then(fn=reset_trellis_stage, outputs=trellis_stage_outputs)

    multiview_manual_input_event = multiview_manual_input.change(
        fn=load_manual_multiview_input,
        inputs=[multiview_manual_input, multiview_source_state],
        outputs=[
            latest_multiview_preview,
            multiview_gallery_state,
            multiview_gallery,
            multiview_status,
            multiview_selected_index_state,
        ],
    )
    multiview_manual_input_event.then(fn=reset_trellis_stage, outputs=trellis_stage_outputs)

    multiview_gallery.select(
        fn=select_multiview_image,
        inputs=[multiview_gallery_state],
        outputs=[
            multiview_selected_index_state,
            latest_multiview_preview,
            multiview_status,
        ],
    )

    multiview_delete_event = multiview_delete_btn.click(
        fn=delete_multiview_image,
        inputs=[multiview_gallery_state, multiview_selected_index_state],
        outputs=[
            multiview_gallery_state,
            multiview_gallery,
            latest_multiview_preview,
            multiview_status,
            multiview_selected_index_state,
        ],
    )
    multiview_delete_event.then(fn=reset_trellis_stage, outputs=trellis_stage_outputs)

    multiview_clear_event = multiview_clear_btn.click(
        fn=clear_multiview_gallery,
        outputs=[
            multiview_gallery_state,
            multiview_gallery,
            latest_multiview_preview,
            multiview_status,
            multiview_selected_index_state,
        ],
    )
    multiview_clear_event.then(fn=reset_trellis_stage, outputs=trellis_stage_outputs)

    multiview_confirm_btn.click(
        fn=confirm_multiview_result,
        inputs=[multiview_source_state, multiview_manual_input, multiview_gallery_state],
        outputs=trellis_stage_outputs,
    )

    trellis_manual_input_files.change(
        fn=update_trellis_manual_inputs,
        inputs=[trellis_manual_input_files, trellis_default_input_state],
        outputs=[
            trellis_input_state,
            trellis_input_gallery,
            trellis_result_state,
            trellis_video_output,
            trellis_model_output,
            trellis_status,
            trellis_extract_glb_btn,
            trellis_extract_gaussian_btn,
            trellis_download_glb,
            trellis_download_gaussian,
        ],
    )

    trellis_generate_btn.click(
        fn=generate_trellis_asset,
        inputs=[
            trellis_input_state,
            trellis_seed,
            trellis_ss_guidance_strength,
            trellis_ss_sampling_steps,
            trellis_slat_guidance_strength,
            trellis_slat_sampling_steps,
            trellis_multiimage_algo,
        ],
        outputs=[
            trellis_result_state,
            trellis_video_output,
            trellis_model_output,
            trellis_status,
            trellis_extract_glb_btn,
            trellis_extract_gaussian_btn,
            trellis_download_glb,
            trellis_download_gaussian,
        ],
    )

    trellis_extract_glb_btn.click(
        fn=extract_trellis_glb,
        inputs=[trellis_result_state, trellis_mesh_simplify, trellis_texture_size],
        outputs=[
            trellis_result_state,
            trellis_model_output,
            trellis_download_glb,
            trellis_status,
        ],
    )

    trellis_extract_gaussian_btn.click(
        fn=extract_trellis_gaussian,
        inputs=[trellis_result_state],
        outputs=[
            trellis_result_state,
            trellis_model_output,
            trellis_download_gaussian,
            trellis_status,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=6006,
        css=APP_CSS,
        head=MULTIANGLE_WIDGET_HEAD,
        # Generated previews and exports may live outside the project directory.
        allowed_paths=[str(TRELLIS_RUNTIME_ROOT.resolve())],
    )
