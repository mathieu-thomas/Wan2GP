"""Wan2GP plugin providing Ultralytics YOLO11 inference tools."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm
from ultralytics import ASSETS, YOLO

try:
    import torch
except Exception:  # pragma: no cover - torch should normally be present via Ultralytics
    torch = None  # type: ignore[assignment]

try:
    import torchvision  # type: ignore
except Exception:  # pragma: no cover - torchvision ships with Ultralytics but be defensive
    torchvision = None  # type: ignore[assignment]

from shared.utils.plugins import WAN2GPPlugin


console = Console()


MODEL_CHOICES: List[Tuple[str, str]] = [
    ("YOLO11n (detect)", "yolo11n.pt"),
    ("YOLO11s (detect)", "yolo11s.pt"),
    ("YOLO11n (seg)", "yolo11n-seg.pt"),
    ("YOLO11s (seg)", "yolo11s-seg.pt"),
    ("YOLO11n (pose)", "yolo11n-pose.pt"),
    ("YOLO11s (pose)", "yolo11s-pose.pt"),
]

WEBCAM_TARGET_FPS: int = 15
PRE_YOLO_MAX_SIDE: int = 640
_FFMPEG_ENV = "WAN2GP_YOLO11_FFMPEG_PATH"
_FFPROBE_ENV = "WAN2GP_YOLO11_FFPROBE_PATH"


def _cuda_nms_supported() -> bool:
    """Return True when torchvision's CUDA NMS kernel is callable."""

    if torch is None or torchvision is None:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device="cuda")
        scores = torch.tensor([0.5], device="cuda")
        torchvision.ops.nms(boxes, scores, 0.5)
        return True
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        console.log(
            "[yellow]torchvision CUDA NMS unavailable; GPU device options disabled."
        )
        console.log(str(exc))
        return False


_CUDA_NMS_AVAILABLE = _cuda_nms_supported()


def _device_choices() -> List[str]:
    choices = ["auto", "cpu"]
    if torch is not None and _CUDA_NMS_AVAILABLE:
        try:
            device_count = torch.cuda.device_count()
        except Exception:  # pragma: no cover - defensive
            device_count = 0
        choices.extend(str(idx) for idx in range(device_count))
    return choices


DEVICE_CHOICES: List[str] = _device_choices()
DEFAULT_DEVICE = "auto" if _CUDA_NMS_AVAILABLE else "cpu"


_MODEL_CACHE: Dict[str, YOLO] = {}


def get_model(model_name: str) -> YOLO:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = YOLO(model_name)
    return _MODEL_CACHE[model_name]


def _check_exe(path: Optional[str], name: str) -> str:
    if path:
        candidate = Path(path)
        if candidate.exists():
            return str(candidate)
    alt = shutil.which(name)
    if alt:
        return alt
    raise FileNotFoundError(
        f"Executable not found: {name}. Checked '{path}' and PATH."
    )


def _ffprobe_stream_info(video_path: str) -> Dict:
    ffprobe = _check_exe(os.getenv(_FFPROBE_ENV), "ffprobe")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        video_path,
    ]
    out = subprocess.check_output(cmd)
    data = json.loads(out.decode("utf-8", errors="ignore"))
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("FFprobe: no video stream found.")
    return streams[0]


def _parse_fps(stream: Dict) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        val = stream.get(key, "0/0")
        if isinstance(val, str) and "/" in val:
            num, den = val.split("/", 1)
            try:
                num = float(num)
                den = float(den)
                if num > 0 and den > 0:
                    return num / den
            except Exception:
                pass
    duration = float(stream.get("duration", 0.0) or 0.0)
    nb_frames = stream.get("nb_frames")
    if nb_frames is not None:
        try:
            nb_frames = int(nb_frames)
            if duration > 0:
                return nb_frames / duration
        except Exception:
            pass
    return 25.0


def _parse_wh(stream: Dict) -> Tuple[int, int]:
    width = int(stream.get("width", 0) or 0)
    height = int(stream.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid video dimensions.")
    return width, height


def _estimate_total_frames(stream: Dict, fps: float) -> Optional[int]:
    nb_frames = stream.get("nb_frames")
    if nb_frames is not None:
        try:
            return int(nb_frames)
        except Exception:
            pass
    duration = float(stream.get("duration", 0.0) or 0.0)
    if duration > 0 and fps > 0:
        return int(duration * fps + 0.5)
    return None


def extract_detections(result) -> List[str]:
    names = result.names
    detections: List[str] = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None and boxes.cls is not None:
        cls = boxes.cls.detach().cpu().numpy().astype(int).tolist()
        conf = boxes.conf.detach().cpu().numpy().tolist()
        for c, p in zip(cls, conf):
            label = names.get(c, str(c))
            detections.append(f"{label}: {p*100:.1f}%")
    return detections


def rich_table_frame(frame_idx: int, det_ms: float, fps_inst: float, detections: List[str]) -> Table:
    table = Table(title=f"Frame {frame_idx}", box=box.SIMPLE_HEAVY, show_header=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="bold")
    table.add_row("Detection time", f"{det_ms:.1f} ms")
    table.add_row("FPS (instant)", f"{fps_inst:.2f}")
    if detections:
        truncated = ", ".join(detections[:8])
        if len(detections) > 8:
            truncated += f" …(+{len(detections) - 8})"
    else:
        truncated = "—"
    table.add_row("Objects", truncated)
    return table


def resize_keep_ar_rgb(np_rgb: np.ndarray, max_side: int) -> np.ndarray:
    height, width = np_rgb.shape[:2]
    if max(height, width) <= max_side:
        return np_rgb
    scale = max_side / float(max(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    pil_image = Image.fromarray(np_rgb)
    pil_image = pil_image.resize((new_width, new_height), resample=Image.BICUBIC)
    return np.array(pil_image, dtype=np.uint8)


def resize_pil_keep_ar(pil_img: Image.Image, max_side: int) -> Image.Image:
    width, height = pil_img.size
    if max(width, height) <= max_side:
        return pil_img
    scale = max_side / float(max(width, height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return pil_img.resize((new_width, new_height), resample=Image.BICUBIC)


def _build_predict_kwargs(
    conf_threshold: float,
    iou_threshold: float,
    imgsz: int,
    device: str,
    fp16: bool,
) -> Tuple[Dict, str]:
    requested_device = device or "auto"
    resolved_device = requested_device
    if requested_device.lower() == "auto" and not _CUDA_NMS_AVAILABLE:
        resolved_device = "cpu"
    kwargs: Dict = dict(
        conf=float(conf_threshold),
        iou=float(iou_threshold),
        imgsz=int(imgsz),
        verbose=False,
        show_labels=True,
        show_conf=True,
        half=bool(fp16) and resolved_device not in {"cpu"},
    )
    if resolved_device.lower() not in {"", "auto", "cpu"}:
        kwargs["device"] = resolved_device
    elif resolved_device.lower() == "cpu":
        kwargs["device"] = "cpu"
    return kwargs, resolved_device


def _predict_with_nms_fallback(
    model: YOLO,
    source,
    predict_kwargs: Dict,
    requested_device: str,
) -> Tuple[List, Optional[str]]:
    try:
        return model.predict(source=source, **predict_kwargs), None
    except RuntimeError as exc:
        message = str(exc)
        if (
            requested_device
            and requested_device.lower() not in {"", "auto", "cpu"}
            and "torchvision::nms" in message
            and "CUDA" in message
        ):
            console.print(
                "[yellow]Falling back to CPU because torchvision CUDA NMS is unavailable."
            )
            fallback_kwargs = dict(predict_kwargs)
            fallback_kwargs["device"] = "cpu"
            fallback_kwargs["half"] = False
            return model.predict(source=source, **fallback_kwargs), "cpu"
        raise


def predict_image(
    img: Image.Image,
    conf_threshold: float,
    iou_threshold: float,
    model_name: str,
    imgsz: int,
    device: str,
    fp16: bool,
) -> Image.Image:
    try:
        if img is None:
            raise gr.Error("No image supplied.")
        img_small = resize_pil_keep_ar(img, PRE_YOLO_MAX_SIDE)
        model = get_model(model_name)
        kwargs, resolved_device = _build_predict_kwargs(
            conf_threshold, iou_threshold, imgsz, device, fp16
        )
        start = time.perf_counter()
        results, fallback_device = _predict_with_nms_fallback(
            model, img_small, kwargs, resolved_device
        )
        duration = (time.perf_counter() - start) * 1000.0
        fps_inst = 1000.0 / duration if duration > 0 else 0.0
        if not results:
            raise RuntimeError("No result returned by the model.")
        result = results[0]
        annotated_small = result.plot()
        annotated_full = Image.fromarray(annotated_small[..., ::-1]).resize(
            img.size, resample=Image.BICUBIC
        )
        detections = extract_detections(result)
        console.print(
            rich_table_frame(frame_idx=0, det_ms=duration, fps_inst=fps_inst, detections=detections)
        )
        if fallback_device:
            console.print("[yellow]Image inference continued on CPU.")
        return annotated_full
    except Exception as exc:
        raise gr.Error(f"Image inference error: {exc}") from exc


def predict_video(
    video_path: str,
    conf_threshold: float,
    iou_threshold: float,
    model_name: str,
    imgsz: int,
    device: str,
    fp16: bool,
) -> str:
    workdir = Path(tempfile.mkdtemp(prefix="yolo11_ui_"))
    start_time = time.time()
    try:
        src = Path(video_path)
        if not src.exists():
            raise gr.Error("Video file not found.")
        stream = _ffprobe_stream_info(str(src))
        width, height = _parse_wh(stream)
        fps_container = _parse_fps(stream)
        total_frames = _estimate_total_frames(stream, fps_container)
        out_mp4 = workdir / f"{src.stem}_yolo11_{uuid.uuid4().hex[:8]}.mp4"
        model = get_model(model_name)
        ffmpeg = _check_exe(os.getenv(_FFMPEG_ENV), "ffmpeg")
        decode_cmd = [
            ffmpeg,
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-vsync",
            "0",
            "pipe:1",
        ]
        decode_proc = subprocess.Popen(
            decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        encode_cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps_container:.6f}",
            "-i",
            "pipe:0",
            "-i",
            str(src),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-preset",
            "faster",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ]
        encode_proc = subprocess.Popen(
            encode_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        frame_size = width * height * 3
        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        progress = tqdm(
            total=total_frames if total_frames else None,
            unit="frame",
            desc="YOLO11 video",
            dynamic_ncols=True,
            smoothing=0.1,
            miniters=1,
            bar_format=bar_format,
        )
        frames_done = 0
        effective_device = device
        effective_fp16 = fp16
        while True:
            raw = decode_proc.stdout.read(frame_size)
            if not raw or len(raw) < frame_size:
                break
            frame_bgr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            frame_rgb = frame_bgr[..., ::-1]
            frame_rgb_small = resize_keep_ar_rgb(frame_rgb, PRE_YOLO_MAX_SIDE)
            predict_kwargs, resolved_device = _build_predict_kwargs(
                conf_threshold,
                iou_threshold,
                imgsz,
                effective_device,
                effective_fp16,
            )
            start_pred = time.perf_counter()
            results, fallback_device = _predict_with_nms_fallback(
                model, frame_rgb_small, predict_kwargs, resolved_device
            )
            det_ms = (time.perf_counter() - start_pred) * 1000.0
            fps_inst = 1000.0 / det_ms if det_ms > 0 else 0.0
            result = results[0]
            annotated_small = result.plot()
            annotated_full = Image.fromarray(annotated_small[..., ::-1]).resize(
                (width, height), resample=Image.BICUBIC
            )
            annotated_bgr = np.array(annotated_full, dtype=np.uint8)[..., ::-1]
            encode_proc.stdin.write(annotated_bgr.tobytes())
            detections = extract_detections(result)
            console.print(
                rich_table_frame(
                    frame_idx=frames_done,
                    det_ms=det_ms,
                    fps_inst=fps_inst,
                    detections=detections,
                )
            )
            if fallback_device:
                effective_device = fallback_device
                effective_fp16 = False
            frames_done += 1
            progress.update(1)
            elapsed = time.time() - start_time
            fps_eff = frames_done / elapsed if elapsed > 0 else 0.0
            if total_frames and fps_eff > 0:
                remaining = max(0, total_frames - frames_done) / fps_eff
                hrs, rem = divmod(remaining, 3600)
                mins, secs = divmod(rem, 60)
                progress.set_postfix_str(
                    f"ETA ~ {int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"
                )
        progress.close()
        if decode_proc.stdout:
            decode_proc.stdout.close()
        if encode_proc.stdin:
            encode_proc.stdin.flush()
            encode_proc.stdin.close()
        decode_proc.wait()
        encode_proc.wait()
        if decode_proc.stderr:
            dec_err = decode_proc.stderr.read().decode("utf-8", errors="ignore")
            if dec_err.strip():
                console.log(f"[ffmpeg decoder] {dec_err.strip()}")
        if encode_proc.stderr:
            enc_err = encode_proc.stderr.read().decode("utf-8", errors="ignore")
            if enc_err.strip():
                console.log(f"[ffmpeg encoder] {enc_err.strip()}")
        if encode_proc.returncode != 0:
            raise gr.Error("ffmpeg encoding failed. Check logs for details.")
        total_elapsed = time.time() - start_time
        avg_fps = frames_done / total_elapsed if total_elapsed > 0 else 0.0
        console.rule("[bold green]Video processing complete")
        console.print(
            Panel.fit(
                "\n".join(
                    [
                        f"File: {out_mp4}",
                        f"Frames: {frames_done}" + (f"/{total_frames}" if total_frames else ""),
                        f"Duration: {total_elapsed:.1f}s | FPS avg: {avg_fps:.2f}",
                    ]
                ),
                border_style="green",
            )
        )
        return str(out_mp4)
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Video inference error: {exc}") from exc


_WEBCAM_FRAME_IDX = 0


def predict_frame_from_webcam(
    frame: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    model_name: str,
    imgsz: int,
    device: str,
    fp16: bool,
) -> np.ndarray:
    global _WEBCAM_FRAME_IDX
    if frame is None:
        raise gr.Error("No frame received from webcam.")
    height, width = frame.shape[:2]
    frame_small = resize_keep_ar_rgb(frame, PRE_YOLO_MAX_SIDE)
    model = get_model(model_name)
    predict_kwargs, resolved_device = _build_predict_kwargs(
        conf_threshold,
        iou_threshold,
        imgsz,
        device,
        fp16,
    )
    start = time.perf_counter()
    results, fallback_device = _predict_with_nms_fallback(
        model, frame_small, predict_kwargs, resolved_device
    )
    det_ms = (time.perf_counter() - start) * 1000.0
    fps_inst = 1000.0 / det_ms if det_ms > 0 else 0.0
    result = results[0]
    annotated_small = result.plot()
    annotated_full = Image.fromarray(annotated_small[..., ::-1]).resize(
        (width, height), resample=Image.BICUBIC
    )
    annotated_rgb = np.array(annotated_full, dtype=np.uint8)
    detections = extract_detections(result)
    console.print(
        rich_table_frame(
            frame_idx=_WEBCAM_FRAME_IDX,
            det_ms=det_ms,
            fps_inst=fps_inst,
            detections=detections,
        )
    )
    if fallback_device:
        console.print("[yellow]Webcam inference continued on CPU.")
    _WEBCAM_FRAME_IDX += 1
    return annotated_rgb


def _label_to_pt(label: str) -> str:
    for friendly, weight in MODEL_CHOICES:
        if friendly == label:
            return weight
    return MODEL_CHOICES[0][1]


class UltralyticsYOLOPlugin(WAN2GPPlugin):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Ultralytics YOLO11"
        self.version = "1.1.0"
        self.description = "Image, video, and webcam inference using Ultralytics YOLO11."

    def setup_ui(self) -> None:
        self.add_tab(
            tab_id="wan2gp-yolo11",
            label="Ultralytics YOLO11",
            component_constructor=self._create_ui,
        )

    def _create_ui(self):
        with gr.Column():
            gr.Markdown("# Ultralytics YOLO11 — Image, Video & Webcam 🚀")
            gr.Markdown(
                "- Adjust detection confidence, IoU, and imgsz for best results.\n"
                "- **PRE_YOLO_MAX_SIDE** applies a pre-resize before running YOLO to speed up inference.\n"
                "- **WEBCAM_TARGET_FPS** controls the perceived latency for webcam streaming."
            )
            with gr.Tabs():
                with gr.Tab("Image"):
                    with gr.Row():
                        in_img = gr.Image(type="pil", label="Input image")
                        out_img = gr.Image(type="pil", label="Annotated output")
                    with gr.Row():
                        model_dd_img = gr.Dropdown(
                            [label for label, _ in MODEL_CHOICES],
                            value=MODEL_CHOICES[0][0],
                            label="Model",
                        )
                        conf_img = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Confidence")
                        iou_img = gr.Slider(0.0, 1.0, value=0.45, step=0.01, label="IoU")
                        imgsz_img = gr.Slider(320, 1280, value=640, step=32, label="imgsz")
                        device_img = gr.Dropdown(
                            DEVICE_CHOICES,
                            value=DEFAULT_DEVICE,
                            label="Device",
                        )
                        fp16_img = gr.Checkbox(value=False, label="fp16")

                    def _img_bridge(img, conf, iou, model_label, imgsz, device, fp16):
                        return predict_image(
                            img,
                            conf,
                            iou,
                            _label_to_pt(model_label),
                            imgsz,
                            device,
                            fp16,
                        )

                    gr.Button("Run image inference", variant="primary").click(
                        _img_bridge,
                        inputs=[
                            in_img,
                            conf_img,
                            iou_img,
                            model_dd_img,
                            imgsz_img,
                            device_img,
                            fp16_img,
                        ],
                        outputs=out_img,
                    )

                    gr.Examples(
                        examples=[
                            [ASSETS / "bus.jpg", 0.25, 0.45, MODEL_CHOICES[0][0], 640, "auto", False],
                            [ASSETS / "zidane.jpg", 0.25, 0.45, MODEL_CHOICES[0][0], 640, "auto", False],
                        ],
                        inputs=[
                            in_img,
                            conf_img,
                            iou_img,
                            model_dd_img,
                            imgsz_img,
                            device_img,
                            fp16_img,
                        ],
                        label="Examples",
                    )

                with gr.Tab("Video (file)"):
                    in_video = gr.Video(label="Source video")
                    out_video = gr.Video(label="Annotated video (audio preserved)")
                    with gr.Row():
                        model_dd_vid = gr.Dropdown(
                            [label for label, _ in MODEL_CHOICES],
                            value=MODEL_CHOICES[0][0],
                            label="Model",
                        )
                        conf_vid = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Confidence")
                        iou_vid = gr.Slider(0.0, 1.0, value=0.45, step=0.01, label="IoU")
                        imgsz_vid = gr.Slider(320, 1280, value=640, step=32, label="imgsz")
                        device_vid = gr.Dropdown(
                            DEVICE_CHOICES,
                            value=DEFAULT_DEVICE,
                            label="Device",
                        )
                        fp16_vid = gr.Checkbox(value=False, label="fp16")

                    def _vid_bridge(vid_path, conf, iou, model_label, imgsz, device, fp16):
                        if not vid_path:
                            raise gr.Error("No video supplied.")
                        return predict_video(
                            vid_path,
                            conf,
                            iou,
                            _label_to_pt(model_label),
                            imgsz,
                            device,
                            fp16,
                        )

                    gr.Button("Process full video", variant="primary").click(
                        _vid_bridge,
                        inputs=[
                            in_video,
                            conf_vid,
                            iou_vid,
                            model_dd_vid,
                            imgsz_vid,
                            device_vid,
                            fp16_vid,
                        ],
                        outputs=out_video,
                    )

                with gr.Tab("Webcam (live)"):
                    with gr.Row():
                        model_dd_cam = gr.Dropdown(
                            [label for label, _ in MODEL_CHOICES],
                            value=MODEL_CHOICES[0][0],
                            label="Model",
                        )
                        conf_cam = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Confidence")
                        iou_cam = gr.Slider(0.0, 1.0, value=0.45, step=0.01, label="IoU")
                        imgsz_cam = gr.Slider(320, 1280, value=640, step=32, label="imgsz")
                        device_cam = gr.Dropdown(
                            DEVICE_CHOICES,
                            value=DEFAULT_DEVICE,
                            label="Device",
                        )
                        fp16_cam = gr.Checkbox(value=False, label="fp16")

                    cam_in = gr.Image(sources=["webcam"], type="numpy", label="Webcam input")
                    cam_out = gr.Image(streaming=True, label="Annotated stream")

                    def _cam_bridge(frame, model_label, conf, iou, imgsz, device, fp16):
                        return predict_frame_from_webcam(
                            frame,
                            conf,
                            iou,
                            _label_to_pt(model_label),
                            imgsz,
                            device,
                            fp16,
                        )

                    stream_interval = max(0.01, 1.0 / float(max(1, WEBCAM_TARGET_FPS)))
                    cam_in.stream(
                        _cam_bridge,
                        inputs=[
                            cam_in,
                            model_dd_cam,
                            conf_cam,
                            iou_cam,
                            imgsz_cam,
                            device_cam,
                            fp16_cam,
                        ],
                        outputs=[cam_out],
                        time_limit=30,
                        stream_every=stream_interval,
                        concurrency_limit=15,
                    )

            gr.Markdown(
                "Quick presets:\n"
                "- **GPU demo**: model `YOLO11n (detect)`, `device='0'`, `fp16=True`, `imgsz=640`, `PRE_YOLO_MAX_SIDE=640`, `WEBCAM_TARGET_FPS=30`.\n"
                "- **CPU**: model `YOLO11n (detect)`, `imgsz=512..640`, `PRE_YOLO_MAX_SIDE=512`, `WEBCAM_TARGET_FPS=10..15`."
            )


__all__ = ["UltralyticsYOLOPlugin"]
