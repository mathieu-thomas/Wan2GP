# Wan 2.2 Animate — Review (Rewritten)

# Wan 2.2 Animate Pipeline

## High-Level Workflow

The `Wan 2.2 Animate` pipeline is designed for advanced video manipulation, allowing users to transfer body and facial motion from a source video to a character image. This can be used to either replace a person in an existing video or animate a still character image using a pose video.

The general workflow is as follows:

1. **Input**: The user provides a source video (for motion) and a character image (the target for the animation).
2. **Masking**: The `Mat Anyone` tool is used to create a precise mask of the person or object in the source video from which the motion will be extracted. This mask isolates the desired motion.
3. **Generation**: The `animate` architecture takes the source video, the character image, and the generated mask as inputs. It then generates a new video where the character from the image is animated with the motion from the source video.
4. **Relighting (Optional)**: A "Relighting" LoRA can be applied during the generation process to adjust the lighting on the animated character, helping it blend more seamlessly into the new scene.
5. **Output**: The final output is a video where the character has been animated or replaced according to the user's inputs.

## Pipeline Components

The `Wan 2.2 Animate` pipeline is composed of several key components that work together to generate the final video.

### Masking: Mat Anyone

The first step in the pipeline is to generate a mask for the source video. This is handled by the **Mat Anyone** tool, which is an interactive masking application integrated into `WanGP`.

- **Implementation**: The core logic for the Mat Anyone tool is located in `preprocessing/matanyone/app.py`. This script provides a Gradio interface for users to load a video and interactively "paint" a mask onto the frames.
- **Functionality**: The tool uses a SAM (Segment Anything Model) to generate precise masks based on user clicks. Users can add positive and negative points to refine the mask until it accurately isolates the desired person or object.
- **Output**: The output of this stage is a black-and-white video mask that is used in the generation stage to specify the area of motion to be transferred.

## Overview

**Wan 2.2 Animate** is the motion-transfer model in the WanGP suite. It maps full-body + facial motion from a driver clip to a target (subject replacement) or animates a still character from a pose video, while preserving the original audio by default. It supports outpainting and is compatible with Wan 2.2 image-to-video LoRA accelerators (e.g., FusioniX).

## Model Package

The default WanGP bundle includes several **14B** weight variants plus an optional **relighting LoRA** for cross-shot consistency. Use **BF16** for peak fidelity; switch to **INT8 hybrids** when VRAM is the bottleneck, then recover coherence with the relighting LoRA.

## Known Behaviors

* **Emotion “normalization”.** Animate tends to normalize expressive cues because of its human-pose priors. Compared to the source, facial emotion can look flatter or less intense.
* **Multi-character confusion.** In busy scenes, identities can bleed or swap if subjects aren’t isolated.
* **Shallow denoise by default.** Limited denoise on the subject can hurt tracking and micro-expression retention.

## Recommended Workflow

1. **Mask → then Animate.** Start with a clean, binary mask of the driver subject (white subject / black background), then refine with a matting pass before running Animate.
2. **One entity per pass.** Animate **one character or one piece of décor per run**. Avoid mixing characters in the same pass to prevent identity bleed.
3. **Plan sliding windows.** Treat each shot as its own window; for long edits, pre-viz end frames with Wan 2.2 i2i and lock prompts per shot.
4. **Pick weights pragmatically.** Begin with **BF16**; drop to **INT8** if memory is tight and add the relighting LoRA when lighting varies across captures.

## Strengths

* **Convincing motion transfer** for body and face, solid for talking-head or performance-driven edits.
* **Outpainting support** to expand the frame during replacement.
* **Audio preserved** by default, reducing dialogue post.

## Limitations

* **Mask-sensitive.** Noisy masks or messy edges leak into the composite.
* **Emotion compression.** Pose normalization mutes expression versus the original driver.
* **Identity mix-ups.** Multiple visible characters without isolation raise confusion risk.
* **Denoise depth.** Insufficient (or unfocused) denoise can weaken tracking on the main subject.

## Tips for Best Results

* **Isolate aggressively.** Build a binary mask for **one** subject or **one** décor element per pass; run multiple passes for multi-character scenes.
* **Prefer “segmentation → binary mask → matting refine”.** Use SAM or Sapiens to get the mask, then refine with a matting tool before Animate.
* **Boost subject denoise.** Increase denoise **steps/strength on the masked subject region** (not the whole frame) to improve tracking, crispness, and subtle mouth/eye motion.
* **Prompt for emotion.** Add explicit emotional descriptors; re-extract masks to ensure lips/eyes are fully included.
* **Window length.** Keep **≥81-frame** windows to maintain style continuity across transitions.
* **Throughput.** Pair with FusioniX (or other Wan 2.2 accelerators) to speed iteration without sacrificing detail.

---

## Segmentation & Matting Toolkit

### Sapiens-Pytorch-Inference (ibaiGorordo)

* **What it is.** Minimal PyTorch wrapper for **Sapiens** (Meta): 2D pose, human part segmentation, depth, normals; examples for image/video/webcam.
* **Highlights.** Auto-download of weights from Hugging Face; multi-task predictor; CLI examples; ONNX export available (typically slower).
* **Usage notes.** 1B models yield better segmentation; inputs around **768×1024** work well; avoid tight person crops; ONNX can be slow.
* **Install.** `pip install sapiens-inferece` (package typo on PyPI) or clone and `pip install -r requirements.txt`.
* **Model context.** Sapiens is trained on ~**300M** in-the-wild images, native **1024×1024** inference, covering pose/segmentation/depth/normals.  
  **Ref:** [GitHub – Sapiens-Pytorch-Inference][1], [Meta Sapiens][2]

### Segment Anything (facebookresearch/segment-anything)

* **What it is.** **SAM** is a promptable segmenter (points/boxes) that returns high-quality masks; can also propose masks for the full image.
* **Data & performance.** Trained on **SA-1B** (~**11M** images, **1.1B** masks); strong zero-shot behavior.
* **Quickstart.** `pip install git+https://github.com/facebookresearch/segment-anything.git`, download a checkpoint, then use `SamPredictor` or `SamAutomaticMaskGenerator`. ONNX export supported.
* **Status.** SAM 2 (images + videos, streaming memory) lives in a separate repo; this one remains the SAM “v1” base.  
  **Ref:** [GitHub – Segment Anything][3]

---

## Verdict

With disciplined **mask isolation**, **one-entity-per-pass** runs, and **subject-focused denoise**, Wan 2.2 Animate delivers reliable, production-ready motion transfer inside the WanGP stack. Expect some **emotion flattening** out of the box; counter it with tighter masks, stronger subject denoise, and explicit emotional prompts. For long edits, sliding-window planning and the relighting LoRA keep results coherent.

[1]: https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference
[2]: https://github.com/facebookresearch/sapiens
[3]: https://github.com/facebookresearch/segment-anything
