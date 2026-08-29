# AGENTS.md

Welcome to **`comic-ocr`**. This document is designed to give AI coding agents and human developers immediate architectural context, code bookmarks, environment instructions, and operational guidelines.

---

## 1. Project Overview & Mission

`comic-ocr` is a Python library for Optical Character Recognition (OCR) fine-tuned for comics and manga.
It follows a **two-stage decoupled pipeline**:
1. **Localization (Detection & Segmentation)**: Semantic segmentation (character/line masks via U-Net/FPN) followed by OpenCV heuristics to group lines and paragraphs.
2. **Recognition (Text Transcription)**: Sequence transcription of cropped line images into text via CRNN (or TrOCR / FCN) with CTC decoding.

---

## 2. Architecture & Code Bookmarks

### Key Entry Points & Public API
* [`comic_ocr/__init__.py`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/__init__.py): High-level public functions (`read_paragraphs`, `read_lines`, `localize_paragraphs`, `localize_lines`, `get_default_model`).
* [`comic_ocr/model.py`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/model.py): `ComicOCRModel` orchestrator combining localization + recognition models; `ModelAccuracy` evaluation metrics.
* [`comic_ocr/hub.py`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/hub.py): Downloading and caching pretrained model weights via PyTorch Hub.

### Geometric Types & Data Structures
* [`comic_ocr/types/shapes.py`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/types/shapes.py): `Rectangle`, `Point`, `Size` primitives (immutable `tuple` subclasses with IoU, union, intersection, and containment logic).
* [`comic_ocr/types/lines.py`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/types/lines.py): `Line` (`(text, Rectangle)`) and `Paragraph` (`(lines, Rectangle)`).
* [`comic_ocr/types/images.py`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/types/images.py): RGB image conversion and type wrappers.

### Text Localization (`comic_ocr/models/localization/`)
* [`localization_model.py`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/models/localization/localization_model.py): Base `LocalizationModel` (`nn.Module`), `WeightedBCEWithLogitsLoss`, and `WeightedDiceLoss`.
* [`conv_unet/conv_unet.py`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/models/localization/conv_unet/conv_unet.py): U-Net implementations (`BaselineConvUnet`, `DeepConvUnet`, `DilatedConvUnet`, `ResNetUnet`).
* [`conv_unet/conv_fpn.py`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/models/localization/conv_unet/conv_fpn.py): Feature Pyramid Network (`ConvFPN`).
* [`localization_open_cv.py`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/models/localization/localization_open_cv.py): Connected components analysis, line grouping (`align_line_horizontal`), and paragraph grouping.
* [`localization_dataset.py`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/models/localization/localization_dataset.py): `LocalizationDataset` and `LocalizationDatasetWithAugmentation` (random cropping, padding, color jitter).
* [`train.py`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/models/localization/train.py): Localization model training loop with TensorBoard logging.

### Text Recognition (`comic_ocr/models/recognition/`)
* [`recognition_model.py`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/models/recognition/recognition_model.py): Base `RecognitionModel`, `CharBaseRecognitionModel`, and CTC loss calculation.
* [`crnn/crnn.py`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/models/recognition/crnn/crnn.py): CRNN (CNN + Bidirectional LSTM + CTC).
* [`trocr/trocr.py`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/models/recognition/trocr/trocr.py): Vision Transformer (TrOCR) encoder-decoder wrapper.
* [`fcn/fcn.py`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/models/recognition/fcn/fcn.py): Fully Convolutional Network recognition model.
* [`encoding.py`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/models/recognition/encoding.py): Character mapping dictionary (`SUPPORT_CHARACTERS`) and CTC greedy decoder.
* [`recognition_dataset.py`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/models/recognition/recognition_dataset.py): `RecognitionDataset` and `RecognitionDatasetWithAugmentation`.
* [`train.py`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/models/recognition/train.py): Recognition model training loop.

### Synthetic Data Generation (`comic_ocr/dataset/`)
* [`generated_manga/`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/dataset/generated_manga/): Synthetic comic page generator (`MangaGenerator`) with speech bubbles (`TextBubble`), caption boxes (`TextRect`), text layout (`TextArea`), and multi-level semantic segmentation masks (`create_image_mask`).
* [`generated_single_line/`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/dataset/generated_single_line/): Synthetic single-line text generator (`SingleLineGenerator`) for OCR recognition training.
* [`annotated_manga/`](file:///Users/wanasit/Workspace/comic-ocr/comic_ocr/dataset/annotated_manga/): Real comic dataset loader for `labeling-notebook` JSON annotation format.

---

## 3. Environment & Development Workflow

### Dependency Management with `uv`
The project uses `uv` and standard PEP 621 `pyproject.toml` with the `hatchling` build backend:

```bash
# Sync dependencies in local .venv
uv sync

# Run tests
uv run pytest

# Build distribution packages (wheel / sdist)
uv build

# Add a runtime or dev package
uv add <package-name>
uv add --group dev <package-name>
```

### Pretrained Models & Git LFS
Default model weights are tracked via Git LFS in [`trained_models/`](file:///Users/wanasit/Workspace/comic-ocr/trained_models/):
* `trained_models/localization.pth`
* `trained_models/recognition.pth`

When working in a fresh clone:
```bash
git lfs install
git lfs pull
```

### Git & Commit Message Format

Follow the existing repository commit conventions:
- `Fix: <description>` for bug fixes and compatibility updates.
- `New: <description>` for new features or architectures.
- `Improve: <description>` or `Refactor: <description>` for enhancements.
- `Update: <description>` for documentation and dependency updates.

---

## 4. Other Notes and Known Pitfalls

1. **PyTorch DataLoader Multiprocessing on macOS / Sandboxed Environments**:
   * PyTorch's `DataLoader(..., num_workers > 0)` uses POSIX shared memory file descriptors (`storage._share_filename_cpu_()`). In macOS sandboxes or environments without shared memory access, this causes `RuntimeError: Operation not permitted`.
   * **Rule**: When writing or running tests, always default to `num_workers=0`.

2. **PyTorch 2.6+ Checkpoint Loading**:
   * `torch.load()` in PyTorch 2.6+ defaults to `weights_only=True`, which raises errors when unpickling custom PyTorch module architectures.
   * **Rule**: When loading full model checkpoints, pass `weights_only=False`.

3. **Pillow 10+ / 11+ Font Metrics & Text Rendering Mechanics**:
   * Do **not** use deprecated `font.getsize()` or `font.getsize_multiline()`.
   * Use `TextArea.text_size(text, font, spacing)` or `font.getbbox(text)` / `ImageDraw.Draw.multiline_textbbox()`.
   * FreeType fonts in Pillow start rendering glyphs at `draw_y + y0` where `y0 = font.getbbox(lines[0])[1]`.
   * To align the top of rendered text tightly with `xy.y`, offset the drawing coordinate: `draw_y = xy.y + padding_line_ascent - first_y0`.

4. **Bounding Box Geometry & `text_rect`**:
   * `text_rect` is defined as `Rectangle.union_bounding_rect([l.location for l in self.text_lines])`.
   * Since each line `l.location` already incorporates `padding_x`, `padding_line_ascent`, and `padding_line_decent`, taking the union ensures tight, symmetrical bounding boxes for text with and without descenders.

5. **Jupyter Notebook Hygiene**:
   * When committing `.ipynb` notebooks (e.g. `notebooks/Generate Datasets.ipynb`), always strip all outputs (`outputs: []`, `execution_count: null`) and remove runtime metadata (`ExecuteTime`).

6. **No Inline Scripts**:
   * Avoid executing inline Python snippets (`python -c "..."`). Write proper test cases under [`tests/`](file:///Users/wanasit/Workspace/comic-ocr/tests/) or place runnable scripts in an appropriate module / scratch directory.

7. **Network Operations in Tests**:
   * Unit tests must run offline without requiring external network downloads. Mock network calls (e.g. `torch.hub.download_url_to_file`) via `tests/conftest.py`.

