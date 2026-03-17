# scanpdf-cli

`scanpdf-cli` turns photos of handwritten pages into cleaner, compressed PDFs.

It:
- detects and warps the page
- deskews slight tilt
- boosts readability (contrast/sharpen/brightness/whitening)
- aggressively compresses images
- builds PDF and post-compresses with Ghostscript

## Before / After

| Before | After |
|---|---|
| ![Before](assets/IMG_2155_before.jpg) | ![After](assets/IMG_2155.jpg) |

## Prerequisites

Install system tools:

- `magick` (ImageMagick)
- `img2pdf`
- `gs` (Ghostscript)

Install Python deps through package install (`opencv-python`, `numpy` are in `pyproject.toml`).

## Install (recommended: pipx)

From GitHub:

```bash
pipx install git+https://github.com/<owner>/<repo>.git
```

From local checkout:

```bash
pipx install .
```

## Usage

```bash
scanpdf [INPUT_DIR] [OUTPUT.pdf] [options]
```

Examples:

```bash
scanpdf ./pages notes.pdf
scanpdf ./pages notes.pdf --brightness-level bright --paper-whiten 0.6
scanpdf ./pages notes.pdf --input-jpeg-quality 50 --jpeg-quality 55 --gs-profile /screen
scanpdf ./pages notes.pdf --debug --debug-dir ./debug_out
```

Key options:
- `--enhance-profile soft|balanced|strong`
- `--brightness-level dark|normal|bright|very-bright`
- `--paper-whiten 0.0-1.0`
- `--input-jpeg-quality 1-100`
- `--jpeg-quality 1-100`
- `--jobs N`
- `--gs-profile /screen|/ebook|/printer|/prepress|/default`

## Preprocess single image

```bash
scanpdf-preprocess input.jpg output.jpg --brightness-level bright --paper-whiten 0.5
```

## Project layout

```text
src/scanpdf/cli.py          # main scanpdf command
src/scanpdf/preprocess.py   # page processing pipeline
pyproject.toml              # packaging metadata
```

## Build a wheel/sdist locally

```bash
python -m pip install --upgrade build
python -m build
```

Artifacts appear in `dist/`.
