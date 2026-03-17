#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from .preprocess import BRIGHTNESS_LEVELS, ENHANCEMENT_PROFILES, PreprocessConfig, process_image

SUPPORTED_SUFFIXES = {".heic", ".png", ".jpg", ".jpeg"}


def natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def run_command(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or "Unknown command failure"
        raise RuntimeError(f"Command failed ({' '.join(cmd[:2])}): {message}")


def sanitize_stem(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess images and build compressed PDF output.")
    parser.add_argument("input_dir", nargs="?", default=".", help="Input directory containing image pages")
    parser.add_argument("output", nargs="?", default="output.pdf", help="Output PDF path")

    parser.add_argument("--stop-on-error", action="store_true", help="Stop on first page failure")
    parser.add_argument("--debug", action="store_true", help="Save intermediate processing outputs")
    parser.add_argument("--debug-dir", help="Directory for debug outputs")
    parser.add_argument("--jobs", type=int, default=0, help="Parallel preprocessing workers (default: CPU count)")
    parser.add_argument("--input-jpeg-quality", type=int, default=55, help="Aggressive input compression quality (default: 55)")
    parser.add_argument("--gs-profile", default="/ebook", help="Ghostscript PDFSETTINGS profile (default: /ebook)")

    parser.add_argument("--inner-crop", type=float, default=0.02, help="Inner crop ratio after warp")
    parser.add_argument("--min-page-area-ratio", type=float, default=0.55, help="Minimum page area ratio for detection")
    parser.add_argument("--deskew-min-angle", type=float, default=0.5, help="Minimum deskew angle to apply")
    parser.add_argument("--deskew-max-angle", type=float, default=20.0, help="Maximum deskew angle to apply")
    parser.add_argument(
        "--enhance-profile",
        choices=sorted(ENHANCEMENT_PROFILES.keys()),
        default="balanced",
        help="Enhancement profile",
    )
    parser.add_argument(
        "--brightness-level",
        choices=sorted(BRIGHTNESS_LEVELS.keys()),
        default="normal",
        help="Output brightness level",
    )
    parser.add_argument("--paper-whiten", type=float, default=0.35, help="Background whitening strength (0.0-1.0)")
    parser.add_argument("--grayscale", action="store_true", help="Convert final pages to grayscale")
    parser.add_argument("--png-compression", type=int, default=3, help="PNG compression level 0-9")
    parser.add_argument("--jpeg-quality", type=int, default=60, help="JPEG quality 1-100 for processed pages")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not Path(args.input_dir).is_dir():
        raise RuntimeError(f"Input directory does not exist: {args.input_dir}")
    if args.jobs < 0:
        raise RuntimeError("--jobs must be >= 0")
    if not 1 <= args.input_jpeg_quality <= 100:
        raise RuntimeError("--input-jpeg-quality must be in [1, 100]")
    if not 0.0 <= args.paper_whiten <= 1.0:
        raise RuntimeError("--paper-whiten must be in [0.0, 1.0]")


def ensure_dependencies() -> None:
    for cmd in ("magick", "img2pdf", "gs"):
        if shutil.which(cmd) is None:
            raise RuntimeError(f"Missing dependency: {cmd}")


def collect_sources(input_dir: Path) -> list[Path]:
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    return sorted(files, key=lambda p: natural_key(p.name))


def convert_inputs_to_jpeg(
    sources: list[Path], tmp_dir: Path, quality: int, stop_on_error: bool
) -> tuple[list[tuple[int, str, str, Path]], list[str]]:
    jobs: list[tuple[int, str, str, Path]] = []
    failures: list[str] = []

    for index, src in enumerate(sources, start=1):
        file_name = src.name
        base = f"{index:04d}_{sanitize_stem(src.stem)}"
        input_path = tmp_dir / f"{base}_input.jpg"
        cmd = [
            "magick",
            str(src),
            "-auto-orient",
            "-background",
            "white",
            "-alpha",
            "remove",
            "-alpha",
            "off",
            "-strip",
            "-sampling-factor",
            "4:2:0",
            "-interlace",
            "Plane",
            "-quality",
            str(quality),
            str(input_path),
        ]
        try:
            run_command(cmd)
        except Exception:
            failures.append(f"{file_name} (input compression)")
            if stop_on_error:
                break
            continue
        jobs.append((index, file_name, base, input_path))
    return jobs, failures


def preprocess_pages(
    jobs: list[tuple[int, str, str, Path]],
    tmp_dir: Path,
    base_config: PreprocessConfig,
    workers: int,
    stop_on_error: bool,
) -> tuple[list[Path], list[str]]:
    failed_pages: list[str] = []
    processed_by_index: dict[int, Path] = {}

    def worker(job: tuple[int, str, str, Path]) -> tuple[int, str, Path]:
        index, page_name, base, input_path = job
        output_path = tmp_dir / f"{base}_processed.jpg"
        config = dataclasses.replace(base_config, debug_prefix=base)
        process_image(str(input_path), str(output_path), config)
        return index, page_name, output_path

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(worker, job): job for job in jobs}
        for future in concurrent.futures.as_completed(future_map):
            index, page_name, _, _ = future_map[future]
            try:
                out_index, _, out_path = future.result()
                processed_by_index[out_index] = out_path
            except Exception as exc:
                failed_pages.append(f"{page_name} (preprocess: {exc})")
                if stop_on_error:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    processed = [processed_by_index[i] for i in sorted(processed_by_index.keys())]
    return processed, failed_pages


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        validate_args(args)
        ensure_dependencies()
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    input_dir = Path(args.input_dir).resolve()
    sources = collect_sources(input_dir)
    if not sources:
        raise SystemExit(f"No supported images found in {input_dir}")

    workers = args.jobs or (os.cpu_count() or 4)
    if args.stop_on_error:
        workers = 1

    debug_enabled = args.debug or bool(args.debug_dir)
    debug_dir = args.debug_dir
    if debug_enabled and not debug_dir:
        debug_dir = str(Path.cwd() / f"scanpdf_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)

    base_config = PreprocessConfig(
        inner_crop=args.inner_crop,
        min_page_area_ratio=args.min_page_area_ratio,
        deskew_min_angle=args.deskew_min_angle,
        deskew_max_angle=args.deskew_max_angle,
        enhance_profile=args.enhance_profile,
        brightness_level=args.brightness_level,
        paper_whiten=args.paper_whiten,
        grayscale=args.grayscale,
        png_compression=args.png_compression,
        jpeg_quality=args.jpeg_quality,
        debug_dir=debug_dir if debug_enabled else None,
        debug_prefix="debug",
    )

    with tempfile.TemporaryDirectory(prefix="scanpdf.") as tmp:
        tmp_dir = Path(tmp)
        print(f"Found {len(sources)} image(s). Using {workers} worker(s).")
        print(f"Pre-compressing inputs (JPEG quality: {args.input_jpeg_quality})...")

        jobs, failures = convert_inputs_to_jpeg(
            sources, tmp_dir, quality=args.input_jpeg_quality, stop_on_error=args.stop_on_error
        )
        if not jobs:
            raise SystemExit("No pages available after input compression; PDF not created.")

        print(f"Processing {len(jobs)} page(s)...")
        processed, preprocess_failures = preprocess_pages(
            jobs=jobs,
            tmp_dir=tmp_dir,
            base_config=base_config,
            workers=workers,
            stop_on_error=args.stop_on_error,
        )
        failures.extend(preprocess_failures)

        if not processed:
            raise SystemExit("No pages were processed successfully; PDF not created.")

        raw_pdf = tmp_dir / "intermediate.pdf"
        output_pdf = Path(args.output).resolve()
        print("Building intermediate PDF...")
        run_command(["img2pdf", *(str(p) for p in processed), "-o", str(raw_pdf)])

        print(f"Applying Ghostscript compression ({args.gs_profile})...")
        run_command(
            [
                "gs",
                "-sDEVICE=pdfwrite",
                "-dCompatibilityLevel=1.4",
                f"-dPDFSETTINGS={args.gs_profile}",
                "-dNOPAUSE",
                "-dQUIET",
                "-dBATCH",
                f"-sOutputFile={output_pdf}",
                str(raw_pdf),
            ]
        )
        print(f"Done: {output_pdf}")

        if failures:
            print(f"Completed with {len(failures)} failed page(s):")
            for item in failures:
                print(f"  - {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
