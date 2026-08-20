from __future__ import annotations

import argparse
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


HEADER_SIZE = 48
CCD_PIXELS = 1024
PIXEL_BYTES = CCD_PIXELS * 3
RECORD_METADATA_SIZE = 8
RECORD_SIZE = RECORD_METADATA_SIZE + PIXEL_BYTES

START_FLAG = 0x01000000
END_FLAG = 0x02000000


@dataclass(frozen=True)
class Segment:
    index: int
    start_line: int
    end_line: int
    line_count: int
    start_tick: int
    end_tick: int
    duration_ms: float
    relative_start_ms: float


def unsigned_delta(value: int, base: int) -> int:
    return (value - base) & 0xFFFFFFFF


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def parse_header(header: bytes) -> dict[str, object]:
    if len(header) != HEADER_SIZE:
        raise ValueError(f"Expected {HEADER_SIZE} header bytes, got {len(header)}")

    return {
        "reference_tick": struct.unpack_from("<I", header, 0)[0],
        "event_group_id": struct.unpack_from("<I", header, 4)[0],
        "header_flag": struct.unpack_from("<I", header, 20)[0],
        "capture_config_hex": header[24:28].hex("-"),
        "date": (
            f"{struct.unpack_from('<H', header, 36)[0]:04d}-"
            f"{header[38]:02d}-{header[39]:02d}"
        ),
        "last_pixel_index": struct.unpack_from("<H", header, 44)[0],
    }


def find_segments(
    ticks: np.ndarray,
    flags: np.ndarray,
    reference_tick: int,
    tick_hz: int,
) -> list[Segment]:
    segments: list[Segment] = []
    start_line: int | None = None

    for line_index, flag in enumerate(flags):
        flag_value = int(flag)
        if flag_value == START_FLAG:
            start_line = line_index
            continue

        if flag_value != END_FLAG or start_line is None:
            continue

        start_tick = int(ticks[start_line])
        end_tick = int(ticks[line_index])
        segments.append(
            Segment(
                index=len(segments) + 1,
                start_line=start_line,
                end_line=line_index,
                line_count=line_index - start_line + 1,
                start_tick=start_tick,
                end_tick=end_tick,
                duration_ms=round(
                    unsigned_delta(end_tick, start_tick) * 1000 / tick_hz,
                    3,
                ),
                relative_start_ms=round(
                    unsigned_delta(start_tick, reference_tick) * 1000 / tick_hz,
                    3,
                ),
            )
        )
        start_line = None

    if start_line is not None:
        raise ValueError(f"Capture starting at line {start_line} has no end marker")

    return segments


def save_contact_sheet(
    segment_images: list[tuple[Segment, Image.Image]],
    output_path: Path,
) -> None:
    columns = 2
    tile_width = 1200
    tile_height = 430
    label_height = 54
    margin = 24
    rows = (len(segment_images) + columns - 1) // columns

    sheet = Image.new(
        "RGB",
        (
            columns * tile_width + (columns + 1) * margin,
            rows * tile_height + (rows + 1) * margin,
        ),
        (238, 241, 244),
    )
    draw = ImageDraw.Draw(sheet)
    font = load_font(24)

    for item_index, (segment, image) in enumerate(segment_images):
        row, column = divmod(item_index, columns)
        left = margin + column * (tile_width + margin)
        top = margin + row * (tile_height + margin)

        draw.rectangle(
            (left, top, left + tile_width, top + tile_height),
            fill=(255, 255, 255),
            outline=(168, 176, 184),
            width=2,
        )
        label = (
            f"Segment {segment.index:02d}  lines {segment.start_line}-{segment.end_line}"
            f"  duration {segment.duration_ms:.3f} ms"
            f"  start +{segment.relative_start_ms:.3f} ms"
        )
        draw.text((left + 16, top + 13), label, fill=(25, 31, 36), font=font)

        preview = image.copy()
        preview.thumbnail((tile_width - 32, tile_height - label_height - 24))
        image_left = left + (tile_width - preview.width) // 2
        image_top = top + label_height + (
            tile_height - label_height - preview.height
        ) // 2
        sheet.paste(preview, (image_left, image_top))

    sheet.save(output_path, quality=94, subsampling=0)


def parse_rgb(source: Path, output_dir: Path, tick_hz: int) -> None:
    raw = source.read_bytes()
    if len(raw) < HEADER_SIZE:
        raise ValueError("File is shorter than the RGB header")

    header = parse_header(raw[:HEADER_SIZE])
    body = raw[HEADER_SIZE:]
    if len(body) % RECORD_SIZE:
        raise ValueError(
            f"Image body size {len(body)} is not divisible by record size {RECORD_SIZE}"
        )

    line_count = len(body) // RECORD_SIZE
    records = np.frombuffer(body, dtype=np.uint8).reshape(line_count, RECORD_SIZE)
    ticks = records[:, :4].copy().view("<u4").reshape(-1)
    flags = records[:, 4:8].copy().view("<u4").reshape(-1)
    pixels = records[:, RECORD_METADATA_SIZE:].reshape(line_count, CCD_PIXELS, 3)

    reference_tick = int(header["reference_tick"])
    segments = find_segments(ticks, flags, reference_tick, tick_hz)
    if not segments:
        raise ValueError("No complete capture segments were found")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Scan lines form the horizontal time axis; CCD pixels form the vertical axis.
    full_image = Image.fromarray(pixels.transpose(1, 0, 2), mode="RGB")
    full_image.save(output_dir / "group3_full.png", optimize=True)

    full_preview = full_image.copy()
    full_preview.thumbnail((5000, 1200))
    full_preview.save(
        output_dir / "group3_full_preview.jpg",
        quality=95,
        subsampling=0,
    )

    segment_images: list[tuple[Segment, Image.Image]] = []
    for segment in segments:
        segment_pixels = pixels[segment.start_line : segment.end_line + 1]
        segment_image = Image.fromarray(
            segment_pixels.transpose(1, 0, 2),
            mode="RGB",
        )
        segment_image.save(
            output_dir / f"segment_{segment.index:02d}.jpg",
            quality=96,
            subsampling=0,
        )
        segment_images.append((segment, segment_image))

    save_contact_sheet(
        segment_images,
        output_dir / "segments_contact_sheet.jpg",
    )

    metadata = {
        "source": str(source),
        "source_size": len(raw),
        "header_size": HEADER_SIZE,
        "record_size": RECORD_SIZE,
        "ccd_pixels": CCD_PIXELS,
        "scan_line_count": line_count,
        "tick_hz": tick_hz,
        "header": header,
        "segments": [asdict(segment) for segment in segments],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse an AYT line-scan RGB file")
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--tick-hz",
        type=int,
        default=20_000,
        help="Device clock frequency used for relative timing",
    )
    args = parser.parse_args()
    parse_rgb(args.source, args.output_dir, args.tick_hz)


if __name__ == "__main__":
    main()
