"""Smoke-test the packaged OCR process contract with one saved bib image."""

from __future__ import annotations

import argparse
import multiprocessing
import time
from pathlib import Path

from realtime.ocr_worker import OcrJob, OcrProcessController


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--result-timeout", type=float, default=15.0)
    args = parser.parse_args()

    image = args.image.expanduser().resolve()
    if not image.is_file():
        print(f"Image not found: {image}")
        return 2

    controller = OcrProcessController(cpu_threads=args.cpu_threads)
    try:
        if not controller.start():
            print("OCR process did not start")
            return 3

        ready_deadline = time.monotonic() + max(1.0, args.startup_timeout)
        while time.monotonic() < ready_deadline:
            for result in controller.poll():
                if result.status == "FAILED":
                    print(result)
                    return 4
                if result.status == "READY":
                    break
            else:
                time.sleep(0.1)
                continue
            break
        else:
            print("OCR process startup timed out")
            return 5

        job = OcrJob(event_id=1, event_dir=str(image.parent), candidate_paths=(str(image),))
        if not controller.submit(job):
            print("OCR job queue rejected the smoke task")
            return 6

        result_deadline = time.monotonic() + max(1.0, args.result_timeout)
        while time.monotonic() < result_deadline:
            for result in controller.poll():
                if result.event_id == job.event_id:
                    print(result)
                    return 0
            time.sleep(0.1)
        print("OCR result timed out")
        return 7
    finally:
        controller.stop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
