#!/usr/bin/env python3
"""Generate sample model and input binaries for the CUDA inference demo."""

from __future__ import annotations

import argparse
import random
import struct
from pathlib import Path


def floats(count: int, rng: random.Random, low: float, high: float) -> list[float]:
    return [rng.uniform(low, high) for _ in range(count)]


def write_floats(handle, values: list[float]) -> None:
    handle.write(struct.pack(f"<{len(values)}f", *values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--input-dim", type=int, default=784)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    w1 = floats(args.input_dim * args.hidden_dim, rng, -0.12, 0.12)
    b1 = floats(args.hidden_dim, rng, -0.02, 0.02)
    w2 = floats(args.hidden_dim * args.output_dim, rng, -0.12, 0.12)
    b2 = floats(args.output_dim, rng, -0.02, 0.02)
    x = floats(args.batch * args.input_dim, rng, 0.0, 1.0)

    with (output_dir / "weights.bin").open("wb") as handle:
        handle.write(b"CNNI")
        handle.write(struct.pack("<Iiiii", 1, args.batch, args.input_dim, args.hidden_dim, args.output_dim))
        write_floats(handle, w1)
        write_floats(handle, b1)
        write_floats(handle, w2)
        write_floats(handle, b2)

    with (output_dir / "sample_input.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(x)))
        write_floats(handle, x)

    print(f"Wrote {output_dir / 'weights.bin'}")
    print(f"Wrote {output_dir / 'sample_input.bin'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

