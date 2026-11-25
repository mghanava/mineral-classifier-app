"""Provide stable seeds and scoped Torch random streams."""

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager

import torch

DEFAULT_SEED = 42
SEED_SCHEME = "sha256-v1"


def validate_seed(
    value: object, key: str, *, maximum: int | None = 2**32 - 1
) -> int | None:
    """Accept a nonnegative integer within the backend range, or explicit null."""
    if value is None:
        return None
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        bounds = (
            "a nonnegative integer"
            if maximum is None
            else f"an integer from 0 to {maximum}"
        )
        raise ValueError(f"Set {key} to {bounds} or null.")
    return value


def derive_seed(seed: int | None, *scope: str | int) -> int | None:
    """Derive a stable 32-bit seed for an ordered stage and purpose scope."""
    if seed is None:
        return None
    payload = json.dumps((SEED_SCHEME, seed, *scope), separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:4], "big")


@contextmanager
def torch_seed(seed: int | None, device: torch.device) -> Iterator[None]:
    """Seed and restore the CPU and selected CUDA stream for a stage operation."""
    if seed is None:
        yield
        return
    cuda = device.type == "cuda"
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if cuda
        else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(seed)
        if cuda:
            with torch.cuda.device(cuda_devices[0]):
                torch.cuda.manual_seed(seed)
        yield


def torch_generator(device: torch.device, seed: int | None) -> torch.Generator:
    """Construct an isolated device-local stream for explicit random draws."""
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    return generator
