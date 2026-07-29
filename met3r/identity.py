"""Stable content identities shared by feature and geometry artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np


def image_set_id(images: Sequence[str | Path | np.ndarray]) -> str:
    digest = hashlib.sha256()
    for image in images:
        if isinstance(image, (str, Path)):
            path = Path(image).expanduser().resolve()
            digest.update(path.read_bytes())
        else:
            array = np.asarray(image)
            digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()
