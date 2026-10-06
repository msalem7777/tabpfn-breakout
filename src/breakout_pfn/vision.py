"""Pixel-based fallback ball detection for the game adapter."""
import numpy as np

def detect_ball_xy(frame: np.ndarray) -> tuple[float, float] | None:
    """Fallback visual ball detector, returning normalized screen coordinates.

    This is deliberately only a fallback when ALE RAM/object state is not
    available. It searches the playfield below the brick band and above the
    paddle for tiny bright connected components.
    """
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] < 3:
        return None
    h, w = image.shape[:2]
    y0, y1 = int(h * 0.43), int(h * 0.84)
    crop = image[y0:y1, :, :3]
    bright = crop.max(axis=2) >= 145

    # Tiny flood-fill components are sufficient at Atari resolution.
    seen = np.zeros_like(bright, dtype=bool)
    candidates: list[tuple[int, float, float]] = []
    for yy in range(bright.shape[0]):
        for xx in range(bright.shape[1]):
            if not bright[yy, xx] or seen[yy, xx]:
                continue
            stack = [(yy, xx)]
            seen[yy, xx] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < bright.shape[0] and 0 <= nx < bright.shape[1] and bright[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if 1 <= len(coords) <= 20:
                ys = np.asarray([c[0] for c in coords], dtype=float)
                xs = np.asarray([c[1] for c in coords], dtype=float)
                width = float(xs.max() - xs.min() + 1)
                height = float(ys.max() - ys.min() + 1)
                if width <= 6 and height <= 6:
                    candidates.append((len(coords), float(xs.mean()), float(ys.mean() + y0)))
    if not candidates:
        return None
    # Prefer the largest tiny component; stable enough as a fallback.
    _, x, y = max(candidates, key=lambda item: item[0])
    return (
        float(np.clip(x / max(w - 1, 1), 0.0, 1.0)),
        float(np.clip(y / max(h - 1, 1), 0.0, 1.0)),
    )
