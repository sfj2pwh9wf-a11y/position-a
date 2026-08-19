import math
from typing import Dict, List, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import gnubg_nn

# Adikus iPhone board geometry. Coordinates are normalized to the uploaded image.
# The app deliberately detects the board from the screenshot rather than relying
# on the text around it (player names, scores, timers, etc.).
X_NORM = [
    0.0564, 0.1326, 0.2073, 0.2835, 0.3597, 0.4360,
    0.5614, 0.6375, 0.7137, 0.7898, 0.8650, 0.9410,
]
TOP_Y_NORM = 0.254
BOTTOM_Y_NORM = 0.804
STACK_STEP_NORM = 0.0340
BAR_X_NORM = 0.500
BAR_Y_NORM = 0.52

TOP_LEFT_POINTS = [24, 23, 22, 21, 20, 19]
TOP_RIGHT_POINTS = [13, 14, 15, 16, 17, 18]
BOTTOM_LEFT_POINTS = [12, 11, 10, 9, 8, 7]
BOTTOM_RIGHT_POINTS = [6, 5, 4, 3, 2, 1]


def rgb_luminance(pixels: np.ndarray) -> np.ndarray:
    return (
        0.2126 * pixels[:, 0]
        + 0.7152 * pixels[:, 1]
        + 0.0722 * pixels[:, 2]
    )


def sample_patch(rgb: np.ndarray, x: int, y: int, radius: int = 12):
    h, w = rgb.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = rgb[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0, 0.0, 0.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    pixels = patch[mask]
    lum = rgb_luminance(pixels)
    return float(np.median(lum)), float(np.mean(lum > 160)), float(np.mean(lum < 70))


def classify_at(rgb: np.ndarray, x: int, y: int) -> str:
    # The Adikus board is brown, while both checker colors are close to neutral
    # gray/white/black.  RGB neutrality is therefore much safer than a simple
    # luminance threshold (brown wood can be quite dark).
    h, w = rgb.shape[:2]
    radius = max(8, int(round(0.017 * w)))
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    pixels = rgb[y0:y1, x0:x1][mask]
    if pixels.size == 0:
        return "."
    med = np.median(pixels, axis=0)
    spread = float(np.max(med) - np.min(med))
    lo, hi = float(np.min(med)), float(np.max(med))
    if lo >= 170 and spread <= 28:
        return "W"
    if hi <= 75 and spread <= 28:
        return "B"
    return "."


def board_geometry(image: Image.Image):
    w, h = image.size
    xs = [int(round(v * w)) for v in X_NORM]
    top_y = int(round(TOP_Y_NORM * h))
    bottom_y = int(round(BOTTOM_Y_NORM * h))
    step = max(20, int(round(STACK_STEP_NORM * h)))
    return xs, top_y, bottom_y, step


def detect_stack(rgb: np.ndarray, x: int, y0: int, direction: int, max_n: int = 15):
    """Read a stack from the outside toward the board.

    A checker stack is a run of same-color circular centers. We stop at the
    first non-checker slot so ordinary wood grain cannot be counted as pieces.
    """
    vals = []
    for i in range(max_n):
        y = int(round(y0 + direction * i * (STACK_STEP_NORM * rgb.shape[0])))
        c = classify_at(rgb, x, y)
        vals.append(c)

    if not vals or vals[0] == ".":
        return ".", 0

    color = vals[0]
    count = 0
    for c in vals:
        if c != color:
            break
        count += 1
    return color, count


def detect_bar(rgb: np.ndarray) -> Tuple[int, int]:
    """Detect checkers sitting on the central bar using Hough circles.

    Bar checkers are uncommon but must be represented because they change legal
    moves. We use Hough only in a narrow central strip and classify each circle.
    """
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    x_center = int(BAR_X_NORM * w)
    x0, x1 = max(0, x_center - int(.055 * w)), min(w, x_center + int(.055 * w))
    y0, y1 = int(.22 * h), int(.84 * h)
    crop = cv2.GaussianBlur(gray[y0:y1, x0:x1], (5, 5), 1.1)
    min_r = max(10, int(.023 * w))
    max_r = max(16, int(.043 * w))
    circles = cv2.HoughCircles(
        crop,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, int(.035 * w)),
        param1=100,
        param2=30,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return 0, 0

    seen: List[Tuple[int, int]] = []
    white = black = 0
    for cx, cy, r in np.round(circles[0]).astype(int):
        x, y = cx + x0, cy + y0
        # Exclude dice/central UI areas; bar checkers are close to the ridge.
        if abs(x - x_center) > int(.045 * w):
            continue
        if any((x - sx) ** 2 + (y - sy) ** 2 < 14 ** 2 for sx, sy in seen):
            continue
        seen.append((x, y))
        c = classify_at(rgb, x, y)
        if c == "W":
            white += 1
        elif c == "B":
            black += 1
    return white, black


def detect_board(image: Image.Image):
    rgb = np.asarray(image.convert("RGB"))
    xs, top_y, bottom_y, step = board_geometry(image)
    white = [0] * 25
    black = [0] * 25

    def put(point: int, color: str, count: int):
        if count <= 0:
            return
        if color == "W":
            white[point] = count
        elif color == "B":
            black[point] = count

    for i, x in enumerate(xs[:6]):
        c, n = detect_stack(rgb, x, top_y, +1)
        put(TOP_LEFT_POINTS[i], c, n)
    for i, x in enumerate(xs[6:]):
        c, n = detect_stack(rgb, x, top_y, +1)
        put(TOP_RIGHT_POINTS[i], c, n)
    for i, x in enumerate(xs[:6]):
        c, n = detect_stack(rgb, x, bottom_y, -1)
        put(BOTTOM_LEFT_POINTS[i], c, n)
    for i, x in enumerate(xs[6:]):
        c, n = detect_stack(rgb, x, bottom_y, -1)
        put(BOTTOM_RIGHT_POINTS[i], c, n)

    bar_w, bar_b = detect_bar(rgb)
    white[0] = bar_w
    black[0] = bar_b

    return white, black


def checker_total(side: List[int]) -> int:
    return sum(side[1:]) + side[0]


def normalize_counts(side: List[int]) -> Tuple[List[int], int]:
    """Return board counts and inferred borne-off count.

    Point 0 is bar. Borne-off checkers are not visible in the board image and
    are inferred from the mandatory 15-checker total.
    """
    visible = checker_total(side)
    off = 15 - visible
    return side, off


def validate_position(w: List[int], b: List[int]) -> List[str]:
    errors = []
    for name, side in (("White", w), ("Black", b)):
        visible = checker_total(side)
        if visible > 15:
            errors.append(f"{name}: detected {visible} checkers; maximum is 15.")
    for p in range(1, 25):
        if w[p] and b[p]:
            errors.append(f"Point {p}: both colors were detected on the same point.")
    # A normal screenshot should show 15 each unless checkers have been borne off.
    if checker_total(w) == 0 or checker_total(b) == 0:
        errors.append("One side has no visible checkers; the screenshot could not be mapped reliably.")
    return errors


def point_text(side: List[int], off: int) -> str:
    parts = []
    if side[0]:
        parts.append(f"bar={side[0]}")
    for p in range(24, 0, -1):
        if side[p]:
            parts.append(f"{p}:{side[p]}")
    if off:
        parts.append(f"off={off}")
    return "  ".join(parts) if parts else "none"


def board_matrix(white: List[int], black: List[int], my_color: str):
    """Convert screen point numbering to GNUBG X/O orientation.

    The Adikus numbering used here is the standard player-relative numbering:
    White moves 24→1 and Black moves 1→24. GNUBG expects a 2x25 board where
    row 0 is X and row 1 is O, with index 0 as the special bar/off slot.
    For the user's White turn, White is X and Black is O.
    """
    if my_color == "White":
        return [list(white), list(black)], "X"
    return [list(black), list(white)], "X"


def normalize_move(move):
    if move is None:
        return []
    # Modern gnubg-nn returns [(from,to), ...].
    if isinstance(move, (list, tuple)) and move:
        if isinstance(move[0], (list, tuple)):
            return [(int(a), int(b)) for a, b in move]
        if all(isinstance(x, (int, np.integer)) for x in move):
            vals = list(map(int, move))
            return list(zip(vals[0::2], vals[1::2]))
    return []


def move_text(move):
    if not move:
        return "No legal move found"
    def fmt(v):
        if v in (0, 25):
            return "bar" if v == 25 else "off"
        return str(v)
    return ", ".join(f"{fmt(a)}/{fmt(b)}" for a, b in move)


def analyze(board, d1: int, d2: int, side: str):
    # Use the documented API. n=2 gives a deeper neural-net lookahead than the
    # old 0-ply call, while list=True exposes alternate candidates for audit.
    result = gnubg_nn.best_move(board, d1, d2, n=2, s=side, list=True)
    best = normalize_move(result[0] if isinstance(result, tuple) and result else result)
    alternatives = []
    if isinstance(result, tuple) and len(result) >= 4:
        move_list = result[3]
        if isinstance(move_list, (list, tuple)):
            for item in move_list[:8]:
                try:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        m = normalize_move(item[1])
                        score = item[-1] if isinstance(item[-1], (float, int)) else None
                        alternatives.append((m, score))
                except Exception:
                    pass
    return best, alternatives


st.set_page_config(page_title="Backgammon Analyzer", page_icon="🎲", layout="centered")
st.title("🎲 Backgammon Analyzer")
st.caption("Adikus screenshot → full 30-checker position → GNUBG move analysis")

with st.expander("How this version works", expanded=False):
    st.write(
        "The screenshot is read fresh every time. Both White and Black checkers are "
        "mapped point-by-point, bar checkers are checked, and the inferred borne-off "
        "count is included. The complete 2×25 position is then sent to GNUBG; the "
        "opponent's position therefore participates in the move calculation."
    )

my_color = st.selectbox("My color", ["White", "Black"], index=0)
turn = st.selectbox("Whose turn is shown in the screenshot?", ["Me", "Opponent"], index=0)

file = st.file_uploader("Upload your latest Adikus screenshot", type=["png", "jpg", "jpeg"])
if file:
    image = Image.open(file).convert("RGB")
    w, b = detect_board(image)
    errors = validate_position(w, b)

    st.image(image, use_container_width=True)

    if errors:
        st.error("Position could not be safely verified — I will not guess a move.")
        for e in errors:
            st.write("• " + e)
        st.stop()

    w, w_off = normalize_counts(w)
    b, b_off = normalize_counts(b)

    st.success("✓ Full position reconstructed")
    st.subheader("Detected board")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**WHITE**")
        st.code(point_text(w, w_off), language="text")
    with c2:
        st.markdown("**BLACK**")
        st.code(point_text(b, b_off), language="text")

    if turn == "Opponent":
        st.warning("This screenshot is marked as the opponent's turn. Switch it to 'Me' before asking for your move.")
        st.stop()

    st.subheader("Dice")
    c1, c2 = st.columns(2)
    with c1:
        d1 = st.number_input("Die 1", 1, 6, 1, 1)
    with c2:
        d2 = st.number_input("Die 2", 1, 6, 1, 1)

    st.caption("The board is read fresh from every screenshot. Enter only the two dice.")

    if st.button("Find strongest move", type="primary", use_container_width=True):
        try:
            board, side = board_matrix(w, b, my_color)

            # A second engine-side structural check prevents malformed OCR/CV
            # output from reaching the evaluator.
            if len(board) != 2 or any(len(row) != 25 for row in board):
                raise ValueError("Internal board must be 2×25.")
            if any(v < 0 for row in board for v in row):
                raise ValueError("Internal board contains a negative checker count.")

            best, alternatives = analyze(board, int(d1), int(d2), side)

            st.success("BEST MOVE")
            st.markdown(f"## {move_text(best)}")
            st.caption("GNUBG 2-ply neural-net move analysis; opponent's full position is included.")

            if alternatives:
                st.subheader("Engine candidates")
                for idx, (m, score) in enumerate(alternatives[:5], 1):
                    label = move_text(m)
                    if score is not None:
                        st.write(f"{idx}. **{label}** — engine score {score:.4f}")
                    else:
                        st.write(f"{idx}. **{label}**")
        except Exception as exc:
            st.error("The engine could not analyze this verified position.")
            st.exception(exc)
