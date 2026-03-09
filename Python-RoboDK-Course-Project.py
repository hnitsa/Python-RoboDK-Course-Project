import cv2
import numpy as np
from robodk import robolink, robomath as rdm

# =============================
# USER SETTINGS
# =============================

# Path to the photo (white paper + hook + 12 cm blue line)
IMAGE_PATH = r"D:\Mantas'\test3.jpg"   # <-- change this

# Real length of the blue line (mm)
BLUE_LINE_LENGTH_MM = 120.0  # 12 cm

# If scale fails, hook will be fitted into this size (mm)
FALLBACK_DRAW_WIDTH_MM  = 120.0
FALLBACK_DRAW_HEIGHT_MM = 160.0

# Hook contour simplification
POINT_STEP       = 1          # take every Nth point
EPSILON_REL      = 0.001      # 0 = no simplification

# HSV range for the blue line, taken from your photo
# (OpenCV HSV: H 0-179, S 0-255, V 0-255)
BLUE_LOWER_HSV = (95, 40, 40)
BLUE_UPPER_HSV = (130, 255, 255)

# Shape filters for the blue line
MIN_LINE_AREA   = 500.0       # px area
MIN_LINE_ASPECT = 8.0         # length/thickness

# Flip directions when mapping image → robot XY
FLIP_X = False
FLIP_Y = True                 # image Y down → robot Y up

# Z heights (relative to StartDraw where pen touches paper)
Z_LIFT = 40.0                 # mm above paper
Z_DRAW = 0.0                  # pen on paper

# Speeds (mm/s)
SPEED_MOVE = 200.0
SPEED_DRAW = 80.0

# RoboDK item names
ROBOT_NAME        = 'Yaskawa HC10DTP'
TOOL_NAME         = 'Marker'
START_TARGET_NAME = 'StartDraw'


# =============================
# Helper: version-safe findContours
# =============================

def find_contours_compat(img, mode, method):
    cnts = cv2.findContours(img, mode, method)
    if len(cnts) == 2:
        contours, hierarchy = cnts
    else:
        _, contours, hierarchy = cnts
    return list(contours), hierarchy


# =============================
# 0. Detect white paper and crop ROI
# =============================

def detect_paper_roi(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # bright paper vs dark table
    _, bin_img = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # make sure white = paper
    if np.mean(gray[bin_img == 255]) < np.mean(gray[bin_img == 0]):
        bin_img = 255 - bin_img

    kernel = np.ones((5, 5), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = find_contours_compat(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        h, w = img_bgr.shape[:2]
        print('[PAPER] No paper found, using full image.')
        return img_bgr.copy(), (0, 0, w, h)

    contours.sort(key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])
    print(f'[PAPER] ROI x={x}, y={y}, w={w}, h={h}')
    roi = img_bgr[y:y + h, x:x + w].copy()
    return roi, (x, y, w, h)


# =============================
# 1. Detect blue line → scale (mm/px)
# =============================

def detect_blue_line_scale(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOWER_HSV, BLUE_UPPER_HSV)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = find_contours_compat(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        print('[BLUE] No blue contours found, fallback scale will be used.')
        return None, None, mask

    h, w = mask.shape[:2]
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_LINE_AREA:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect

        length = max(rw, rh)
        thickness = max(1.0, min(rw, rh))
        aspect = length / thickness

        # long thin shape near bottom of the paper
        if aspect < MIN_LINE_ASPECT:
            continue
        if cy < 0.6 * h:
            continue

        candidates.append((aspect, length, rect, cnt))

    if not candidates:
        # fallback: choose contour with max aspect ratio
        print('[BLUE] No strong line candidates, using contour with largest aspect ratio.')
        best = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_LINE_AREA:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect
            length = max(rw, rh)
            thickness = max(1.0, min(rw, rh))
            aspect = length / thickness
            if best is None or aspect > best[0]:
                best = (aspect, length, rect, cnt)
        if best is None:
            print('[BLUE] Still no usable blue contour, fallback scale.')
            return None, None, mask
        candidates = [best]

    # pick best candidate (highest aspect)
    candidates.sort(key=lambda x: x[0], reverse=True)
    aspect, length, rect, line_cnt = candidates[0]

    length_px = length
    mm_per_px = BLUE_LINE_LENGTH_MM / float(length_px)
    print(f'[BLUE] Chosen line: aspect={aspect:.2f}, length_px={length_px:.1f}')
    print(f'[BLUE] Estimated mm_per_px from blue line: {mm_per_px:.4f} mm/px')

    return mm_per_px, line_cnt, mask


# =============================
# 2. Detect hook contour on paper
# =============================

def find_hook_contour(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # hook is darker than paper → invert
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    h, w = binary.shape[:2]
    img_area = float(h * w)

    contours, _ = find_contours_compat(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise RuntimeError('[OBJ] No contours found on paper.')

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)
        x, y, bw, bh = cv2.boundingRect(c)
        rect = cv2.minAreaRect(c)
        (rw, rh) = rect[1]
        elong = max(rw, rh) / max(1.0, min(rw, rh)) if rw and rh else 0.0

        # ignore tiny junk
        if area < 500:
            continue
        # ignore whole paper region
        if area > 0.8 * img_area or (bw > 0.9 * w and bh > 0.9 * h):
            continue
        # ignore line-like contours
        if elong > 8.0:
            continue

        if EPSILON_REL > 0.0:
            eps = EPSILON_REL * cv2.arcLength(c, True)
            c = cv2.approxPolyDP(c, eps, True)
        c = c[::max(1, POINT_STEP)]
        candidates.append(c.reshape(-1, 2))

    if not candidates:
        print('[OBJ] No candidates after filters, using largest non-huge contour.')
        nonhuge = [c for c in contours if cv2.contourArea(c) < 0.8 * img_area]
        if not nonhuge:
            nonhuge = contours
        nonhuge.sort(key=cv2.contourArea, reverse=True)
        c = nonhuge[0]
        if EPSILON_REL > 0.0:
            eps = EPSILON_REL * cv2.arcLength(c, True)
            c = cv2.approxPolyDP(c, eps, True)
        c = c[::max(1, POINT_STEP)]
        candidates = [c.reshape(-1, 2)]

    candidates.sort(key=lambda c: cv2.contourArea(c.reshape(-1, 1, 2)), reverse=True)
    best = candidates[0]
    print(f'[OBJ] Using contour with {len(best)} points for hook.')
    return [best], binary


# =============================
# 3. Preview (for debugging)
# =============================

def preview_all(roi_bgr, hook_contours, hook_binary, blue_cnt, blue_mask):
    cv2.imwrite('hook_binary_debug.png', hook_binary)
    print('Saved hook_binary_debug.png')

    if blue_mask is not None:
        cv2.imwrite('blue_mask_debug.png', blue_mask)
        print('Saved blue_mask_debug.png')

    img_vis = roi_bgr.copy()
    if len(img_vis.shape) == 2:
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)

    # hook – green
    for cnt in hook_contours:
        pts = np.array(cnt, dtype=np.int32).reshape(-1, 1, 2)
        if pts.shape[0] < 2:
            continue
        cv2.polylines(img_vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # blue line – blue
    if blue_cnt is not None:
        blue_pts = blue_cnt.reshape(-1, 1, 2)
        cv2.polylines(img_vis, [blue_pts], isClosed=False, color=(255, 0, 0), thickness=3)

    cv2.imwrite('contours_preview.png', img_vis)
    print('Saved contours_preview.png')

    try:
        cv2.imshow('Hook binary', hook_binary)
        if blue_mask is not None:
            cv2.imshow('Blue mask', blue_mask)
        cv2.imshow('Preview', img_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        # in RoboDK GUI may not be available – ignore
        pass


# =============================
# 4. Convert hook contour from px to mm
# =============================

def contours_px_to_mm(contours, mm_per_px=None):
    all_points = np.vstack(contours)
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)

    width_px  = float(x_max - x_min)
    height_px = float(y_max - y_min)
    if width_px <= 0 or height_px <= 0:
        raise RuntimeError('Hook bounding box has zero size.')

    cx_px = (x_min + x_max) / 2.0
    cy_px = (y_min + y_max) / 2.0

    if mm_per_px is None:
        scale_x = FALLBACK_DRAW_WIDTH_MM  / width_px
        scale_y = FALLBACK_DRAW_HEIGHT_MM / height_px
        mm_per_px = min(scale_x, scale_y)
        print(f'[SCALE] Fallback mm_per_px = {mm_per_px:.4f} mm/px')
    else:
        print(f'[SCALE] Using mm_per_px from blue line: {mm_per_px:.4f} mm/px')

    hook_width_mm  = width_px  * mm_per_px
    hook_height_mm = height_px * mm_per_px
    print(f'[OBJ] Approx physical size: {hook_width_mm:.1f} mm x {hook_height_mm:.1f} mm')

    polylines_mm = []
    for cnt in contours:
        pts_mm = []
        for (x_px, y_px) in cnt:
            dx = (x_px - cx_px) * mm_per_px
            dy = (y_px - cy_px) * mm_per_px
            if FLIP_X:
                dx = -dx
            if FLIP_Y:
                dy = -dy
            pts_mm.append([dx, dy])
        polylines_mm.append(pts_mm)

    return polylines_mm


# =============================
# 5. Draw hook with RoboDK
# =============================

def draw_with_robot(polylines_mm):
    RDK = robolink.Robolink()

    robot = RDK.Item(ROBOT_NAME, robolink.ITEM_TYPE_ROBOT)
    if not robot.Valid():
        raise RuntimeError(f'Robot "{ROBOT_NAME}" not found.')

    tool = RDK.Item(TOOL_NAME, robolink.ITEM_TYPE_TOOL)
    if not tool.Valid():
        raise RuntimeError(f'Tool "{TOOL_NAME}" not found.')

    start_target = RDK.Item(START_TARGET_NAME, robolink.ITEM_TYPE_TARGET)
    if not start_target.Valid():
        raise RuntimeError(
            f'Target "{START_TARGET_NAME}" not found. '
            f'Create it at paper center with the pen touching the paper.'
        )

    start_pose = start_target.Pose()
    parent_frame = start_target.Parent()
    if parent_frame.Valid():
        robot.setPoseFrame(parent_frame)
    robot.setPoseTool(tool)

    RDK.Render(False)

    robot.setSpeed(SPEED_MOVE)
    pose_start_safe = start_pose * rdm.transl(0, 0, Z_LIFT)
    robot.MoveJ(pose_start_safe)

    def pose_xyz(x_mm, y_mm, z_mm):
        return start_pose * rdm.transl(x_mm, y_mm, z_mm)

    for poly in polylines_mm:
        if len(poly) < 2:
            continue
        pts = poly
        x0, y0 = pts[0]

        # travel move (pen up)
        robot.setSpeed(SPEED_MOVE)
        robot.MoveL(pose_xyz(x0, y0, Z_LIFT))

        # drawing (pen down)
        robot.setSpeed(SPEED_DRAW)
        robot.MoveL(pose_xyz(x0, y0, Z_DRAW))
        for (x, y) in pts[1:]:
            robot.MoveL(pose_xyz(x, y, Z_DRAW))

        # lift pen
        robot.setSpeed(SPEED_MOVE)
        robot.MoveL(pose_xyz(pts[-1][0], pts[-1][1], Z_LIFT))

    robot.MoveL(pose_start_safe)
    RDK.Render(True)


# =============================
# MAIN
# =============================

def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise RuntimeError(f'Cannot load image from path: {IMAGE_PATH}')

    # 0) crop to paper only
    roi, paper_rect = detect_paper_roi(img)

    # 1) blue line -> scale
    mm_per_px, blue_cnt, blue_mask = detect_blue_line_scale(roi)

    # 2) hook contour on paper
    hook_contours_px, hook_binary = find_hook_contour(roi)

    # 3) preview images (for debugging)
    preview_all(roi, hook_contours_px, hook_binary, blue_cnt, blue_mask)

    # 4) convert hook contour to millimetres
    polylines_mm = contours_px_to_mm(hook_contours_px, mm_per_px=mm_per_px)

    # 5) draw ONLY the hook with correct scale
    draw_with_robot(polylines_mm)


if __name__ == '__main__':
    main()
