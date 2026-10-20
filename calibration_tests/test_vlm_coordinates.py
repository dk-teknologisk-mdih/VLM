"""Round-trip test for VLM point coordinates: known pixels -> VLM -> [y, x] 0-1000 -> pixels.

By default uses the whiteboard photo Test_frame_fake.jpg (hand-measured vertex
pixels in PHOTO_GT); synthetic images are available via --variants synth_*. Sends it
through the real detector (`GeminiRoboticsDetector.detect_objects`) with the real
path-inspection prompt, un-normalizes the answer exactly like `tasks/common.py`,
and compares against ground truth. Also scores alternative interpretations of the
raw numbers (swapped axes, wrong denominator, letterboxing, raw pixels, ...) to
point at the most likely cause of a systematic error.

    python calibration_tests/test_vlm_coordinates.py --dry-run
    python calibration_tests/test_vlm_coordinates.py --trials 5
    python calibration_tests/test_vlm_coordinates.py --variants photo_pil synth_1280x720
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vlm_programming_orchestrator.tasks.common import load_prompt_template  # noqa: E402
from vlm_programming_orchestrator.vision.registry import get_detector_backend  # noqa: E402

PROMPT_FILE = os.path.join(ROOT, "vlm_programming_orchestrator", "tasks",
                           "path_inspection", "prompts", "vlm_detect_path.txt")
SYNTH_TARGET = "dark line path; place exactly one waypoint on the center of each red circular marker"
PHOTO_TARGET = ("bike frame hand-drawn in red on the whiteboard; place exactly one waypoint "
                "on the center of each black vertex marker")
PHOTO_PATH = os.path.join(ROOT, "Test_frame_fake.jpg")
# Hand-measured (u, v) pixel centers of the black vertex markers in PHOTO_PATH (1920x1080).
PHOTO_GT = [(595, 300), (676, 67), (1167, 375), (1425, 67)]
OUT_DIR = os.path.join(ROOT, "calibration_tests", "vlm_coord_results")

# Asymmetric (u, v) fractions so swaps / scale errors cannot cancel out.
GT_FRACTIONS = [(0.12, 0.20), (0.35, 0.12), (0.62, 0.30), (0.88, 0.18),
                (0.80, 0.66), (0.55, 0.85), (0.22, 0.72)]

# via_path=True goes through detector.resize_image (800 px wide); PIL input is sent at full size.
VARIANTS = {
    "photo_pil": {"photo": True, "via_path": False},
    "photo_path": {"photo": True, "via_path": True},
    "synth_1280x720": {"size": (1280, 720), "via_path": False},
    "synth_1000x1000": {"size": (1000, 1000), "via_path": False},
    "synth_path_1280x720": {"size": (1280, 720), "via_path": True},
}
DEFAULT_VARIANTS = ["photo_pil", "photo_path"]


def make_test_image(width, height):
    img = Image.new("RGB", (width, height), (235, 235, 235))
    draw = ImageDraw.Draw(img)
    gt = np.array([(fu * width, fv * height) for fu, fv in GT_FRACTIONS])
    line_w = max(4, int(0.012 * min(width, height)))
    radius = max(6, int(0.02 * min(width, height)))
    draw.line([tuple(p) for p in gt], fill=(40, 40, 40), width=line_w, joint="curve")
    for u, v in gt:
        draw.ellipse((u - radius, v - radius, u + radius, v + radius),
                     fill=(220, 30, 30), outline=(0, 0, 0), width=2)
    return img, gt


def load_photo():
    img = Image.open(PHOTO_PATH).convert("RGB")
    if img.size != (1920, 1080):
        print(f"WARNING: {PHOTO_PATH} is {img.size}, PHOTO_GT assumes 1920x1080")
    return img, np.array(PHOTO_GT, dtype=float)


def parse_raw_points(response):
    raw = []
    if not isinstance(response, list):
        return raw
    for item in response:
        p = item.get("point") if isinstance(item, dict) else None
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                raw.append((float(p[0]), float(p[1])))
            except (TypeError, ValueError):
                continue
    return raw


def pipeline_unnormalize(raw, width, height):
    """Same math as tasks/common.py::add_depth_to_detections (int truncation)."""
    return np.array([(int(b / 1000.0 * width), int(a / 1000.0 * height)) for a, b in raw], dtype=float)


def hypotheses(width, height, sent_w, sent_h):
    """Ways the raw (a, b) pair could map to original-image pixels (u, v)."""
    s = max(width, height)
    return {
        "pipeline [y,x]/1000 (float)": lambda a, b: (b / 1000 * width, a / 1000 * height),
        "swapped [x,y]/1000": lambda a, b: (a / 1000 * width, b / 1000 * height),
        "both axes / width": lambda a, b: (b / 1000 * width, a / 1000 * width),
        "both axes / height": lambda a, b: (b / 1000 * height, a / 1000 * height),
        "letterboxed square, centered": lambda a, b: (b / 1000 * s - (s - width) / 2,
                                                      a / 1000 * s - (s - height) / 2),
        "normalized 0-1024": lambda a, b: (b / 1024 * width, a / 1024 * height),
        "raw pixels [y,x] of sent image": lambda a, b: (b * width / sent_w, a * height / sent_h),
        "raw pixels [x,y] of sent image": lambda a, b: (a * width / sent_w, b * height / sent_h),
    }


def match(gt, pred, max_dist):
    """Greedy nearest-neighbour matching. Returns list of (gt_idx, pred_idx)."""
    if len(pred) == 0:
        return []
    d = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=2)
    pairs = []
    while np.isfinite(d).any():
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] > max_dist:
            break
        pairs.append((int(i), int(j)))
        d[i, :] = np.inf
        d[:, j] = np.inf
    return pairs


def draw_overlay(img, gt, trials_pred, path):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for u, v in gt:
        draw.ellipse((u - 10, v - 10, u + 10, v + 10), outline=(0, 200, 0), width=3)
    for pred, pairs in trials_pred:
        for gi, pj in pairs:
            draw.line([tuple(gt[gi]), tuple(pred[pj])], fill=(0, 0, 255), width=1)
        for u, v in pred:
            draw.line((u - 7, v - 7, u + 7, v + 7), fill=(255, 0, 255), width=2)
            draw.line((u - 7, v + 7, u + 7, v - 7), fill=(255, 0, 255), width=2)
    img.save(path)


def fake_response(gt, width, height):
    return [{"point": [round(v / height * 1000), round(u / width * 1000)], "label": f"m{i}"}
            for i, (u, v) in enumerate(gt)]


def run_variant(detector, model, name, spec, prompts, args):
    if spec.get("photo"):
        img, gt = load_photo()
        prompt = prompts["photo"]
    else:
        img, gt = make_test_image(*spec["size"])
        prompt = prompts["synth"]
    width, height = img.size
    max_dist = 0.15 * np.hypot(width, height)

    tmp_path = None
    if spec["via_path"] and spec.get("photo"):
        sent_w, sent_h = detector.resize_image(PHOTO_PATH).size
        vlm_input = PHOTO_PATH
    elif spec["via_path"]:
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(tmp_path)
        sent_w, sent_h = detector.resize_image(tmp_path).size
        vlm_input = tmp_path
    else:
        sent_w, sent_h = width, height
        vlm_input = img

    hyps = hypotheses(width, height, sent_w, sent_h)
    hyp_errors = {h: [] for h in hyps}
    rows, overlay_data, raw_log = [], [], []
    ok_trials = 0

    try:
        for t in range(args.trials):
            if args.dry_run:
                response = fake_response(gt, width, height)
            else:
                response = detector.detect_objects(vlm_input, prompt, temperature=args.temperature)
            raw = parse_raw_points(response)
            raw_log.append(raw)
            if not raw:
                print(f"  trial {t}: no usable points (response={str(response)[:120]})")
                continue
            ok_trials += 1

            pred = pipeline_unnormalize(raw, width, height)
            pairs = match(gt, pred, max_dist)
            overlay_data.append((pred, pairs))
            order = [gi for gi, _ in sorted(pairs, key=lambda p: p[1])]
            out_of_range = sum(1 for a, b in raw if not (0 <= a <= 1000 and 0 <= b <= 1000))
            print(f"  trial {t}: {len(raw)} pts, matched {len(pairs)}/{len(gt)}, "
                  f"gt order {order}, out-of-range {out_of_range}")

            pred_float = np.array([hyps["pipeline [y,x]/1000 (float)"](a, b) for a, b in raw])
            for gi, pj in pairs:
                du, dv = pred[pj] - gt[gi]
                rows.append({"trial": t, "gt_idx": gi, "gt": gt[gi].tolist(), "pred": pred[pj].tolist(),
                             "du": float(du), "dv": float(dv),
                             "trunc_du": float(pred[pj][0] - pred_float[pj][0]),
                             "trunc_dv": float(pred[pj][1] - pred_float[pj][1])})

            for h, fn in hyps.items():
                hp = np.array([fn(a, b) for a, b in raw])
                hpairs = match(gt, hp, np.inf)
                hyp_errors[h].extend(float(np.linalg.norm(hp[pj] - gt[gi])) for gi, pj in hpairs)
    finally:
        if tmp_path:
            os.remove(tmp_path)

    safe_model = model.replace("/", "_")
    draw_overlay(img, gt, overlay_data, os.path.join(OUT_DIR, f"{safe_model}__{name}.png"))

    summary = {"model": model, "variant": name, "image_size": [width, height],
               "sent_size": [sent_w, sent_h], "trials": args.trials, "ok_trials": ok_trials,
               "expected_points": len(gt), "matched_points": len(rows)}
    if rows:
        du = np.array([r["du"] for r in rows])
        dv = np.array([r["dv"] for r in rows])
        gu = np.array([r["gt"][0] for r in rows])
        gv = np.array([r["gt"][1] for r in rows])
        pu = gu + du
        pv = gv + dv
        fit_u = np.polyfit(gu, pu, 1) if len(set(gu)) > 1 else [np.nan, np.nan]
        fit_v = np.polyfit(gv, pv, 1) if len(set(gv)) > 1 else [np.nan, np.nan]
        per_gt = {}
        for r in rows:
            per_gt.setdefault(r["gt_idx"], []).append((r["du"], r["dv"]))
        summary.update({
            "mean_du": float(du.mean()), "mean_dv": float(dv.mean()),
            "std_du": float(du.std()), "std_dv": float(dv.std()),
            "mean_err_px": float(np.hypot(du, dv).mean()),
            "mean_trunc_du": float(np.mean([r["trunc_du"] for r in rows])),
            "mean_trunc_dv": float(np.mean([r["trunc_dv"] for r in rows])),
            "fit_u": {"slope": float(fit_u[0]), "intercept": float(fit_u[1])},
            "fit_v": {"slope": float(fit_v[0]), "intercept": float(fit_v[1])},
            "per_gt_mean": {int(k): np.mean(v, axis=0).round(1).tolist() for k, v in sorted(per_gt.items())},
        })
    hyp_mean = {h: float(np.mean(e)) for h, e in hyp_errors.items() if e}
    summary["hypotheses_mean_err_px"] = dict(sorted(hyp_mean.items(), key=lambda kv: kv[1]))
    summary["best_hypothesis"] = next(iter(summary["hypotheses_mean_err_px"]), None)
    summary["rows"] = rows
    summary["raw_points"] = raw_log
    return summary


def print_report(summaries):
    print("\n" + "=" * 118)
    print(f"{'model':<32}{'variant':<16}{'ok':>4}{'match':>8}{'du':>8}{'dv':>8}{'sd_u':>7}{'sd_v':>7}"
          f"{'err':>7}{'slope_u':>9}{'icpt_u':>8}{'slope_v':>9}{'icpt_v':>8}")
    print("-" * 118)
    for s in summaries:
        matched = f"{s['matched_points']}/{s['expected_points'] * s['trials']}"
        head = f"{s['model']:<32}{s['variant']:<16}{s['ok_trials']:>4}{matched:>8}"
        if "mean_du" not in s:
            print(head + "   (no matched points)")
            continue
        print(head + f"{s['mean_du']:8.1f}{s['mean_dv']:8.1f}{s['std_du']:7.1f}{s['std_dv']:7.1f}"
              f"{s['mean_err_px']:7.1f}{s['fit_u']['slope']:9.3f}{s['fit_u']['intercept']:8.1f}"
              f"{s['fit_v']['slope']:9.3f}{s['fit_v']['intercept']:8.1f}")

    print("\nInterpretation hypotheses (mean error px, lower = better explains the raw output):")
    for s in summaries:
        print(f"  {s['model']} / {s['variant']}:")
        for h, e in list(s["hypotheses_mean_err_px"].items())[:4]:
            print(f"      {e:8.1f}  {h}")
        if "mean_du" in s:
            print(f"      per-marker mean (du, dv): {s['per_gt_mean']}")
            print(f"      int() truncation bias: du={s['mean_trunc_du']:.2f} dv={s['mean_trunc_dv']:.2f} px")

    print("\nReading the result:")
    print("  - |mean du/dv| >> sd, or slope != 1 -> systematic conversion error (see best hypothesis).")
    print("  - mean ~0 but large sd             -> VLM localisation noise; compare models.")
    print("  - only path_* variant is off        -> resize path in detector.resize_image.")
    print("  - only non-square variants are off  -> aspect ratio / letterbox handling.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=["gemini-robotics-er-2-preview"])
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANTS))
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--base-url", default=None, help="Override the detector's default Gemini proxy URL.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip the VLM; feed back perfectly normalized ground truth to validate the harness.")
    args = parser.parse_args()

    load_dotenv(os.path.join(ROOT, ".env"))
    api_key = os.environ.get("API_KEY", "")
    if not api_key and not args.dry_run:
        sys.exit("ERROR: API_KEY must be set (e.g. via .env), or use --dry-run.")

    os.makedirs(OUT_DIR, exist_ok=True)
    prompts = {"photo": load_prompt_template(PROMPT_FILE, target_object=PHOTO_TARGET),
               "synth": load_prompt_template(PROMPT_FILE, target_object=SYNTH_TARGET)}
    kwargs = {"use_realsense": False}
    if args.base_url:
        kwargs["base_url"] = args.base_url
    detector = get_detector_backend("gemini_realsense", api_key, **kwargs)

    summaries = []
    for model in args.models:
        detector.model_id = model
        for name in args.variants:
            print(f"\n[{model}] {name}")
            summaries.append(run_variant(detector, model, name, VARIANTS[name], prompts, args))

    print_report(summaries)
    out_json = os.path.join(OUT_DIR, "results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nOverlays (green=truth, magenta=VLM, blue=error) and {out_json} written to {OUT_DIR}")


if __name__ == "__main__":
    main()
