"""Does the detector call real objects by their right name, on real captured frames?

fixtures/bottle_misclassified_as_cup.png is a crop of the actual dashboard screenshot that
reported the bug this guards: a dark insulated bottle read as "drinking cup" (0.66) over
"water bottle" (which scored near zero), fed a wrong height hint into the footprint
correction, and produced a 68 cm phantom box (see test_height_hint_guard.py for that half
of the chain). NOTE: this fixture is a screenshot of the ANNOTATED dashboard view, not a
raw camera frame — the yellow detection overlay is baked into the pixels because a clean
frame of that exact scene was not available when this test was written. It still reproduces
the reported misclassification faithfully (confirmed against the live report), but a
clean re-capture would be a better fixture if one is ever taken.

fixtures/phone_low_confidence.png is a clean, unannotated frame from the phone-detection
fix (dashboard_sources.py's own comments reference the same numbers). It is the control:
the bottle-prompt wording changed for this fix, and this confirms that change did not cost
the phone fix from earlier.
"""
import sys

import cv2

import dashboard_sources as ds

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


def top_label(model, img, prompts, imgsz=960):
    alias = {v: k for k, v in prompts.items()}
    labels = list(prompts.values())
    model.set_classes(labels, model.get_text_pe(labels))
    r = model.predict(img, verbose=False, imgsz=imgsz, conf=0.01)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None, 0.0, {}
    rows = sorted(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()), key=lambda x: -x[1])
    scores = {}
    for cls, conf in rows:
        lbl = alias.get(r.names[int(cls)], r.names[int(cls)])
        scores.setdefault(lbl, conf)         # first hit per label is the highest (sorted)
    best_cls, best_conf = rows[0]
    return alias.get(r.names[int(best_cls)], r.names[int(best_cls)]), best_conf, scores


def main():
    from ultralytics import YOLOE

    model = YOLOE(ds.MODEL)

    bottle_img = cv2.imread("fixtures/bottle_misclassified_as_cup.png")
    phone_img = cv2.imread("fixtures/phone_low_confidence.png")
    if bottle_img is None or phone_img is None:
        print("  [FAIL] fixture image missing")
        return 1

    print("--- the real bottle that used to read as 'cup' ---")
    label, conf, scores = top_label(model, bottle_img, ds.PROMPTS)
    check("bottle wins, not cup", label == "bottle", f"top={label} ({conf:.2f})  all={scores}")
    check("bottle beats cup by a real margin, not a coin flip",
          scores.get("bottle", 0) > scores.get("cup", 0) * 1.5,
          f"bottle={scores.get('bottle', 0):.2f} cup={scores.get('cup', 0):.2f}")

    print("\n--- the phone fix from earlier must still hold ---")
    label, conf, scores = top_label(model, phone_img, ds.PROMPTS)
    check("phone still wins on its own frame", label == "phone",
          f"top={label} ({conf:.2f})  all={scores}")
    check("phone still clears the live confidence threshold", conf >= ds.MIN_CONF,
          f"{conf:.2f} >= {ds.MIN_CONF}")

    print("\n" + "=" * 58)
    print(f"PASSED {passed} / {passed + failed}")
    print("CLASSIFICATION OK" if not failed else "REGRESSED")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
