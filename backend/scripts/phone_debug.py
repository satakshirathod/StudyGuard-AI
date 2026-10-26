"""Live phone-detection diagnostic (run from the backend folder).

    python scripts/phone_debug.py

Opens the webcam, runs the SAME object detector the app uses and prints every
object it sees (any label, score >= 0.10) so we can tell whether the phone is
being seen at all and with what score. A window shows the boxes; press Q to quit.

IMPORTANT: stop the backend (uvicorn) first - only one program can use the webcam.
Optional: python scripts/phone_debug.py path/to/image.jpg   (test on a photo instead)
"""

import sys
import time
from pathlib import Path

import cv2

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from ai.phone_detection import PhoneObjectDetector  # noqa: E402

THRESHOLD = 0.50  # what the app uses today (SG_PHONE_CONFIDENCE_THRESHOLD)


def draw(frame, dets):
    for d in dets:
        x1, y1, x2, y2 = d.box
        is_phone = "phone" in d.label.lower()
        color = (0, 200, 0) if is_phone else (180, 180, 180)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{d.label} {d.score:.2f}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def main():
    detector = PhoneObjectDetector(str(BACKEND / "models" / "efficientdet_lite0.tflite"))
    if not detector.initialize():
        print("MODEL FAILED TO LOAD:", detector.last_error)
        return

    if len(sys.argv) > 1:  # single image mode
        frame = cv2.imread(sys.argv[1])
        if frame is None:
            print("Could not read image:", sys.argv[1])
            return
        for d in detector.detect(frame):
            print(f"{d.label:15s} score={d.score:.2f}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("CAMERA COULD NOT BE OPENED - stop the backend (Ctrl+C) and close other camera apps.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Running. Hold your phone in front of the camera. Press Q in the video window to quit.\n")
    last_print = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame from camera")
            break
        dets = [d for d in detector.detect(frame) if d.score >= 0.10]
        if time.time() - last_print > 0.5:
            last_print = time.time()
            if not dets:
                print("nothing detected")
            else:
                print(" | ".join(
                    f"{d.label} {d.score:.2f}{'  <-- PHONE (counts)' if 'phone' in d.label.lower() and d.score >= THRESHOLD else ''}"
                    for d in sorted(dets, key=lambda d: -d.score)[:5]
                ))
        cv2.imshow("phone debug (Q to quit)", draw(frame, dets))
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
