"""
Merged API — Face Liveness Detection + Exam Registration
=========================================================
Liveness Detection (connected to UI):
  GET  /                        → serves livelinessui.html
  GET  /health                  → model status
  POST /detect                  → single frame analysis (multipart)
  POST /detect/base64           → single frame analysis (JSON)
  POST /session/start           → start liveness session
  GET  /session/state           → get session state
  POST /session/update          → feed frame (multipart)
  POST /session/update/base64   → feed frame (JSON)

Exam Registration:
  POST /form                    → save candidate form data
  POST /check-photo             → validate candidate photo
  POST /upload                  → upload & OCR documents
  GET  /verify                  → verify all documents against form
"""

import os
import cv2
import time
import base64
import random
import urllib.request
import numpy as np
import requests
import mimetypes
import json
import re
import io
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from scipy.spatial import distance as dist
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════
#  ENV
# ═══════════════════════════════════════════════════════════════════════

OCR_API_URL = os.getenv("OCR_API_URL")
MODEL_URL   = os.getenv("MODEL_URL")
MODEL_NAME  = os.getenv("MODEL_NAME")

# ═══════════════════════════════════════════════════════════════════════
#  LIVENESS CONFIG
# ═══════════════════════════════════════════════════════════════════════

WEIGHTS_PATH    = os.getenv("WEIGHTS_PATH", "best.pt")
LANDMARKER_PATH = os.getenv("LANDMARKER_PATH", "face_landmarker.task")
LANDMARKER_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
NOSE_TIP  = 1

EAR_THRESH         = 0.22
EAR_CONSEC         = 1
HEAD_THRESH_H      = 0.030  # horizontal (left/right)
HEAD_THRESH_V      = 0.020  # vertical (up/down) — smaller threshold, less natural range
HEAD_HOLD_DURATION = 1.0    # seconds user must hold position

NOSE_HIST_LEN    = 30
BASELINE_SETTLE  = 10
BASELINE_REFRESH = 30

CHALLENGES_NEEDED = 3

CHALLENGE_POOL = [
    {"text": "Blink TWICE",       "type": "blink", "count": 2,     "timeout": 8},
    {"text": "Turn head LEFT",    "type": "head",  "dir": "left",  "timeout": 10},
    {"text": "Turn head RIGHT",   "type": "head",  "dir": "right", "timeout": 10},
    {"text": "Nod UP",            "type": "head",  "dir": "up",    "timeout": 10},
    {"text": "Nod DOWN",          "type": "head",  "dir": "down",  "timeout": 10},
    {"text": "Blink THREE times", "type": "blink", "count": 3,     "timeout": 10},
]

# ═══════════════════════════════════════════════════════════════════════
#  EXAM REGISTRATION CONFIG
# ═══════════════════════════════════════════════════════════════════════

REJECT_CLASSES = ["Caps-Hats", "Sunglasses", "Smile"]

BOARD_JSON = {
    "CBSE":     ["CENTRAL BOARD OF SECONDARY EDUCATION", "CBSE"],
    "ICSE":     ["COUNCIL FOR THE INDIAN SCHOOL CERTIFICATE EXAMINATIONS", "ICSE", "ISC"],
    "NIOS":     ["NATIONAL INSTITUTE OF OPEN SCHOOLING", "NIOS"],
    "IB":       ["INTERNATIONAL BACCALAUREATE", "IB"],
    "CAMBRIDGE":["CAMBRIDGE INTERNATIONAL", "IGCSE", "CAIE"],
    "GSEB":     ["GUJARAT SECONDARY AND HIGHER SECONDARY EDUCATION BOARD", "GSEB"],
    "BSEB":     ["BIHAR SCHOOL EXAMINATION BOARD", "BSEB"],
    "UP":       ["UTTAR PRADESH MADHYAMIK SHIKSHA PARISHAD", "UP BOARD"],
    "RBSE":     ["RAJASTHAN BOARD OF SECONDARY EDUCATION", "RBSE"],
    "MPBSE":    ["MADHYA PRADESH BOARD OF SECONDARY EDUCATION", "MPBSE"],
    "MSBSHSE":  ["MAHARASHTRA STATE BOARD OF SECONDARY AND HIGHER SECONDARY EDUCATION", "MSBSHSE"],
    "WBBSE":    ["WEST BENGAL BOARD OF SECONDARY EDUCATION", "WBBSE"],
    "PSEB":     ["PUNJAB SCHOOL EDUCATION BOARD", "PSEB"],
    "HBSE":     ["HARYANA BOARD OF SCHOOL EDUCATION", "HBSE"],
    "HPBOSE":   ["HIMACHAL PRADESH BOARD OF SCHOOL EDUCATION", "HPBOSE"],
    "JKBOSE":   ["JAMMU AND KASHMIR BOARD OF SCHOOL EDUCATION", "JKBOSE"],
    "UBSE":     ["UTTARAKHAND BOARD OF SCHOOL EDUCATION", "UBSE"],
    "CGBSE":    ["CHHATTISGARH BOARD OF SECONDARY EDUCATION", "CGBSE"],
    "JAC":      ["JHARKHAND ACADEMIC COUNCIL", "JAC"],
    "BSEODISHA":["BOARD OF SECONDARY EDUCATION ODISHA", "BSE ODISHA"],
    "SEBA":     ["BOARD OF SECONDARY EDUCATION ASSAM", "SEBA"],
    "TBSE":     ["TRIPURA BOARD OF SECONDARY EDUCATION", "TBSE"],
    "MBSE":     ["MIZORAM BOARD OF SCHOOL EDUCATION", "MBSE"],
    "NBSE":     ["NAGALAND BOARD OF SCHOOL EDUCATION", "NBSE"],
    "BOSEM":    ["BOARD OF SECONDARY EDUCATION MANIPUR", "BOSEM"],
    "MBOSE":    ["MEGHALAYA BOARD OF SCHOOL EDUCATION", "MBOSE"],
    "GOA":      ["GOA BOARD OF SECONDARY AND HIGHER SECONDARY EDUCATION", "GBSHSE"],
    "KSEAB":    ["KARNATAKA SCHOOL EXAMINATION AND ASSESSMENT BOARD", "KSEAB"],
    "TNBSE":    ["TAMIL NADU BOARD OF SECONDARY EDUCATION", "TNBSE"],
    "KBPE":     ["KERALA BOARD OF PUBLIC EXAMINATIONS", "KBPE"],
    "BSEAP":    ["BOARD OF SECONDARY EDUCATION ANDHRA PRADESH", "BSEAP"],
    "BSETS":    ["BOARD OF SECONDARY EDUCATION TELANGANA", "BSETS"],
    "DELHI":    ["DELHI BOARD OF SCHOOL EDUCATION", "DBSE"],
    "SBOSE":    ["SIKKIM BOARD OF SECONDARY EDUCATION", "SBOSE"],
}

# ═══════════════════════════════════════════════════════════════════════
#  GLOBAL MODEL HANDLES
# ═══════════════════════════════════════════════════════════════════════

# Liveness models
MP_OK        = False
face_mesh    = None
YOLO_OK      = False
yolo_model   = None
USING_CUSTOM = False
haar         = None
mp_module    = None

# Exam registration models
face_model   = None
verify_model = None


def _load_models():
    global MP_OK, face_mesh, YOLO_OK, yolo_model, USING_CUSTOM, haar, mp_module
    global face_model, verify_model

    # ── MediaPipe (liveness) ─────────────────────────────────────────
    try:
        import mediapipe as mp
        mp_module = mp

        if not os.path.exists(LANDMARKER_PATH):
            print(f"[INFO] Downloading face landmarker → {LANDMARKER_PATH}")
            urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_PATH)

        BaseOptions           = mp.tasks.BaseOptions
        FaceLandmarker        = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode     = mp.tasks.vision.RunningMode

        opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=LANDMARKER_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        face_mesh = FaceLandmarker.create_from_options(opts)
        MP_OK = True
        print("[INFO] MediaPipe FaceLandmarker ready.")
    except ImportError:
        print("[WARN] mediapipe not installed — pip install mediapipe==0.10.13")
    except Exception as exc:
        print(f"[WARN] MediaPipe failed: {exc}")

    # ── YOLO (liveness) ──────────────────────────────────────────────
    try:
        from ultralytics import YOLO
        if os.path.exists(WEIGHTS_PATH):
            yolo_model   = YOLO(WEIGHTS_PATH)
            USING_CUSTOM = True
            print(f"[INFO] Liveness YOLO loaded: {WEIGHTS_PATH}")
        else:
            yolo_model   = YOLO("yolov8n.pt")
            USING_CUSTOM = False
            print("[WARN] best.pt not found — using yolov8n fallback.")
        YOLO_OK = True
    except Exception as exc:
        print(f"[WARN] Liveness YOLO unavailable ({exc}) — Haar cascade fallback.")

    # ── Haar fallback (liveness) ─────────────────────────────────────
    haar = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # ── YOLO models (exam registration) ─────────────────────────────
    try:
        from ultralytics import YOLO as _YOLO
        face_model   = _YOLO("face_weights.pt")
        verify_model = _YOLO("accessories_weights.pt")
        print("[INFO] Exam registration YOLO models loaded.")
    except Exception as exc:
        print(f"[WARN] Exam registration YOLO models failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════
#  LIFESPAN
# ═══════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield
    if face_mesh:
        face_mesh.close()
    print("[INFO] Models released.")


# ═══════════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Anti-Spoof App API",
    version="2.0.0",
    description="Face Liveness Detection + Exam Registration API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════
#  UI ROUTE — serves livelinessui.html at GET /
#  Place livelinessui.html in the same folder as app.py on your VM.
# ═══════════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
def serve_ui():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "livelinessui.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="livelinessui.html not found next to app.py")
    return FileResponse(html_path, media_type="text/html")


# ═══════════════════════════════════════════════════════════════════════
#  LIVENESS HELPERS
# ═══════════════════════════════════════════════════════════════════════

def eye_aspect_ratio(pts):
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0


def get_eye_pts(landmarks, indices, w, h):
    return np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])


def texture_score(gray_crop):
    if gray_crop is None or gray_crop.size == 0:
        return 999.0
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")
    return img


def decode_base64_image(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return decode_image(base64.b64decode(b64))


def detect_faces_liveness(frame: np.ndarray):
    if YOLO_OK and yolo_model:
        boxes = []
        cls_filter = None if USING_CUSTOM else [0]
        results = yolo_model(frame, classes=cls_filter, conf=0.45, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                if not USING_CUSTOM:
                    y2 = y1 + int((y2 - y1) * 0.35)
                boxes.append((x1, y1, x2, y2, conf))
        return boxes
    else:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        return [(x, y, x + w, y + h, 1.0) for (x, y, w, h) in faces]


def run_mediapipe(frame: np.ndarray):
    if not (MP_OK and face_mesh and mp_module):
        return None
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb)
    res = face_mesh.detect(mp_image)
    if res.face_landmarks:
        return res.face_landmarks[0]
    return None


def analyse_frame(frame: np.ndarray) -> dict:
    h, w  = frame.shape[:2]
    faces = detect_faces_liveness(frame)
    lm    = run_mediapipe(frame)

    face_list = [
        {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "conf": round(b[4], 3)}
        for b in faces
    ]

    ear, nose_pos, tex = 0.0, None, 999.0

    if lm:
        lp  = get_eye_pts(lm, LEFT_EYE,  w, h)
        rp  = get_eye_pts(lm, RIGHT_EYE, w, h)
        ear = round((eye_aspect_ratio(lp) + eye_aspect_ratio(rp)) / 2.0, 4)
        n   = lm[NOSE_TIP]
        nose_pos = {"x": round(n.x, 4), "y": round(n.y, 4)}

    if faces:
        x1, y1, x2, y2 = faces[0][:4]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY) \
               if (y2 > y1 and x2 > x1) else None
        tex  = round(texture_score(crop), 2)

    return {
        "face_count":    len(faces),
        "faces":         face_list,
        "landmarks_ok":  lm is not None,
        "ear":           ear,
        "nose_position": nose_pos,
        "texture_score": tex,
    }


# ═══════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════

class SessionState:
    def __init__(self):
        self.reset()

    def reset(self):
        pool = CHALLENGE_POOL.copy()
        random.shuffle(pool)
        self.challenges: list            = pool[:CHALLENGES_NEEDED]
        self.idx:        int             = 0
        self.passes:     int             = 0
        self.fails:      int             = 0
        self.verdict:    str             = "CHECKING"
        self.final_done: bool            = False
        self.result_at:  Optional[float] = None
        self.photo_check: Optional[dict] = None
        self._load_current()

    def _load_current(self):
        self.ch_start        = time.time()
        self.ch_done         = False
        self.blinks          = 0
        self.consec          = 0
        self.frame_count     = 0
        self.nose_hist       = deque(maxlen=NOSE_HIST_LEN)
        self.head_moves      = dict(left=False, right=False, up=False, down=False)
        self.baseline        = None
        self.head_hold_start = dict(left=None, right=None, up=None, down=None)

    # photo_check is set once at session end (REAL PERSON verdict only)
    photo_check: Optional[dict] = None

    @property
    def current_challenge(self) -> dict:
        return self.challenges[self.idx]

    def time_left(self) -> float:
        return max(0.0, self.current_challenge["timeout"] - (time.time() - self.ch_start))

    def public_state(self) -> dict:
        ch   = self.current_challenge
        dir_ = ch.get("dir", "") if not self.final_done else ""

        hold_start    = self.head_hold_start.get(dir_)
        hold_progress = 0.0
        if hold_start is not None and not self.head_moves.get(dir_, False):
            hold_progress = min(1.0, (time.time() - hold_start) / HEAD_HOLD_DURATION)

        return {
            "verdict":           self.verdict,
            "final_done":        self.final_done,
            "passes":            self.passes,
            "fails":             self.fails,
            "challenges_needed": CHALLENGES_NEEDED,
            "current_challenge": {
                "index":         self.idx + 1,
                "total":         CHALLENGES_NEEDED,
                "text":          ch["text"],
                "type":          ch["type"],
                "dir":           ch.get("dir", ""),
                "time_left":     round(self.time_left(), 2),
                "hold_progress": round(hold_progress, 3),
            } if not self.final_done else None,
            "blinks":      self.blinks,
            "photo_check": self.photo_check,  # None until liveness passes
        }


_session = SessionState()


# ═══════════════════════════════════════════════════════════════════════
#  SCHEMAS
# ═══════════════════════════════════════════════════════════════════════

class Base64Frame(BaseModel):
    image: str


# ═══════════════════════════════════════════════════════════════════════
#  LIVENESS ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status":         "ok",
        "mediapipe":      MP_OK,
        "yolo":           YOLO_OK,
        "custom_weights": USING_CUSTOM,
    }


@app.post("/detect", summary="Single-frame analysis (multipart or form)")
async def detect(file:  Optional[UploadFile] = File(None),
                 image: Optional[str]        = Form(None)):
    if file:
        frame = decode_image(await file.read())
    elif image:
        try:
            frame = decode_base64_image(image)
        except Exception:
            raise HTTPException(400, "Invalid base64 image.")
    else:
        raise HTTPException(400, "Provide file or base64 image field.")
    return JSONResponse(analyse_frame(frame))


@app.post("/detect/base64", summary="Single-frame analysis (JSON body)")
async def detect_base64(body: Base64Frame):
    try:
        frame = decode_base64_image(body.image)
    except Exception:
        raise HTTPException(400, "Invalid base64 image.")
    return JSONResponse(analyse_frame(frame))


@app.post("/session/start")
def session_start():
    _session.reset()
    return _session.public_state()


@app.get("/session/state")
def session_state():
    return _session.public_state()


@app.post("/session/capture", summary="Capture photo after liveness passes")
async def session_capture(body: Base64Frame):
    """Called when user clicks Capture button after passing liveness.
    Runs photo check on the submitted frame and stores result in session."""
    if not _session.final_done or _session.verdict != "REAL PERSON":
        raise HTTPException(400, "Liveness not passed yet")
    try:
        frame = decode_base64_image(body.image)
    except Exception:
        raise HTTPException(400, "Invalid base64 image.")
    _session.photo_check = _run_photo_check(frame)
    print(f"[PHOTO CHECK] {_session.photo_check}")
    return _session.photo_check


def _run_photo_check(frame: np.ndarray) -> dict:
    """Run exam-registration photo check on a raw BGR numpy frame.
    Reuses face_model + verify_model directly — no file upload needed."""
    try:
        if face_model is None or verify_model is None:
            return {"error": "Photo check models not loaded"}

        # Convert BGR (cv2) → RGB (PIL) for YOLO
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        face_results = face_model.predict(np.array(image), conf=0.4)
        face_count   = sum(len(r.boxes) for r in face_results if r.boxes is not None)

        if face_count == 0:
            return {"face_detected": 0, "face_count": 0,
                    "accessories_detected": [], "decision": 0,
                    "reason": "No face detected"}
        if face_count > 1:
            return {"face_detected": 1, "face_count": face_count,
                    "accessories_detected": [], "decision": 0,
                    "reason": "Multiple faces detected"}

        results = verify_model.predict(image, imgsz=640, conf=0.25)
        labels  = []
        for r in results:
            for cls, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                if conf > 0.4:
                    labels.append(verify_model.names[int(cls)])
        labels   = list(set(labels))
        decision = 0 if any(x in labels for x in REJECT_CLASSES) else 1

        return {"face_detected": 1, "face_count": 1,
                "accessories_detected": labels, "decision": decision}

    except Exception as e:
        return {"error": str(e)}


def _process_session_frame(frame: np.ndarray) -> dict:
    h, w  = frame.shape[:2]
    faces = detect_faces_liveness(frame)
    lm    = run_mediapipe(frame)
    sess  = _session

    if sess.final_done:
        if sess.result_at and (time.time() - sess.result_at > 4.0):
            sess.reset()
        return {**sess.public_state(), "metrics": analyse_frame(frame), "faces": [
            {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "conf": round(b[4], 3)}
            for b in faces
        ]}

    ear = 0.0
    sess.frame_count += 1

    if lm:
        lp  = get_eye_pts(lm, LEFT_EYE,  w, h)
        rp  = get_eye_pts(lm, RIGHT_EYE, w, h)
        ear = (eye_aspect_ratio(lp) + eye_aspect_ratio(rp)) / 2.0

        if ear < EAR_THRESH:
            sess.consec += 1
        else:
            if sess.consec >= EAR_CONSEC:
                sess.blinks += 1
                print(f"[BLINK] count={sess.blinks}  EAR={ear:.3f}")
            sess.consec = 0

        n = lm[NOSE_TIP]
        sess.nose_hist.append(np.array([n.x, n.y]))

        if len(sess.nose_hist) >= BASELINE_SETTLE:
            if sess.baseline is None:
                pts = np.array(list(sess.nose_hist)[:BASELINE_SETTLE])
                sess.baseline = np.median(pts, axis=0)
                print(f"[BASELINE] set at {sess.baseline}")
            elif sess.frame_count % BASELINE_REFRESH == 0:
                if not any(sess.head_moves.values()):
                    pts = np.array(list(sess.nose_hist)[-BASELINE_SETTLE:])
                    sess.baseline = np.median(pts, axis=0)
                    print(f"[BASELINE] refreshed at {sess.baseline}")

            if sess.baseline is not None:
                hist_arr  = np.array(list(sess.nose_hist))
                deltas    = hist_arr - sess.baseline
                max_left  = float(np.min(deltas[:, 0]))
                max_right = float(np.max(deltas[:, 0]))
                max_up    = float(np.min(deltas[:, 1]))
                max_down  = float(np.max(deltas[:, 1]))
                now = time.time()

                def _update_hold(dir_key: str, exceeded: bool):
                    if sess.head_moves[dir_key]:
                        return
                    if exceeded:
                        if sess.head_hold_start[dir_key] is None:
                            sess.head_hold_start[dir_key] = now
                            print(f"[HEAD] {dir_key} threshold crossed — hold timer started")
                        else:
                            held = now - sess.head_hold_start[dir_key]
                            if held >= HEAD_HOLD_DURATION:
                                sess.head_moves[dir_key] = True
                                print(f"[HEAD] {dir_key} CONFIRMED after {held:.2f}s hold")
                    else:
                        if sess.head_hold_start[dir_key] is not None:
                            print(f"[HEAD] {dir_key} hold reset (looked away)")
                            sess.head_hold_start[dir_key] = None

                # Raw frame directions (no server-side swap needed):
                # UI challengeText() already swaps LEFT<->RIGHT in display text.
                # So server "left" challenge → UI shows "Turn RIGHT" → user turns right
                # → nose goes LEFT in raw frame → max_left < -threshold → triggers "left" ✓
                # Server "right" challenge → UI shows "Turn LEFT" → user turns left
                # → nose goes RIGHT in raw frame → max_right > threshold → triggers "right" ✓
                # Debug every 10 frames so you can see live nose displacement
                if sess.frame_count % 10 == 0:
                    print(f"[NOSE] Δleft={max_left:.4f} Δright={max_right:.4f} "
                          f"Δup={max_up:.4f} Δdown={max_down:.4f} "
                          f"thresh_h={HEAD_THRESH_H} thresh_v={HEAD_THRESH_V}")

                _update_hold("left",  max_left  < -HEAD_THRESH_H)
                _update_hold("right", max_right >  HEAD_THRESH_H)
                _update_hold("up",    max_up    < -HEAD_THRESH_V)
                _update_hold("down",  max_down  >  HEAD_THRESH_V)

    ch        = sess.current_challenge
    time_left = sess.time_left()

    if not sess.ch_done:
        passed = False
        if ch["type"] == "blink" and sess.blinks >= ch.get("count", 1):
            passed = True
        elif ch["type"] == "head" and sess.head_moves.get(ch.get("dir", ""), False):
            passed = True

        if passed:
            sess.ch_done = True
            sess.passes += 1
            print(f"[PASS {sess.passes}/{CHALLENGES_NEEDED}] {ch['text']}")
        elif time_left <= 0:
            sess.ch_done = True
            sess.fails  += 1
            print(f"[FAIL] timed out: {ch['text']}")

    if sess.ch_done:
        next_idx = sess.idx + 1
        if next_idx < CHALLENGES_NEEDED:
            sess.idx += 1
            sess._load_current()
        else:
            sess.final_done = True
            sess.result_at  = time.time()
            sess.verdict    = (
                "REAL PERSON" if sess.passes == CHALLENGES_NEEDED else "FAKE / SPOOF"
            )
            print(f"[RESULT] {sess.verdict} — {sess.passes}/{CHALLENGES_NEEDED}")

            # Do NOT auto-capture — wait for user to press Capture button
            # photo_check stays None until /session/capture is called
            if sess.verdict == "REAL PERSON":
                sess.photo_check = {"pending": True}
                print("[PHOTO CHECK] waiting for user capture")
            else:
                sess.photo_check = None

    face_list = [
        {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "conf": round(b[4], 3)}
        for b in faces
    ]

    metrics = {
        "face_count":   len(faces),
        "landmarks_ok": lm is not None,
        "ear":          round(ear, 4),
        "time_left":    round(time_left, 2),
    }
    # Call public_state() AFTER photo_check is assigned so it's included in response
    return {**sess.public_state(), "metrics": metrics, "faces": face_list}


@app.post("/session/update", summary="Feed frame (multipart or form)")
async def session_update(file:  Optional[UploadFile] = File(None),
                         image: Optional[str]        = Form(None)):
    if file:
        frame = decode_image(await file.read())
    elif image:
        try:
            frame = decode_base64_image(image)
        except Exception:
            raise HTTPException(400, "Invalid base64 image.")
    else:
        raise HTTPException(400, "Provide file or base64 image field.")
    return JSONResponse(_process_session_frame(frame))


@app.post("/session/update/base64", summary="Feed frame (JSON body)")
async def session_update_base64(body: Base64Frame):
    try:
        frame = decode_base64_image(body.image)
    except Exception:
        raise HTTPException(400, "Invalid base64 image.")
    return JSONResponse(_process_session_frame(frame))


# ═══════════════════════════════════════════════════════════════════════
#  EXAM REGISTRATION HELPERS
# ═══════════════════════════════════════════════════════════════════════

def run_ocr(file_bytes, filename):
    mime  = mimetypes.guess_type(filename)[0]
    files = {"file": (filename, file_bytes, mime)}
    response = requests.post(OCR_API_URL, files=files)
    return response.json().get("plain_text", "")


def extract_json(text):
    match = re.search(r'\{[\s\S]*\}', text)
    return match.group() if match else "{}"


def get_prompt(section, doc, text):
    if section == "identity":
        return f"Extract:\ndocument_number\nname\ndob\n\nReturn JSON only\n\n{text}"
    if section == "education" and doc == "class_10":
        return f"Extract:\nname\ndob\nboard\nresult\n\nReturn JSON only\n\n{text}"
    if section == "education" and doc == "graduation_degree":
        return f"Extract:\nname\nresult\n\nReturn JSON only\n\n{text}"
    if section == "certificate" and doc == "caste":
        return f"Extract:\nname\n\nReturn JSON only\n\n{text}"
    if section == "certificate" and doc in ["ews", "obc_ncl"]:
        return f"Extract:\nname\nvalidity\n\nReturn JSON only\n\n{text}"


def extract_with_model(section, doc, text):
    prompt   = get_prompt(section, doc, text)
    response = requests.post(
        MODEL_URL,
        headers={"accept": "application/json", "Content-Type": "application/json"},
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "top_p": 0.1,
        },
    )
    output = response.json()["choices"][0]["message"]["content"]
    return json.loads(extract_json(output))


def normalize(text):
    if not text:
        return None
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return " ".join(text.split())


def normalize_dob(dob):
    for fmt in ["%d/%m/%Y", "%d-%m-%Y"]:
        try:
            return datetime.strptime(dob, fmt).date()
        except:
            continue
    return None


def validate_result(result):
    if not result:
        return None
    fail_keywords = ["fail", "atkt", "compartment", "backlog", "essential repeat", "er"]
    for keyword in fail_keywords:
        if keyword in result.lower():
            return {"result": result, "allowed": 0, "reason": "Candidate Failed / Backlog Present"}
    return {"result": result, "allowed": 1}


def parse_validity(validity):
    if not validity:
        return None
    validity = validity.lower()
    validity = re.sub(r'valid|upto|up to|till|period|from', '', validity)
    match = re.search(r'(\d{4})\s*-\s*(\d{2,4})', validity)
    if match:
        end = int(match.group(2))
        return datetime(end, 3, 31)
    for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d-%b-%Y"]:
        try:
            return datetime.strptime(validity.strip(), fmt)
        except:
            continue
    return None


# ═══════════════════════════════════════════════════════════════════════
#  EXAM REGISTRATION STATE
# ═══════════════════════════════════════════════════════════════════════

FORM_DATA = {}
DOCUMENTS = {}


# ═══════════════════════════════════════════════════════════════════════
#  EXAM REGISTRATION ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.post("/form")
async def form(
    name:  str            = Form(...),
    dob:   str            = Form(...),
    email: Optional[str]  = Form(None),
):
    global FORM_DATA
    FORM_DATA = {"name": name, "dob": dob, "email": email}
    return {"message": "Form Saved"}


@app.post("/check-photo")
async def check_photo(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np   = np.array(image)

        face_results = face_model.predict(img_np, conf=0.4)
        face_count   = sum(len(r.boxes) for r in face_results if r.boxes is not None)

        if face_count == 0:
            return {"face_detected": 0, "face_count": 0, "accessories_detected": [],
                    "decision": 0, "reason": "No face detected"}
        if face_count > 1:
            return {"face_detected": 1, "face_count": face_count, "accessories_detected": [],
                    "decision": 0, "reason": "Multiple faces detected"}

        results = verify_model.predict(image, imgsz=640, conf=0.25)
        labels  = []
        for r in results:
            for cls, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                if conf > 0.4:
                    labels.append(verify_model.names[int(cls)])
        labels = list(set(labels))

        decision = 0 if any(x in labels for x in REJECT_CLASSES) else 1
        return {"face_detected": 1, "face_count": 1,
                "accessories_detected": labels, "decision": decision}

    except Exception as e:
        return {"error": str(e)}


async def _process_document(file, section, doc):
    content   = await file.read()
    text      = run_ocr(content, file.filename)
    extracted = extract_with_model(section, doc, text)
    return {"filename": file.filename, "raw_text": text, "extracted_fields": extracted}


@app.post("/upload")
async def upload(
    aadhaar_card:       UploadFile = File(None),
    pan_card:           UploadFile = File(None),
    driving_license:    UploadFile = File(None),
    class_10:           UploadFile = File(None),
    graduation_degree:  UploadFile = File(None),
    caste:              UploadFile = File(None),
    ews:                UploadFile = File(None),
    obc_ncl:            UploadFile = File(None),
):
    global DOCUMENTS
    DOCUMENTS = {"identity": {}, "education": {}, "certificate": {}}

    if aadhaar_card:
        DOCUMENTS["identity"]["aadhaar_card"]      = await _process_document(aadhaar_card,      "identity",    "aadhaar_card")
    if pan_card:
        DOCUMENTS["identity"]["pan_card"]          = await _process_document(pan_card,          "identity",    "pan_card")
    if driving_license:
        DOCUMENTS["identity"]["driving_license"]   = await _process_document(driving_license,   "identity",    "driving_license")
    if class_10:
        DOCUMENTS["education"]["class_10"]         = await _process_document(class_10,          "education",   "class_10")
    if graduation_degree:
        DOCUMENTS["education"]["graduation_degree"]= await _process_document(graduation_degree, "education",   "graduation_degree")
    if caste:
        DOCUMENTS["certificate"]["caste"]          = await _process_document(caste,             "certificate", "caste")
    if ews:
        DOCUMENTS["certificate"]["ews"]            = await _process_document(ews,               "certificate", "ews")
    if obc_ncl:
        DOCUMENTS["certificate"]["obc_ncl"]        = await _process_document(obc_ncl,           "certificate", "obc_ncl")

    return DOCUMENTS


@app.get("/verify")
def verify():
    if not FORM_DATA:
        raise HTTPException(400, "Submit form first")

    form_name = normalize(FORM_DATA["name"])
    form_dob  = normalize_dob(FORM_DATA["dob"])

    name_validation = []
    dob_validation  = []
    all_names       = []

    for section in DOCUMENTS:
        for doc in DOCUMENTS[section]:
            fields = DOCUMENTS[section][doc]["extracted_fields"]
            name   = fields.get("name")
            dob    = fields.get("dob")

            if name:
                normalized = normalize(name)
                name_validation.append({
                    "document": doc,
                    "name":     name,
                    "match":    int(normalized == form_name),
                })
                all_names.append(normalized)

            if dob:
                dob_validation.append({
                    "document": doc,
                    "dob":      dob,
                    "match":    int(normalize_dob(dob) == form_dob),
                })

    global_name_consistency = int(len(set(all_names)) == 1)

    # Board validation
    board = board_standard = None
    class10 = DOCUMENTS["education"].get("class_10")
    if class10:
        board = class10["extracted_fields"].get("board")
        if board:
            board_upper = board.upper()
            for key in BOARD_JSON:
                for value in BOARD_JSON[key]:
                    if value in board_upper:
                        board_standard = key
                        break

    # Result validation
    result_validation = {}
    if class10:
        result_validation["class_10"] = validate_result(
            class10["extracted_fields"].get("result")
        )
    graduation = DOCUMENTS["education"].get("graduation_degree")
    if graduation:
        result_validation["graduation"] = validate_result(
            graduation["extracted_fields"].get("result")
        )

    # Certificate validation
    certificate_validation = {}
    for cert in ["ews", "obc_ncl"]:
        cert_data = DOCUMENTS["certificate"].get(cert)
        if not cert_data:
            continue
        validity = cert_data["extracted_fields"].get("validity")
        expiry   = parse_validity(validity)
        certificate_validation[cert] = {
            "validity":    validity,
            "expiry_date": expiry.strftime("%d/%m/%Y") if expiry else None,
            "expired":     int(datetime.now() > expiry) if expiry else None,
        }

    return {
        "form":                     FORM_DATA,
        "name_validation":          name_validation,
        "global_name_consistency":  global_name_consistency,
        "dob_validation":           dob_validation,
        "board_validation":         {"board_found": board, "recognized_board": board_standard},
        "result_validation":        result_validation,
        "certificate_validation":   certificate_validation,
    }