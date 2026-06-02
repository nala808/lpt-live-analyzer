"""
LPT Live Analyzer — Flask + OpenCV
كاميرا مباشرة + تحليل صور LPT في الوقت الفعلي
"""

from flask import Flask, render_template_string, request, jsonify, Response
import cv2
import numpy as np
import base64
import json
import threading
import time
from datetime import datetime

app = Flask(__name__)

# ─── إعدادات الكاميرا ───
camera = None
camera_lock = threading.Lock()
latest_frame = None
frame_lock = threading.Lock()

# ─── معايير القبول ───
CRITERIA = {
    "max_linear_length_mm":    3.0,
    "max_rounded_diameter_mm": 4.0,
    "max_defect_count":        5,
    "min_defect_area_px":      30,
    "pixels_per_mm":           10.0,
}

# ══════════════════════════════════════════════
#  كشف العيوب
# ══════════════════════════════════════════════

def detect_red_dye(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0,70,50]),   np.array([10,255,255]))
    m2 = cv2.inRange(hsv, np.array([155,70,50]),  np.array([180,255,255]))
    mask = cv2.bitwise_or(m1, m2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask

def detect_fluorescent(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([25,80,80]), np.array([90,255,255]))
    _, vt = cv2.threshold(hsv[:,:,2], 140, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, vt)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask

def auto_detect_mode(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return "fluorescent" if np.mean(hsv[:,:,2]) < 60 else "red_dye"

def classify_defect(cnt, ppm):
    area = cv2.contourArea(cnt)
    if area < CRITERIA["min_defect_area_px"]:
        return None
    x, y, w, h = cv2.boundingRect(cnt)
    length_mm = round(max(w, h) / ppm, 2)
    width_mm  = round(min(w, h) / ppm, 2)
    ar = max(w, h) / max(min(w, h), 1)
    dtype = "Linear Crack" if ar > 3 else "Elongated Porosity" if ar > 1.5 else "Round Porosity"
    return {
        "type": dtype,
        "length_mm": length_mm,
        "width_mm": width_mm,
        "area_mm2": round(area / ppm**2, 3),
        "aspect_ratio": round(ar, 1),
        "bbox": [x, y, w, h],
    }

def check_acceptance(defects):
    reasons = []
    ml  = CRITERIA["max_linear_length_mm"]
    mr  = CRITERIA["max_rounded_diameter_mm"]
    mc  = CRITERIA["max_defect_count"]
    for i, d in enumerate(defects):
        if d["type"] == "Linear Crack" and d["length_mm"] > ml:
            reasons.append(f"شق #{i+1}: {d['length_mm']}mm > {ml}mm")
        elif d["type"] != "Linear Crack" and d["length_mm"] > mr:
            reasons.append(f"مسامية #{i+1}: {d['length_mm']}mm > {mr}mm")
    if len(defects) > mc:
        reasons.append(f"عدد العيوب {len(defects)} > الحد {mc}")
    return ("REJECT" if reasons else "ACCEPT"), reasons

def analyze_frame(img, mode):
    ppm = CRITERIA["pixels_per_mm"]
    mask = detect_fluorescent(img) if mode == "fluorescent" else detect_red_dye(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defects = [d for cnt in contours if (d := classify_defect(cnt, ppm))]
    defects.sort(key=lambda x: x["area_mm2"], reverse=True)
    decision, reasons = check_acceptance(defects)
    stats = {
        "total":      len(defects),
        "cracks":     sum(1 for d in defects if d["type"] == "Linear Crack"),
        "porosity":   sum(1 for d in defects if d["type"] != "Linear Crack"),
        "max_length": max((d["length_mm"] for d in defects), default=0),
    }
    return defects, decision, reasons, stats, mask

def draw_annotations(img, defects, decision):
    out = img.copy()
    colors = {
        "Linear Crack":       (0, 0, 220),
        "Round Porosity":     (0, 140, 255),
        "Elongated Porosity": (0, 220, 220),
    }
    for i, d in enumerate(defects):
        x, y, w, h = d["bbox"]
        c = colors.get(d["type"], (200, 200, 0))
        cv2.rectangle(out, (x,y), (x+w,y+h), c, 2)
        label = f"#{i+1} {d['length_mm']}mm"
        cv2.putText(out, label, (x, max(y-6,12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)
    # شريط القرار
    bar = (0,40,180) if decision == "REJECT" else (0,130,0)
    cv2.rectangle(out, (0,0), (out.shape[1], 32), bar, -1)
    cv2.putText(out, f"  {decision}  |  {len(defects)} defects",
                (6,21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 2)
    return out

def img_to_b64(img, quality=85):
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()

# ══════════════════════════════════════════════
#  إدارة الكاميرا
# ══════════════════════════════════════════════

def camera_thread():
    global latest_frame
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        ok, frame = cam.read()
        if ok:
            with frame_lock:
                latest_frame = frame.copy()
        time.sleep(0.033)  # ~30 fps

# ══════════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/start_camera", methods=["POST"])
def start_camera():
    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()
    time.sleep(0.8)
    return jsonify({"ok": True})

@app.route("/capture_and_analyze", methods=["POST"])
def capture_and_analyze():
    data = request.json or {}
    mode = data.get("mode", "auto")
    CRITERIA["pixels_per_mm"]          = float(data.get("ppm", 10))
    CRITERIA["max_linear_length_mm"]   = float(data.get("max_linear", 3))
    CRITERIA["max_rounded_diameter_mm"]= float(data.get("max_round", 4))

    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None

    if frame is None:
        return jsonify({"error": "لا توجد صورة من الكاميرا"}), 400

    if mode == "auto":
        mode = auto_detect_mode(frame)

    defects, decision, reasons, stats, mask = analyze_frame(frame, mode)
    annotated = draw_annotations(frame, defects, decision)

    clean_defects = [{k:v for k,v in d.items() if k!="bbox"} for d in defects]

    return jsonify({
        "decision":      decision,
        "mode":          mode,
        "reasons":       reasons,
        "stats":         stats,
        "defects":       clean_defects,
        "original_b64":  img_to_b64(frame),
        "annotated_b64": img_to_b64(annotated),
        "mask_b64":      img_to_b64(mask),
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

@app.route("/analyze_upload", methods=["POST"])
def analyze_upload():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "لا توجد صورة"}), 400
    mode = request.form.get("mode", "auto")
    CRITERIA["pixels_per_mm"]          = float(request.form.get("ppm", 10))
    CRITERIA["max_linear_length_mm"]   = float(request.form.get("max_linear", 3))
    CRITERIA["max_rounded_diameter_mm"]= float(request.form.get("max_round", 4))

    buf = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "تعذر قراءة الصورة"}), 400

    if mode == "auto":
        mode = auto_detect_mode(frame)

    defects, decision, reasons, stats, mask = analyze_frame(frame, mode)
    annotated = draw_annotations(frame, defects, decision)
    clean_defects = [{k:v for k,v in d.items() if k!="bbox"} for d in defects]

    return jsonify({
        "decision":      decision,
        "mode":          mode,
        "reasons":       reasons,
        "stats":         stats,
        "defects":       clean_defects,
        "original_b64":  img_to_b64(frame),
        "annotated_b64": img_to_b64(annotated),
        "mask_b64":      img_to_b64(mask),
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

@app.route("/live_frame")
def live_frame():
    """إرسال الإطار الحالي للكاميرا كـ MJPEG"""
    def gen():
        while True:
            with frame_lock:
                f = latest_frame
            if f is not None:
                _, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')
            time.sleep(0.05)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ══════════════════════════════════════════════
#  HTML — الواجهة الكاملة
# ══════════════════════════════════════════════

HTML = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LPT Live Analyzer</title>
<style>
  :root{
    --bg:#07090f; --surf:#0e1017; --surf2:#141720;
    --border:#1c2030; --accent:#00d4ff;
    --ok:#00c47a; --err:#ff3355; --warn:#ffaa00;
    --txt:#d0d8f0; --muted:#5a6180;
    --mono:'Courier New',monospace;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--txt);font-family:system-ui,sans-serif;font-size:14px;min-height:100vh}

  header{
    background:var(--surf);border-bottom:1px solid var(--border);
    padding:14px 24px;display:flex;align-items:center;gap:14px;
    position:sticky;top:0;z-index:50;
  }
  .logo{
    width:36px;height:36px;border:1.5px solid var(--accent);border-radius:8px;
    display:flex;align-items:center;justify-content:center;font-size:16px;
  }
  header h1{font-size:16px;font-weight:700;color:var(--accent);letter-spacing:1px}
  header small{font-size:11px;color:var(--muted);font-family:var(--mono)}
  .hbadge{
    margin-right:auto;padding:3px 12px;border-radius:20px;
    border:1px solid var(--accent);color:var(--accent);
    font-size:11px;font-family:var(--mono);
  }

  main{display:grid;grid-template-columns:320px 1fr;gap:0;height:calc(100vh - 57px)}
  @media(max-width:800px){main{grid-template-columns:1fr;height:auto}}

  /* ── SIDEBAR ── */
  aside{
    background:var(--surf);border-left:1px solid var(--border);
    padding:18px;overflow-y:auto;display:flex;flex-direction:column;gap:14px;
  }
  .sec-title{
    font-size:10px;letter-spacing:2px;color:var(--accent);
    font-family:var(--mono);text-transform:uppercase;
    margin-bottom:10px;display:flex;align-items:center;gap:6px;
  }
  .sec-title::before{content:'';width:3px;height:12px;background:var(--accent);border-radius:2px;display:inline-block}

  select,input[type=number]{
    width:100%;background:var(--bg);border:1px solid var(--border);
    color:var(--txt);padding:8px 10px;border-radius:7px;
    font-size:13px;font-family:var(--mono);outline:none;
    transition:border-color .2s;margin-bottom:8px;
  }
  select:focus,input:focus{border-color:var(--accent)}

  .lbl{font-size:11px;color:var(--muted);margin-bottom:4px;font-family:var(--mono)}

  .btn{
    width:100%;padding:11px;border-radius:8px;font-size:13px;
    font-family:var(--mono);font-weight:700;letter-spacing:.5px;
    cursor:pointer;border:none;transition:all .18s;
  }
  .btn-primary{background:var(--accent);color:#000}
  .btn-primary:hover{filter:brightness(1.12)}
  .btn-primary:disabled{opacity:.4;cursor:not-allowed}
  .btn-sec{background:transparent;border:1px solid var(--border);color:var(--txt);margin-top:6px}
  .btn-sec:hover{border-color:var(--accent);color:var(--accent)}

  .stats-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .stat{
    background:var(--bg);border:1px solid var(--border);
    border-radius:8px;padding:10px;text-align:center;
  }
  .stat-n{font-size:26px;font-weight:700;font-family:var(--mono);color:var(--accent)}
  .stat-l{font-size:10px;color:var(--muted);margin-top:2px}

  .verdict{
    border-radius:8px;padding:12px;text-align:center;
    font-family:var(--mono);font-size:15px;font-weight:700;
    display:none;letter-spacing:1px;
  }
  .verdict.ok{background:rgba(0,196,122,.1);border:1px solid var(--ok);color:var(--ok)}
  .verdict.bad{background:rgba(255,51,85,.1);border:1px solid var(--err);color:var(--err)}

  .defect-list{max-height:180px;overflow-y:auto}
  .defect-item{
    padding:7px 0;border-bottom:1px solid var(--border);
    font-size:12px;font-family:var(--mono);
    display:flex;justify-content:space-between;align-items:center;
  }
  .defect-item:last-child{border:none}
  .dtag{
    font-size:10px;padding:2px 6px;border-radius:4px;
    display:inline-block;
  }
  .tc{background:rgba(255,51,85,.15);color:var(--err);border:1px solid var(--err)}
  .tp{background:rgba(255,170,0,.15);color:var(--warn);border:1px solid var(--warn)}
  .te{background:rgba(0,212,255,.15);color:var(--accent);border:1px solid var(--accent)}

  /* ── MAIN PANEL ── */
  .view-area{
    display:flex;flex-direction:column;gap:0;overflow:hidden;
  }
  .cam-bar{
    background:var(--surf2);border-bottom:1px solid var(--border);
    padding:10px 18px;display:flex;align-items:center;gap:12px;
  }
  .cam-dot{
    width:8px;height:8px;border-radius:50%;
    background:var(--muted);transition:background .3s;
  }
  .cam-dot.live{background:var(--ok);box-shadow:0 0 6px var(--ok)}
  .cam-label{font-size:12px;font-family:var(--mono);color:var(--muted)}

  .views{
    flex:1;display:grid;grid-template-columns:1fr 1fr;
    grid-template-rows:1fr 1fr;gap:1px;background:var(--border);
    overflow:hidden;
  }
  @media(max-width:800px){.views{grid-template-columns:1fr;grid-template-rows:none}}

  .view-cell{
    background:var(--bg);position:relative;
    display:flex;align-items:center;justify-content:center;
    overflow:hidden;min-height:200px;
  }
  .view-cell img,.view-cell video{
    width:100%;height:100%;object-fit:contain;
  }
  .view-cell-label{
    position:absolute;bottom:8px;right:10px;
    background:rgba(0,0,0,.7);color:var(--muted);
    font-size:10px;font-family:var(--mono);
    padding:3px 8px;border-radius:4px;letter-spacing:1px;
  }
  .placeholder{
    color:var(--muted);font-size:12px;font-family:var(--mono);
    text-align:center;padding:20px;
  }
  .placeholder span{display:block;font-size:28px;margin-bottom:8px;opacity:.3}

  input[type=file]{display:none}
  .upload-btn{
    display:flex;align-items:center;justify-content:center;gap:8px;
    padding:10px;border:1px dashed var(--border);border-radius:8px;
    color:var(--muted);font-size:12px;font-family:var(--mono);
    cursor:pointer;transition:all .2s;margin-bottom:8px;
  }
  .upload-btn:hover{border-color:var(--accent);color:var(--accent)}

  .ts{font-size:11px;color:var(--muted);font-family:var(--mono);text-align:center;margin-top:4px}
  .reason{
    background:rgba(255,51,85,.07);border:1px solid rgba(255,51,85,.2);
    border-radius:6px;padding:7px 10px;font-size:11px;
    font-family:var(--mono);color:var(--err);margin-bottom:6px;
  }
</style>
</head>
<body>

<header>
  <div class="logo">🔬</div>
  <div>
    <h1>LPT LIVE ANALYZER</h1>
    <small>Liquid Penetrant Testing — Real-Time</small>
  </div>
  <span class="hbadge">NDT · v2.0</span>
</header>

<main>
  <!-- ── SIDEBAR ── -->
  <aside>
    <!-- وضع الكشف -->
    <div>
      <div class="sec-title">وضع الكشف</div>
      <div class="lbl">نوع LPT</div>
      <select id="mode">
        <option value="auto">🔄 تلقائي</option>
        <option value="red_dye">🔴 Red Dye</option>
        <option value="fluorescent">💚 Fluorescent UV</option>
      </select>
    </div>

    
    <!-- أزرار -->
    <div>
      <div class="sec-title">التحكم</div>
      <button class="btn btn-primary" id="btn-cam" onclick="startCamera()">تشغيل الكاميرا</button>
      <button class="btn btn-primary" id="btn-capture" onclick="captureAndAnalyze()" disabled style="margin-top:6px">التقاط وتحليل</button>

      <div style="margin-top:12px">
        <label class="upload-btn" for="file-input">
          ⬆ رفع صورة للتحليل
        </label>
        <input type="file" id="file-input" accept="image/*" onchange="analyzeUpload(this)">
      </div>
    </div>

    <!-- نتائج -->
    <div>
      <div class="sec-title">النتائج</div>
      <div class="verdict" id="verdict"></div>
      <div class="ts" id="ts-time"></div>

      <div class="stats-grid" style="margin-top:10px">
        <div class="stat"><div class="stat-n" id="s-total">—</div><div class="stat-l">عيوب</div></div>
        <div class="stat"><div class="stat-n" id="s-maxlen">—</div><div class="stat-l">أطول (mm)</div></div>
        <div class="stat"><div class="stat-n" id="s-cracks">—</div><div class="stat-l">شقوق</div></div>
        <div class="stat"><div class="stat-n" id="s-poros">—</div><div class="stat-l">مسامية</div></div>
      </div>

      <div id="reasons-wrap" style="margin-top:10px;display:none"></div>

      <div class="defect-list" id="defect-list" style="margin-top:8px"></div>
    </div>
  </aside>

  <!-- ── MAIN VIEW ── -->
  <div class="view-area">
    <div class="cam-bar">
      <div class="cam-dot" id="cam-dot"></div>
      <span class="cam-label" id="cam-label">الكاميرا غير مفعّلة</span>
    </div>

    <div class="views">
      <!-- Live -->
      <div class="view-cell" id="cell-live">
        <div class="placeholder"><span>📷</span>اضغط "تشغيل الكاميرا"</div>
        <div class="view-cell-label">LIVE</div>
      </div>

      <!-- Original -->
      <div class="view-cell" id="cell-orig">
        <div class="placeholder"><span>🖼</span>الصورة الملتقطة</div>
        <div class="view-cell-label">CAPTURED</div>
      </div>

      <!-- Annotated -->
      <div class="view-cell" id="cell-ann">
        <div class="placeholder"><span>🔍</span>الصورة المحللة</div>
        <div class="view-cell-label">ANALYZED</div>
      </div>

      <!-- Mask -->
      <div class="view-cell" id="cell-mask">
        <div class="placeholder"><span>🎭</span>عزل العيوب (Mask)</div>
        <div class="view-cell-label">MASK</div>
      </div>
    </div>
  </div>
</main>

<script>
let cameraActive = false;

async function startCamera() {
  const btn = document.getElementById('btn-cam');
  btn.disabled = true;
  btn.textContent = 'جاري التشغيل...';

  try {
    const r = await fetch('/start_camera', {method:'POST'});
    const d = await r.json();
    if (d.ok) {
      cameraActive = true;
      document.getElementById('cam-dot').classList.add('live');
      document.getElementById('cam-label').textContent = 'الكاميرا نشطة — LIVE';
      btn.textContent = 'الكاميرا نشطة ✓';
      document.getElementById('btn-capture').disabled = false;

      // عرض البث المباشر
      const cell = document.getElementById('cell-live');
      cell.innerHTML = '<img src="/live_frame" style="width:100%;height:100%;object-fit:contain"><div class="view-cell-label">LIVE</div>';
    }
  } catch(e) {
    btn.textContent = 'فشل التشغيل';
    btn.disabled = false;
    alert('تعذر تشغيل الكاميرا: ' + e.message);
  }
}

async function captureAndAnalyze() {
  const btn = document.getElementById('btn-capture');
  btn.disabled = true;
  btn.textContent = 'جاري التحليل...';

  try {
    const r = await fetch('/capture_and_analyze', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        mode:       document.getElementById('mode').value,
        ppm:        document.getElementById('ppm').value,
        max_linear: document.getElementById('max_linear').value,
        max_round:  document.getElementById('max_round').value,
      })
    });
    const data = await r.json();
    if (data.error) { alert(data.error); return; }
    showResults(data);
  } catch(e) {
    alert('خطأ في التحليل: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'التقاط وتحليل';
  }
}

async function analyzeUpload(input) {
  const file = input.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append('image', file);
  fd.append('mode',       document.getElementById('mode').value);
  fd.append('ppm',        document.getElementById('ppm').value);
  fd.append('max_linear', document.getElementById('max_linear').value);
  fd.append('max_round',  document.getElementById('max_round').value);

  document.getElementById('btn-capture').textContent = 'جاري التحليل...';
  try {
    const r    = await fetch('/analyze_upload', {method:'POST', body:fd});
    const data = await r.json();
    if (data.error) { alert(data.error); return; }
    showResults(data);
  } catch(e) {
    alert('خطأ: ' + e.message);
  } finally {
    document.getElementById('btn-capture').textContent = 'التقاط وتحليل';
  }
}

function showResults(data) {
  // الصور الأربع
  setImg('cell-orig', data.original_b64,  'CAPTURED');
  setImg('cell-ann',  data.annotated_b64, 'ANALYZED');
  setImg('cell-mask', data.mask_b64,      'MASK');

  // قرار
  const v = document.getElementById('verdict');
  v.style.display = 'block';
  if (data.decision === 'ACCEPT') {
    v.className = 'verdict ok';
    v.textContent = '✓ ACCEPT — القطعة سليمة';
  } else {
    v.className = 'verdict bad';
    v.textContent = '✗ REJECT — مرفوض';
  }

  document.getElementById('ts-time').textContent = data.timestamp + ' · ' + data.mode.toUpperCase();

  // إحصائيات
  document.getElementById('s-total').textContent  = data.stats.total;
  document.getElementById('s-maxlen').textContent = (+data.stats.max_length).toFixed(1);
  document.getElementById('s-cracks').textContent = data.stats.cracks;
  document.getElementById('s-poros').textContent  = data.stats.porosity;

  // أسباب الرفض
  const rw = document.getElementById('reasons-wrap');
  if (data.reasons && data.reasons.length) {
    rw.style.display = 'block';
    rw.innerHTML = data.reasons.map(r=>`<div class="reason">${r}</div>`).join('');
  } else {
    rw.style.display = 'none';
  }

  // جدول العيوب
  const dl = document.getElementById('defect-list');
  if (data.defects && data.defects.length) {
    dl.innerHTML = data.defects.map((d,i)=>{
      const cls = d.type==='Linear Crack'?'tc':d.type==='Round Porosity'?'tp':'te';
      const lbl = d.type==='Linear Crack'?'شق':d.type==='Round Porosity'?'مسامية':'ممتد';
      return `<div class="defect-item">
        <span>#${i+1} <span class="dtag ${cls}">${lbl}</span></span>
        <span>${d.length_mm}×${d.width_mm}mm</span>
      </div>`;
    }).join('');
  } else {
    dl.innerHTML = '<div class="defect-item" style="color:var(--muted)">لا توجد عيوب مكتشفة</div>';
  }
}

function setImg(cellId, b64, label) {
  document.getElementById(cellId).innerHTML =
    `<img src="data:image/jpeg;base64,${b64}" style="width:100%;height:100%;object-fit:contain">
     <div class="view-cell-label">${label}</div>`;
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  🔬 LPT Live Analyzer")
    print("  افتح المتصفح على: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)