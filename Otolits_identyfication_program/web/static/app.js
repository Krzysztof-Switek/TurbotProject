// Turbot — frontend (vanilla JS, ES2022 module-less).
// Łączymy się z backendem przez fetch (/api/*). Stan UI po stronie przeglądarki.
//
// Pliki łączone w stages — kolejne kroki planu (12 → 18) dodają funkcje
// stopniowo. Aktualna zawartość: klasy domenowe (BBox/RowLine/Row),
// Api wrapper, FileBrowser z nawigacją po /api/fs/*.
//
// Kroki 13–18 dorzucą: shared helpers (computeRowLabels etc.), ImageCanvas
// z renderingiem i eventami, tryb EDIT_LABEL z modalem, walidacja + crop.

"use strict";

// ============================================================================
// Tryby (port z input_handler.ManualMode na desktop)
// ============================================================================
const Modes = Object.freeze({
  ADD_BOX:    "ADD_BOX",
  ADD_LINE:   "ADD_LINE",
  MOVE:       "MOVE",
  RESIZE:     "RESIZE",
  DELETE:     "DELETE",
  EDIT_LABEL: "EDIT_LABEL",
});

// ============================================================================
// SHARED HELPERS — port 1:1 z image_cropper.py
// ============================================================================
// KRYTYCZNE: algorytmy muszą być algorytmicznie identyczne z Pythonem.
// Najmniejsze rozjazdy (np. inny tie-break w sortowaniu) dadzą różne etykiety
// w UI vs nazwy plików zapisane przez backend. Jeśli ten plik się zmienia,
// porównaj wyniki na kilku scenariuszach z Python referencyjnym.
//
// W Pythonie kluczem dict labels jest `id(row)` (adres pamięci). W JS używamy
// Map z referencjami obiektów Row — semantyka identyczna.

/** Sortowanie stable po `min(b.y1)` boxów wiersza. Identyczne z key z Pythona. */
function _rowTopY(row) {
  return Math.min(...row.boxes.map(b => b.y1));
}

/** Port z _split_auto_by_largest_gap. */
function splitAutoByLargestGap(sortedRows) {
  const n = sortedRows.length;
  if (n === 0) return [[], []];
  if (n === 1) return [[...sortedRows], []];

  const centroids = sortedRows.map(row =>
    row.boxes.reduce((s, b) => s + (b.y1 + b.y2) / 2, 0) / row.boxes.length
  );
  // gaps: (value, index) — w Pythonie `max` na krotkach robi tie-break po
  // drugim elemencie (index). Bierzemy LARGEST gap, przy tie najwyższy index.
  let bestGap = -Infinity;
  let bestIdx = -1;
  for (let i = 0; i < n - 1; i++) {
    const gap = centroids[i + 1] - centroids[i];
    // ">" — przy równych zostawiamy poprzedni; "===" wpada do else,
    // gdzie nadpisujemy bestIdx żeby symulować Python `max` z tie-break po idx.
    if (gap > bestGap) { bestGap = gap; bestIdx = i; }
    else if (gap === bestGap) { bestIdx = i; }
  }

  return [sortedRows.slice(0, bestIdx + 1), sortedRows.slice(bestIdx + 1)];
}

/** Port z _split_compartments — respektuje row.compartmentOverride. */
function splitCompartments(sortedRows) {
  const forcedA = sortedRows.filter(r => r.compartmentOverride === "A");
  const forcedB = sortedRows.filter(r => r.compartmentOverride === "B");
  const auto    = sortedRows.filter(r => r.compartmentOverride == null);

  const [autoA, autoB] = splitAutoByLargestGap(auto);

  const aRows = [...forcedA, ...autoA].sort((x, y) => _rowTopY(x) - _rowTopY(y));
  const bRows = [...forcedB, ...autoB].sort((x, y) => _rowTopY(x) - _rowTopY(y));
  return [aRows, bRows];
}

/**
 * Port z compute_row_labels.
 * Zwraca { labels: Map<Row, "A_n">, error: string|null }.
 */
function computeRowLabels(rows) {
  // sort top→bottom, pomijając wiersze bez boxów (jak Python comprehension)
  const sortedRows = rows
    .filter(r => r.boxes && r.boxes.length > 0)
    .sort((a, b) => _rowTopY(a) - _rowTopY(b));

  const total = sortedRows.length;
  if (total > 6) {
    return {
      labels: new Map(),
      error: `BŁĄD: wykryto ${total} wierszy łącznie (max 6: po 3 na wycinek). Popraw wiersze ręcznie (usuń nadmiarowe linie).`,
    };
  }

  const [aRows, bRows] = splitCompartments(sortedRows);
  const nA = aRows.length;
  const nB = bRows.length;

  if (nA > 3 || nB > 3) {
    return {
      labels: new Map(),
      error: `BŁĄD: wycinek A ma ${nA} wierszy, wycinek B ma ${nB} wierszy (max 3 na wycinek). Popraw wiersze ręcznie.`,
    };
  }

  const labels = new Map();
  for (const [label, rowsInC] of [["A", aRows], ["B", bRows]]) {
    const n = rowsInC.length;
    rowsInC.forEach((row, offset) => {
      const rowNum = 3 - (n - 1 - offset);
      labels.set(row, `${label}_${rowNum}`);
    });
  }
  return { labels, error: null };
}

/**
 * Port z compute_compartment_bboxes — bbox każdego wycinka jako bounding-box
 * wszystkich boxów w jego wierszach + margines.
 * Zwraca { A: [x1,y1,x2,y2], B: [x1,y1,x2,y2] } — wycinki bez wierszy pomijane.
 */
function computeCompartmentBboxes(rows, { margin = 40, labels = null } = {}) {
  let effLabels = labels;
  if (effLabels === null) {
    const result = computeRowLabels(rows);
    if (result.error !== null) return {};
    effLabels = result.labels;
  }

  const byCompartment = { A: [], B: [] };
  for (const row of rows) {
    if (!row.boxes || row.boxes.length === 0) continue;
    const label = effLabels.get(row);
    if (!label) continue;
    byCompartment[label[0]].push(...row.boxes);
  }

  const result = {};
  for (const label of ["A", "B"]) {
    const boxes = byCompartment[label];
    if (boxes.length === 0) continue;
    const x1 = Math.min(...boxes.map(b => b.x1)) - margin;
    const y1 = Math.min(...boxes.map(b => b.y1)) - margin;
    const x2 = Math.max(...boxes.map(b => b.x2)) + margin;
    const y2 = Math.max(...boxes.map(b => b.y2)) + margin;
    result[label] = [x1, y1, x2, y2];
  }
  return result;
}

// ============================================================================
// BBox — port z bounding_box.BoundingBox
// ============================================================================
class BBox {
  constructor(x1, y1, x2, y2, { id = null, label = "user" } = {}) {
    this.x1 = Math.min(x1, x2);
    this.y1 = Math.min(y1, y2);
    this.x2 = Math.max(x1, x2);
    this.y2 = Math.max(y1, y2);
    this.id = id ?? crypto.randomUUID();
    this.label = label;
  }

  get width()  { return this.x2 - this.x1; }
  get height() { return this.y2 - this.y1; }
  get centerX() { return (this.x1 + this.x2) / 2; }
  get centerY() { return (this.y1 + this.y2) / 2; }

  contains(x, y, tolerance = 0) {
    return x >= this.x1 - tolerance && x <= this.x2 + tolerance
        && y >= this.y1 - tolerance && y <= this.y2 + tolerance;
  }

  /** Indeks najbliższego rogu (0=LU, 1=PG, 2=LD, 3=PD) — port z BoundingBox. */
  getNearestCorner(x, y) {
    const corners = [
      [this.x1, this.y1], [this.x2, this.y1],
      [this.x1, this.y2], [this.x2, this.y2],
    ];
    let bestIdx = 0;
    let bestDist = Infinity;
    for (let i = 0; i < 4; i++) {
      const d = Math.hypot(corners[i][0] - x, corners[i][1] - y);
      if (d < bestDist) { bestDist = d; bestIdx = i; }
    }
    return bestIdx;
  }

  resizeCorner(idx, x, y) {
    if (idx === 0) { this.x1 = x; this.y1 = y; }
    else if (idx === 1) { this.x2 = x; this.y1 = y; }
    else if (idx === 2) { this.x1 = x; this.y2 = y; }
    else if (idx === 3) { this.x2 = x; this.y2 = y; }
    [this.x1, this.x2] = [Math.min(this.x1, this.x2), Math.max(this.x1, this.x2)];
    [this.y1, this.y2] = [Math.min(this.y1, this.y2), Math.max(this.y1, this.y2)];
  }

  move(dx, dy) {
    this.x1 += dx; this.x2 += dx;
    this.y1 += dy; this.y2 += dy;
  }

  toJSON() { return { x1: this.x1, y1: this.y1, x2: this.x2, y2: this.y2 }; }
}

// ============================================================================
// RowLine — port z row_detector.RowLine
// ============================================================================
class RowLine {
  constructor(p1, p2, { id = null } = {}) {
    this.p1 = [...p1];   // [x, y]
    this.p2 = [...p2];
    this.id = id ?? crypto.randomUUID();
  }

  move(dx, dy) {
    this.p1[0] += dx; this.p1[1] += dy;
    this.p2[0] += dx; this.p2[1] += dy;
  }

  /** Odległość punktu od linii (port z _distance_to_line). */
  distanceTo(x, y) {
    const [x1, y1] = this.p1;
    const [x2, y2] = this.p2;
    const num = Math.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1);
    const den = Math.hypot(y2 - y1, x2 - x1);
    return den > 0 ? num / den : 0;
  }

  /** Minimum z odległości do p1 / p2 / od odcinka — port z get_line_at. */
  proximity(x, y) {
    const d1 = Math.hypot(x - this.p1[0], y - this.p1[1]);
    const d2 = Math.hypot(x - this.p2[0], y - this.p2[1]);
    return Math.min(d1, d2, this.distanceTo(x, y));
  }

  toJSON() { return { p1: this.p1, p2: this.p2 }; }
}

// ============================================================================
// Row — port z row_detector.RowDetector.Row
// ============================================================================
class Row {
  constructor(line, { boxes = [], compartmentOverride = null } = {}) {
    this.line = line;
    this.boxes = boxes;          // tablica BBox referencji
    this.compartmentOverride = compartmentOverride;  // null | 'A' | 'B'
  }

  /** Sortowanie "top→bottom" wewnątrz wycinka — pochodne z min(y1) boxów. */
  topY() {
    return this.boxes.length === 0 ? Infinity : Math.min(...this.boxes.map(b => b.y1));
  }

  /** Centroid Y średnia środków boxów — używana przy largest-gap split. */
  centerY() {
    if (this.boxes.length === 0) return 0;
    return this.boxes.reduce((sum, b) => sum + b.centerY, 0) / this.boxes.length;
  }

  /** Port z _does_line_intersect_box (cross-product). */
  intersectsBox(box) {
    const corners = [
      [box.x1, box.y1], [box.x2, box.y1],
      [box.x2, box.y2], [box.x1, box.y2],
    ];
    for (let i = 0; i < 4; i++) {
      if (segmentsIntersect(this.line.p1, this.line.p2, corners[i], corners[(i + 1) % 4])) {
        return true;
      }
    }
    return box.contains(this.line.p1[0], this.line.p1[1])
        || box.contains(this.line.p2[0], this.line.p2[1]);
  }

  toJSON() {
    return {
      line: this.line.toJSON(),
      boxes: this.boxes.map(b => b.toJSON()),
      compartment_override: this.compartmentOverride,
    };
  }
}

/** Port z _line_segments_intersect (cross-product 2D). */
function segmentsIntersect(p1, p2, q1, q2) {
  const r = [p2[0] - p1[0], p2[1] - p1[1]];
  const s = [q2[0] - q1[0], q2[1] - q1[1]];
  const qp = [q1[0] - p1[0], q1[1] - p1[1]];
  const rs = r[0] * s[1] - r[1] * s[0];
  if (Math.abs(rs) < 1e-12) return false;
  const t = (qp[0] * s[1] - qp[1] * s[0]) / rs;
  const u = (qp[0] * r[1] - qp[1] * r[0]) / rs;
  return t >= 0 && t <= 1 && u >= 0 && u <= 1;
}

// ============================================================================
// Api — wrappery na fetch
// ============================================================================
const Api = {
  async listDir(path = "") {
    const url = `/api/fs/list?path=${encodeURIComponent(path)}`;
    const res = await fetch(url);
    if (!res.ok) throw new ApiError(res.status, await res.json().catch(() => ({})));
    return res.json();
  },

  async mkdir(path) {
    const res = await fetch("/api/fs/mkdir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    if (!res.ok) throw new ApiError(res.status, await res.json().catch(() => ({})));
    return res.json();
  },

  previewUrl(path) {
    return `/api/image/preview?path=${encodeURIComponent(path)}`;
  },

  async detect(path) {
    const res = await fetch("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    if (!res.ok) throw new ApiError(res.status, await res.json().catch(() => ({})));
    return res.json();
  },

  async crop(payload) {
    const res = await fetch("/api/crop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new ApiError(res.status, await res.json().catch(() => ({})));
    return res.json();
  },
};

class ApiError extends Error {
  constructor(status, body) {
    super(body?.detail ?? `HTTP ${status}`);
    this.status = status;
    this.body = body;
  }
}

// ============================================================================
// FileBrowser — nawigacja po katalogach + lista obrazów + mkdir
// ============================================================================
class FileBrowser {
  constructor({ onSelectImage, onSelectOutputDir } = {}) {
    this.currentPath = "";
    this.currentDirCache = null;
    this.onSelectImage = onSelectImage ?? (() => {});
    this.onSelectOutputDir = onSelectOutputDir ?? (() => {});

    this.$breadcrumbs = document.getElementById("breadcrumbs");
    this.$dirs = document.getElementById("dirs");
    this.$images = document.getElementById("images");
    this.$outputDirCurrent = document.getElementById("output-dir-current");
    this.$btnRefresh = document.getElementById("btn-refresh");
    this.$btnMkdir = document.getElementById("btn-mkdir");
    this.$btnSetOutput = document.getElementById("btn-set-output");

    this.outputDir = null;

    this.$btnRefresh.addEventListener("click", () => this.refresh());
    this.$btnMkdir.addEventListener("click", () => this._promptMkdir());
    this.$btnSetOutput.addEventListener("click", () => this.setOutputDir(this.currentPath));
  }

  async cdTo(path) {
    try {
      const data = await Api.listDir(path);
      this.currentPath = data.path;
      this.currentDirCache = data;
      this._render();
    } catch (e) {
      console.error("listDir failed:", e);
      alert(`Błąd ładowania katalogu '${path}': ${e.message}`);
    }
  }

  refresh() { this.cdTo(this.currentPath); }

  setOutputDir(path) {
    this.outputDir = path;
    this.$outputDirCurrent.textContent = path === "" ? "(root)" : path;
  }

  getOutputDir() { return this.outputDir; }

  _render() {
    if (!this.currentDirCache) return;
    const data = this.currentDirCache;

    // Breadcrumbs
    this.$breadcrumbs.innerHTML = "";
    const rootCrumb = document.createElement("span");
    rootCrumb.className = "crumb";
    rootCrumb.textContent = "/";
    rootCrumb.addEventListener("click", () => this.cdTo(""));
    this.$breadcrumbs.appendChild(rootCrumb);
    if (data.path) {
      const parts = data.path.split("/");
      let acc = "";
      parts.forEach((part, i) => {
        const sep = document.createElement("span");
        sep.className = "sep";
        sep.textContent = "/";
        this.$breadcrumbs.appendChild(sep);
        acc = acc === "" ? part : `${acc}/${part}`;
        const crumb = document.createElement("span");
        crumb.className = "crumb";
        crumb.textContent = part;
        const target = acc;
        crumb.addEventListener("click", () => this.cdTo(target));
        this.$breadcrumbs.appendChild(crumb);
      });
    }

    // Katalogi (parent jeśli istnieje, potem podkatalogi)
    this.$dirs.innerHTML = "";
    if (data.parent !== null && data.parent !== undefined) {
      const li = document.createElement("li");
      li.textContent = ".. (wyżej)";
      li.style.fontStyle = "italic";
      li.addEventListener("click", () => this.cdTo(data.parent));
      this.$dirs.appendChild(li);
    }
    data.dirs.forEach(d => {
      const li = document.createElement("li");
      li.textContent = d.name;
      const target = data.path === "" ? d.name : `${data.path}/${d.name}`;
      li.addEventListener("click", () => this.cdTo(target));
      this.$dirs.appendChild(li);
    });

    // Obrazy
    this.$images.innerHTML = "";
    data.images.forEach(img => {
      const li = document.createElement("li");
      li.textContent = `${img.name} (${formatBytes(img.size)})`;
      const fullPath = data.path === "" ? img.name : `${data.path}/${img.name}`;
      li.addEventListener("click", () => {
        Array.from(this.$images.children).forEach(c => c.classList.remove("selected"));
        li.classList.add("selected");
        this.onSelectImage(fullPath);
      });
      this.$images.appendChild(li);
    });
  }

  async _promptMkdir() {
    const backdrop = document.getElementById("mkdir-backdrop");
    const nameInput = document.getElementById("mkdir-name");
    const errorEl = document.getElementById("mkdir-error");
    const okBtn = document.getElementById("mkdir-ok");
    const cancelBtn = document.getElementById("mkdir-cancel");

    nameInput.value = "";
    errorEl.hidden = true;
    backdrop.hidden = false;
    nameInput.focus();

    const close = () => { backdrop.hidden = true; };

    const onOk = async () => {
      const name = nameInput.value.trim();
      if (!name) { errorEl.textContent = "Podaj nazwę"; errorEl.hidden = false; return; }
      const path = this.currentPath === "" ? name : `${this.currentPath}/${name}`;
      try {
        await Api.mkdir(path);
        close();
        this.refresh();
      } catch (e) {
        errorEl.textContent = e.message;
        errorEl.hidden = false;
      }
    };

    const cleanup = () => {
      okBtn.removeEventListener("click", onOk);
      cancelBtn.removeEventListener("click", onCancel);
      nameInput.removeEventListener("keydown", onKey);
    };
    const onCancel = () => { cleanup(); close(); };
    const onKey = (e) => {
      if (e.key === "Enter") { e.preventDefault(); onOk(); }
      else if (e.key === "Escape") { e.preventDefault(); onCancel(); }
    };

    okBtn.addEventListener("click", onOk);
    cancelBtn.addEventListener("click", onCancel);
    nameInput.addEventListener("keydown", onKey);
  }
}

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} kB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

// ============================================================================
// ImageCanvas — render warstw + obsługa myszy/klawiatury
// ============================================================================
// Stan UI w pełni po stronie przeglądarki: imagePath, scale, originalSize,
// boxes[], rows[], mode, selection (drag / temp_box / temp_line).
//
// Render warstwowy (port z desktop ImageWindow.update_display):
//   1. Tło (przeskalowany obraz preview)
//   2. Boxy (zielone gdy w wierszu, czerwone gdy nie)
//   3. Linie wierszy (żółte)
//   4. Ramki wycinków (niebieska A, fioletowa B)
//   (5–7 etykiety + statusbar — krok 15; 8 walidacja — krok 17)

const COLORS = Object.freeze({
  BOX_OUT_OF_ROW: "#FF0000",
  BOX_IN_ROW:     "#00FF00",
  LINE:           "#FFFF00",
  COMPARTMENT_A:  "#0000FF",
  COMPARTMENT_B:  "#C800C8",
  SELECTED:       "#FFFFFF",
});

const LINE_PROXIMITY_PX = 10;   // port get_line_at tolerancji
const BOX_PICK_TOLERANCE = 5;   // port get_box_at tolerancji

class ImageCanvas {
  constructor({ canvas, onStateChange } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onStateChange = onStateChange ?? (() => {});

    // Stan
    this.imagePath = null;
    this.image = null;            // HTMLImageElement
    this.scale = 1.0;
    this.originalSize = [0, 0];
    this.boxes = [];              // BBox[]
    this.rows = [];               // Row[]
    this.mode = Modes.ADD_LINE;
    this.selection = this._emptySelection();
    this.tempBox = null;          // BBox podczas rysowania
    this.lastLabels = new Map();  // Map<Row, "A_n"> z ostatniego renderu
    this.lastCounts = { A: 0, B: 0 };
    this.lastError = null;        // err z computeRowLabels, do statusbara

    // Event handlers
    canvas.addEventListener("mousedown", (e) => this._onMouseDown(e));
    canvas.addEventListener("mousemove", (e) => this._onMouseMove(e));
    canvas.addEventListener("mouseup",   (e) => this._onMouseUp(e));
    canvas.addEventListener("mouseleave",(e) => this._onMouseUp(e)); // zachowanie jak LMB up
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  _emptySelection() {
    return { element: null, dragStart: null, isDrawing: false, cornerIdx: null };
  }

  // ---------- public API ----------

  async loadImage(path) {
    const res = await fetch(Api.previewUrl(path));
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new ApiError(res.status, body);
    }
    const scale = parseFloat(res.headers.get("X-Scale") ?? "1");
    const origW = parseInt(res.headers.get("X-Original-Width") ?? "0", 10);
    const origH = parseInt(res.headers.get("X-Original-Height") ?? "0", 10);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);

    const img = await new Promise((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = reject;
      el.src = url;
    });

    this.imagePath = path;
    this.image = img;
    this.scale = scale;
    this.originalSize = [origW, origH];
    this.boxes = [];
    this.rows = [];
    this.selection = this._emptySelection();
    this.tempBox = null;
    this.lastError = null;

    this.canvas.width = img.width;
    this.canvas.height = img.height;
    this.render();
  }

  setMode(mode) {
    if (!Object.values(Modes).includes(mode)) return;
    this.mode = mode;
    this.selection = this._emptySelection();
    this.tempBox = null;
    this.render();
  }

  applyDetectedBoxes(detectedBoxes) {
    // Dodawane do istniejących (user może już mieć ręczne boxy).
    for (const d of detectedBoxes) {
      this.boxes.push(new BBox(d.x1, d.y1, d.x2, d.y2, { label: d.label ?? "auto" }));
    }
    this._reassignAllBoxesToRows();
    this.render();
  }

  toCropPayload(outputDir, saveAnnotations) {
    return {
      image_path: this.imagePath,
      output_dir: outputDir,
      scale: this.scale,
      save_annotations: !!saveAnnotations,
      rows: this.rows
        .filter(r => r.boxes.length > 0)
        .map(r => r.toJSON()),
    };
  }

  // ---------- selekcja: znajdź box / linię pod kursorem ----------

  _findBoxAt(x, y, tolerance = BOX_PICK_TOLERANCE) {
    // Iteracja od końca (najnowsze wyżej) — port get_box_at.
    for (let i = this.boxes.length - 1; i >= 0; i--) {
      if (this.boxes[i].contains(x, y, tolerance)) return this.boxes[i];
    }
    return null;
  }

  _findRowByLineAt(x, y, tolerance = LINE_PROXIMITY_PX) {
    for (const row of this.rows) {
      if (row.line.proximity(x, y) < tolerance) return row;
    }
    return null;
  }

  // ---------- przypisania boxów do wierszy ----------

  /** Port _update_row_boxes — przelicza boxes wiersza po edycji linii. */
  _updateRowBoxes(row) {
    row.boxes = this.boxes.filter(b => row.intersectsBox(b));
    row.boxes.sort((a, b) => a.centerX - b.centerX);
  }

  _reassignAllBoxesToRows() {
    for (const row of this.rows) this._updateRowBoxes(row);
  }

  // ---------- event handlers myszy ----------

  _eventCoords(e) {
    const rect = this.canvas.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  }

  _onMouseDown(e) {
    if (e.button !== 0) return; // tylko LMB
    const [x, y] = this._eventCoords(e);

    if (this.mode === Modes.ADD_BOX) {
      this.tempBox = new BBox(x, y, x + 1, y + 1, { label: "user" });
      this.selection.element = this.tempBox;
      this.selection.dragStart = [x, y];
      this.selection.isDrawing = true;
    } else if (this.mode === Modes.ADD_LINE) {
      // Tymczasowa linia jako element w selection (rysowana w render()).
      const tempLine = new RowLine([x, y], [x, y]);
      this.selection.element = tempLine;
      this.selection.dragStart = [x, y];
      this.selection.isDrawing = true;
    } else if (this.mode === Modes.MOVE) {
      const box = this._findBoxAt(x, y);
      if (box) {
        this.selection.element = box;
      } else {
        const row = this._findRowByLineAt(x, y);
        if (row) this.selection.element = row;
      }
      this.selection.dragStart = [x, y];
      this.selection.isDrawing = !!this.selection.element;
    } else if (this.mode === Modes.RESIZE) {
      const box = this._findBoxAt(x, y);
      if (box) {
        this.selection.element = box;
        this.selection.cornerIdx = box.getNearestCorner(x, y);
      } else {
        const row = this._findRowByLineAt(x, y);
        if (row) {
          const d1 = Math.hypot(x - row.line.p1[0], y - row.line.p1[1]);
          const d2 = Math.hypot(x - row.line.p2[0], y - row.line.p2[1]);
          this.selection.element = row;
          this.selection.cornerIdx = d1 < d2 ? "p1" : "p2";
        }
      }
      this.selection.dragStart = [x, y];
      this.selection.isDrawing = !!this.selection.element;
    } else if (this.mode === Modes.DELETE) {
      const box = this._findBoxAt(x, y);
      if (box) {
        this.boxes = this.boxes.filter(b => b !== box);
        for (const row of this.rows) row.boxes = row.boxes.filter(b => b !== box);
      } else {
        const row = this._findRowByLineAt(x, y);
        if (row) this.rows = this.rows.filter(r => r !== row);
      }
      this.render();
    } else if (this.mode === Modes.EDIT_LABEL) {
      // Klik na linię → modal. Implementacja w kroku 16.
      const row = this._findRowByLineAt(x, y);
      if (row) this._openEditLabelModal?.(row);
    }
  }

  _onMouseMove(e) {
    if (!this.selection.isDrawing) return;
    const [x, y] = this._eventCoords(e);

    if (this.mode === Modes.ADD_BOX && this.tempBox) {
      // Aktualizuj prawy-dolny róg (start fixed w dragStart).
      const [sx, sy] = this.selection.dragStart;
      this.tempBox.x1 = Math.min(sx, x);
      this.tempBox.y1 = Math.min(sy, y);
      this.tempBox.x2 = Math.max(sx, x);
      this.tempBox.y2 = Math.max(sy, y);
      this.render();
      return;
    }

    if (this.mode === Modes.ADD_LINE && this.selection.element instanceof RowLine) {
      this.selection.element.p2 = [x, y];
      this.render();
      return;
    }

    if (this.selection.element == null) return;

    const [px, py] = this.selection.dragStart;
    const dx = x - px;
    const dy = y - py;
    this.selection.dragStart = [x, y];

    if (this.mode === Modes.MOVE) {
      if (this.selection.element instanceof Row) {
        this.selection.element.line.move(dx, dy);
        this._reassignAllBoxesToRows();
      } else if (this.selection.element instanceof BBox) {
        this.selection.element.move(dx, dy);
      }
    } else if (this.mode === Modes.RESIZE) {
      if (this.selection.element instanceof Row) {
        if (this.selection.cornerIdx === "p1") this.selection.element.line.p1 = [x, y];
        else if (this.selection.cornerIdx === "p2") this.selection.element.line.p2 = [x, y];
        this._reassignAllBoxesToRows();
      } else if (this.selection.element instanceof BBox) {
        this.selection.element.resizeCorner(this.selection.cornerIdx, x, y);
      }
    }
    this.render();
  }

  _onMouseUp(e) {
    if (!this.selection.isDrawing) return;
    const wasDrawing = this.selection.isDrawing;
    const elem = this.selection.element;
    const mode = this.mode;
    this.selection = this._emptySelection();

    if (mode === Modes.ADD_BOX && this.tempBox) {
      if (this.tempBox.width > 2 && this.tempBox.height > 2) {
        this.boxes.push(this.tempBox);
        this._reassignAllBoxesToRows();
      }
      this.tempBox = null;
    } else if (mode === Modes.ADD_LINE && elem instanceof RowLine) {
      const length = Math.hypot(elem.p2[0] - elem.p1[0], elem.p2[1] - elem.p1[1]);
      if (length >= 10) {
        // Tworzymy Row z linią + boxami które przecina (port finish_line).
        const row = new Row(elem, { boxes: [] });
        this._updateRowBoxes(row);
        if (row.boxes.length > 0) this.rows.push(row);
      }
    }

    if (wasDrawing) {
      this.render();
    }
  }

  // ---------- render ----------

  render() {
    if (!this.image) {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.onStateChange();
      return;
    }
    const ctx = this.ctx;

    // Warstwa 1: tło
    ctx.drawImage(this.image, 0, 0);

    // Oblicz etykiety + ramki + liczniki (cache do statusbara).
    const { labels, error } = computeRowLabels(this.rows);
    this.lastLabels = labels;
    this.lastError = error;
    const counts = { A: 0, B: 0 };
    for (const lbl of labels.values()) counts[lbl[0]]++;
    this.lastCounts = counts;
    const bboxes = error === null
      ? computeCompartmentBboxes(this.rows, { labels })
      : {};

    // Warstwa 2: boxy
    for (const box of this.boxes) {
      const inRow = this.rows.some(r => r.boxes.includes(box));
      ctx.strokeStyle = inRow ? COLORS.BOX_IN_ROW : COLORS.BOX_OUT_OF_ROW;
      ctx.lineWidth = 1;
      ctx.strokeRect(box.x1, box.y1, box.width, box.height);
    }
    if (this.tempBox) {
      ctx.strokeStyle = COLORS.BOX_OUT_OF_ROW;
      ctx.lineWidth = 1;
      ctx.strokeRect(this.tempBox.x1, this.tempBox.y1, this.tempBox.width, this.tempBox.height);
    }

    // Warstwa 3: linie wierszy (+ aktualnie rysowana)
    ctx.strokeStyle = COLORS.LINE;
    ctx.lineWidth = 1;
    for (const row of this.rows) {
      ctx.beginPath();
      ctx.moveTo(row.line.p1[0], row.line.p1[1]);
      ctx.lineTo(row.line.p2[0], row.line.p2[1]);
      ctx.stroke();
    }
    if (this.selection.element instanceof RowLine && this.selection.isDrawing && this.mode === Modes.ADD_LINE) {
      const l = this.selection.element;
      ctx.beginPath();
      ctx.moveTo(l.p1[0], l.p1[1]);
      ctx.lineTo(l.p2[0], l.p2[1]);
      ctx.stroke();
    }

    // Warstwa 4: ramki wycinków
    for (const label of ["A", "B"]) {
      const b = bboxes[label];
      if (!b) continue;
      ctx.strokeStyle = label === "A" ? COLORS.COMPARTMENT_A : COLORS.COMPARTMENT_B;
      ctx.lineWidth = 2;
      ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
    }

    // Warstwa 5: literowe A/B w lewym górnym rogu ramki wycinka
    ctx.font = "bold 18px sans-serif";
    ctx.textBaseline = "top";
    for (const label of ["A", "B"]) {
      const b = bboxes[label];
      if (!b) continue;
      const x = b[0] + 4;
      const y = b[1] + 2;
      ctx.lineWidth = 3;
      ctx.strokeStyle = "#FFFFFF";
      ctx.strokeText(label, x, y);
      ctx.fillStyle = label === "A" ? COLORS.COMPARTMENT_A : COLORS.COMPARTMENT_B;
      ctx.fillText(label, x, y);
    }

    // Warstwa 6: etykiety wierszy (A_1, B_3) przy row.line.p1 + offset
    ctx.font = "bold 14px sans-serif";
    if (error === null) {
      for (const row of this.rows) {
        const label = labels.get(row);
        if (!label) continue;
        const x = row.line.p1[0] + 8;
        const y = row.line.p1[1] - 20;
        ctx.lineWidth = 3;
        ctx.strokeStyle = "#000000";
        ctx.strokeText(label, x, y);
        ctx.fillStyle = "#FFFFFF";
        ctx.fillText(label, x, y);
      }
    }
    // (warstwa 7 statusbar — w bootstrap czyta this.lastCounts / lastError;
    //  warstwa 8 komunikat błędu na canvasie — krok 17)

    // Powiadom listenera (statusbar / przyciski) po każdym renderze.
    this.onStateChange();
  }
}

// ============================================================================
// Smoke testy shared helpers (port spójności JS ↔ Python)
// ============================================================================
// Mini sanity check uruchamiany na starcie. Nie zastępuje prawdziwych testów,
// ale wyłapie najgrubsze regresje (np. zła kolejność numerowania "od dołu").
// Jeśli któryś assert pada, console.error informuje + opis. Strona dalej
// działa — to nie blokada.

function _mockBox(x1, y1, x2, y2) {
  return { x1, y1, x2, y2, centerY: (y1 + y2) / 2 };
}
function _mockRow(boxes, compartmentOverride = null) {
  return { boxes, compartmentOverride };
}

function _runSmokeTests() {
  const cases = [];

  // 3+3 wierszy → A_1,A_2,A_3 + B_1,B_2,B_3 (default split przez luki Y)
  {
    const rows = [
      _mockRow([_mockBox(0, 0, 10, 50)]),     // y mid 25
      _mockRow([_mockBox(0, 60, 10, 110)]),   // y mid 85
      _mockRow([_mockBox(0, 120, 10, 170)]),  // y mid 145
      _mockRow([_mockBox(0, 400, 10, 450)]),  // y mid 425  ← duża luka 145→425
      _mockRow([_mockBox(0, 460, 10, 510)]),  // y mid 485
      _mockRow([_mockBox(0, 520, 10, 570)]),  // y mid 545
    ];
    const { labels, error } = computeRowLabels(rows);
    cases.push({ name: "3+3 wierszy", pass: error === null
      && labels.get(rows[0]) === "A_1"
      && labels.get(rows[1]) === "A_2"
      && labels.get(rows[2]) === "A_3"
      && labels.get(rows[3]) === "B_1"
      && labels.get(rows[4]) === "B_2"
      && labels.get(rows[5]) === "B_3" });
  }

  // 2 wiersze "od dołu" → A_2, A_3 (z gap split: 1 do A, 1 do B; ale tu jeśli
  // tylko 2 wiersze i obie nad nielogicznym progiem, gap zawsze cos znajdzie).
  // Test specjalny: 2 wiersze z dużą luką między sobą.
  {
    const rows = [
      _mockRow([_mockBox(0, 0, 10, 50)]),
      _mockRow([_mockBox(0, 400, 10, 450)]),
    ];
    const { labels, error } = computeRowLabels(rows);
    // splitAutoByLargestGap z 2 wierszami: idx=0 -> [0], [1]. Czyli A=1, B=1.
    cases.push({ name: "2 wiersze z luka 1+1", pass: error === null
      && labels.get(rows[0]) === "A_3"
      && labels.get(rows[1]) === "B_3" });
  }

  // Override 'A' na wszystkie 2 wiersze → A_2, A_3 (wymusza A).
  {
    const rows = [
      _mockRow([_mockBox(0, 0, 10, 50)], "A"),
      _mockRow([_mockBox(0, 400, 10, 450)], "A"),
    ];
    const { labels, error } = computeRowLabels(rows);
    cases.push({ name: "2 forced A", pass: error === null
      && labels.get(rows[0]) === "A_2"
      && labels.get(rows[1]) === "A_3" });
  }

  // 4 forced A → error
  {
    const rows = [
      _mockRow([_mockBox(0, 0, 10, 50)], "A"),
      _mockRow([_mockBox(0, 60, 10, 110)], "A"),
      _mockRow([_mockBox(0, 120, 10, 170)], "A"),
      _mockRow([_mockBox(0, 180, 10, 230)], "A"),
    ];
    const { error } = computeRowLabels(rows);
    cases.push({ name: "4 forced A → error", pass: error !== null && error.includes("4 wierszy") });
  }

  // 7 wierszy global → error
  {
    const rows = Array.from({ length: 7 }, (_, i) =>
      _mockRow([_mockBox(0, i * 100, 10, i * 100 + 50)])
    );
    const { error } = computeRowLabels(rows);
    cases.push({ name: "7 wierszy → error", pass: error !== null && error.includes("7 wierszy") });
  }

  const failed = cases.filter(c => !c.pass);
  if (failed.length === 0) {
    console.log("[smoke] computeRowLabels OK (%d/%d testów)", cases.length, cases.length);
  } else {
    console.error("[smoke] computeRowLabels FAIL:", failed.map(f => f.name));
  }
  return failed.length === 0;
}

// ============================================================================
// Bootstrap
// ============================================================================
const KEY_TO_MODE = {
  b: Modes.ADD_BOX,
  l: Modes.ADD_LINE,
  v: Modes.MOVE,
  r: Modes.RESIZE,
  d: Modes.DELETE,
  e: Modes.EDIT_LABEL,
};

window.addEventListener("DOMContentLoaded", () => {
  _runSmokeTests();

  // Canvas
  const canvas = document.getElementById("canvas");
  const $statusMode = document.getElementById("status-mode");
  const $statusFile = document.getElementById("status-file");
  const $statusScale = document.getElementById("status-scale");
  const $statusCounts = document.getElementById("status-counts");
  const $statusError = document.getElementById("status-error");
  const $btnDetect = document.getElementById("btn-detect");
  const $btnCrop = document.getElementById("btn-crop");

  const updateStatusBar = () => {
    $statusMode.textContent = `Tryb: ${imageCanvas.mode}`;
    $statusFile.textContent = `Plik: ${imageCanvas.imagePath ?? "—"}`;
    $statusScale.textContent = imageCanvas.image
      ? `Skala: ${imageCanvas.scale.toFixed(3)}`
      : "Skala: —";
    const c = imageCanvas.lastCounts;
    $statusCounts.textContent = `Wycinek A: ${c.A} wierszy | Wycinek B: ${c.B} wierszy`;
    $statusError.textContent = imageCanvas.lastError ?? "";
  };

  const updateActionButtons = () => {
    const hasImage = !!imageCanvas.image;
    $btnDetect.disabled = !hasImage;
    $btnCrop.disabled = !hasImage || !fileBrowser.getOutputDir();
  };

  const updateModeButtons = () => {
    document.querySelectorAll(".mode-btn").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.mode === imageCanvas.mode);
    });
  };

  const imageCanvas = new ImageCanvas({
    canvas,
    onStateChange: () => {
      updateStatusBar();
      updateActionButtons();
      updateModeButtons();
    },
  });

  // FileBrowser → loadImage
  const fileBrowser = new FileBrowser({
    onSelectImage: async (path) => {
      try {
        await imageCanvas.loadImage(path);
      } catch (e) {
        alert(`Błąd ładowania obrazu: ${e.message}`);
      }
    },
  });
  fileBrowser.cdTo("");

  // Mode buttons
  document.querySelectorAll(".mode-btn").forEach(btn => {
    btn.addEventListener("click", () => imageCanvas.setMode(btn.dataset.mode));
  });

  // Auto-detect button
  $btnDetect.addEventListener("click", async () => {
    if (!imageCanvas.imagePath) return;
    $btnDetect.disabled = true;
    try {
      const res = await Api.detect(imageCanvas.imagePath);
      imageCanvas.applyDetectedBoxes(res.boxes);
    } catch (e) {
      alert(`Auto-detect: ${e.message}`);
    } finally {
      updateActionButtons();
    }
  });

  // Crop button
  $btnCrop.addEventListener("click", async () => {
    const outputDir = fileBrowser.getOutputDir();
    if (outputDir === null || !imageCanvas.imagePath) return;
    const saveAnnotations = document.getElementById("chk-save-annotations").checked;
    const payload = imageCanvas.toCropPayload(outputDir, saveAnnotations);
    try {
      const res = await Api.crop(payload);
      const annot = res.annotations_path ? `\nAdnotacje: ${res.annotations_path}` : "";
      alert(
        `Zapisano ${res.saved.length} plików.\n` +
        `Wycinek A: ${res.compartments.A}, B: ${res.compartments.B}${annot}`
      );
    } catch (e) {
      alert(`Crop: ${e.message}`);
    }
  });

  // Klawiatura globalna
  document.addEventListener("keydown", (e) => {
    // Ignoruj gdy focus jest w input lub modal otwarty.
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    const backdrop = document.getElementById("modal-backdrop");
    const mkdirBackdrop = document.getElementById("mkdir-backdrop");
    if (!backdrop.hidden || !mkdirBackdrop.hidden) return;

    const key = e.key.toLowerCase();
    if (KEY_TO_MODE[key]) {
      e.preventDefault();
      imageCanvas.setMode(KEY_TO_MODE[key]);
    } else if (e.key === "Escape") {
      imageCanvas.setMode(imageCanvas.mode); // reset selection
    } else if (e.key === "Enter") {
      if (!$btnCrop.disabled) {
        e.preventDefault();
        $btnCrop.click();
      }
    }
  });

  // Initial state render
  updateStatusBar();
  updateActionButtons();
  updateModeButtons();

  // Eksport globalny dla debugowania w DevTools.
  window.Turbot = {
    Modes, BBox, RowLine, Row, Api, fileBrowser, imageCanvas,
    computeRowLabels, computeCompartmentBboxes,
    splitCompartments, splitAutoByLargestGap,
    _runSmokeTests,
  };
});
