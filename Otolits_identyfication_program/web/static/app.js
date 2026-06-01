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
// Bootstrap (na razie tylko FileBrowser; ImageCanvas i mode handling w kolejnych krokach)
// ============================================================================
window.addEventListener("DOMContentLoaded", () => {
  const fileBrowser = new FileBrowser({
    onSelectImage: (path) => {
      console.log("Wybrano obraz:", path);
      // ImageCanvas.loadImage(path) — krok 14
    },
  });
  fileBrowser.cdTo("");

  // Eksport globalny dla debugowania w DevTools.
  window.Turbot = { Modes, BBox, RowLine, Row, Api, fileBrowser };
});
