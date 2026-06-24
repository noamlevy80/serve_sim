"use strict";

// The page is a pure renderer: it fetches the Python-derived view model and
// draws each graph descriptor. No run analysis happens here -- only layout,
// scaling to pixels, colour assignment and hover lookups.

// --- engineering-notation formatter (mirrors serve_sim.viz.graphs.eng_format) ---
const MAGS = [
  [1e15, "P"], [1e12, "T"], [1e9, "G"], [1e6, "M"], [1e3, "K"],
  [1, ""], [1e-3, "m"], [1e-6, "u"], [1e-9, "n"],
];
function formatEng(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "";
  if (v === 0) return "0";
  const sign = v < 0 ? "-" : "";
  const a = Math.abs(v);
  for (const [scale, suf] of MAGS) {
    if (a >= scale) {
      const x = a / scale;
      const t = x >= 100 ? x.toFixed(0) : x.toFixed(1).replace(/\.0$/, "");
      return `${sign}${t}${suf}`;
    }
  }
  const x = a / 1e-9;
  const t = x >= 100 ? x.toFixed(0) : x.toFixed(1).replace(/\.0$/, "");
  return `${sign}${t}n`;
}

// --- deterministic colour per object id (same id => same colour everywhere) -----
function colorFor(key) {
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) >>> 0;
  const hue = h % 360;
  const sat = 55 + (h % 3) * 12;
  const lit = 45 + (h % 4) * 6;
  return `hsl(${hue}, ${sat}%, ${lit}%)`;
}

// --- view model state -----------------------------------------------------------
const STATE = { vm: null, cols: 2, t0: 0, t1: 1, order: [], drag: null,
                hidden: new Set(), treeRefs: { leaves: [], groups: [] } };
const PAD = { l: 40, r: 42, t: 8, b: 14 };

async function boot() {
  await loadRun(null);     // fetch + render the default run
  setupTabs();
  setupControls();
  await setupRunPicker();
  window.addEventListener("resize", () => renderGrid());
}

// Fetch a run's view model (the default when ``run`` is null) and (re)render.
async function loadRun(run) {
  const url = run ? `api/view-model?run=${encodeURIComponent(run)}` : "api/view-model";
  const resp = await fetch(url);
  STATE.vm = await resp.json();
  STATE.t0 = 0;
  STATE.t1 = STATE.vm.makespan_s || 1;
  STATE.order = STATE.vm.graphs.map((g) => g.id);
  STATE.hidden = new Set();
  document.getElementById("run-id").textContent = STATE.vm.run_id;
  const start = document.getElementById("slider-start");
  const end = document.getElementById("slider-end");
  if (start && end) { start.value = 0; end.value = 1000; }
  const label = document.getElementById("window-label");
  if (label) label.textContent = `${formatEng(STATE.t0)}s – ${formatEng(STATE.t1)}s`;
  renderSummary();
  renderTree();
  renderGrid();
}

// Populate the run dropdown and switch runs on selection.
async function setupRunPicker() {
  const sel = document.getElementById("run-select");
  if (!sel) return;
  try {
    const data = await (await fetch("api/runs")).json();
    sel.innerHTML = "";
    for (const name of data.runs) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      if (name === data.current) opt.selected = true;
      sel.appendChild(opt);
    }
    sel.addEventListener("change", () => loadRun(sel.value));
  } catch (e) {
    sel.hidden = true;     // single-run launch with no listing -- hide the picker
  }
}

// --- tabs -----------------------------------------------------------------------
function setupTabs() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
      if (btn.dataset.tab === "timeline") renderGrid();
    });
  });
}

// --- summary --------------------------------------------------------------------
function renderSummary() {
  const host = document.getElementById("summary-tables");
  host.innerHTML = "";
  for (const t of STATE.vm.summary_tables) {
    const table = document.createElement("table");
    table.className = "summary";
    const caption = document.createElement("caption");
    caption.textContent = t.title;
    table.appendChild(caption);
    const thead = document.createElement("thead");
    const hr = document.createElement("tr");
    for (const c of t.columns) {
      const th = document.createElement("th");
      th.textContent = c;
      hr.appendChild(th);
    }
    thead.appendChild(hr);
    table.appendChild(thead);
    const tbody = document.createElement("tbody");
    for (const row of t.rows) {
      const tr = document.createElement("tr");
      for (const cell of row) {
        const td = document.createElement("td");
        td.textContent = cell;
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    host.appendChild(table);
  }
}

// --- timeline controls ----------------------------------------------------------
function setupControls() {
  document.querySelectorAll("#col-selector button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#col-selector button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      STATE.cols = parseInt(btn.dataset.cols, 10);
      const grid = document.getElementById("graph-grid");
      grid.className = `grid cols-${STATE.cols}`;
      renderGrid();
    });
  });

  const start = document.getElementById("slider-start");
  const end = document.getElementById("slider-end");
  const update = () => {
    const span = STATE.vm.makespan_s || 1;
    const a = (parseInt(start.value, 10) / 1000) * span;
    const b = (parseInt(end.value, 10) / 1000) * span;
    let lo = Math.min(a, b);
    let hi = Math.max(a, b);
    if (hi - lo < span / 1000) hi = Math.min(span, lo + span / 1000);
    STATE.t0 = Math.max(0, lo);
    STATE.t1 = Math.min(span, hi);
    document.getElementById("window-label").textContent =
      `${formatEng(STATE.t0)}s – ${formatEng(STATE.t1)}s`;
    renderGrid();
  };
  start.addEventListener("input", update);
  end.addEventListener("input", update);
  update();
}

// --- grid + drag/drop -----------------------------------------------------------
function renderGrid() {
  const grid = document.getElementById("graph-grid");
  if (!document.getElementById("timeline").classList.contains("active")) return;
  grid.className = `grid cols-${STATE.cols}`;
  grid.innerHTML = "";
  const byId = new Map(STATE.vm.graphs.map((g) => [g.id, g]));
  for (const id of STATE.order) {
    const g = byId.get(id);
    if (!g || STATE.hidden.has(id)) continue;
    grid.appendChild(buildCell(g));
  }
}

function buildCell(g) {
  const cell = document.createElement("div");
  cell.className = `cell section-${g.section}`;
  cell.draggable = true;
  cell.dataset.id = g.id;

  const title = document.createElement("div");
  title.className = "cell-title";
  const name = document.createElement("span");
  name.textContent = g.title;
  const unit = document.createElement("span");
  unit.className = "unit";
  unit.textContent = g.unit || "";
  title.append(name, unit);
  cell.appendChild(title);

  const canvas = document.createElement("canvas");
  cell.appendChild(canvas);

  cell.addEventListener("dragstart", () => { STATE.drag = g.id; cell.classList.add("dragging"); });
  cell.addEventListener("dragend", () => { STATE.drag = null; cell.classList.remove("dragging"); });
  cell.addEventListener("dragover", (e) => { e.preventDefault(); cell.classList.add("drop-target"); });
  cell.addEventListener("dragleave", () => cell.classList.remove("drop-target"));
  cell.addEventListener("drop", (e) => {
    e.preventDefault();
    cell.classList.remove("drop-target");
    reorder(STATE.drag, g.id);
  });

  requestAnimationFrame(() => drawGraph(canvas, g));
  return cell;
}

function reorder(fromId, toId) {
  if (!fromId || fromId === toId) return;
  const order = STATE.order.filter((id) => id !== fromId);
  const idx = order.indexOf(toId);
  order.splice(idx, 0, fromId);
  STATE.order = order;
  renderGrid();
}

// --- graph selection tree -------------------------------------------------------
function renderTree() {
  const panel = document.getElementById("graph-tree");
  panel.innerHTML = "";
  STATE.treeRefs = { leaves: [], groups: [] };
  for (const cat of STATE.vm.graph_tree) {
    // Graph-type branches start collapsed so the panel opens as the flat list of
    // graph types (one layer above the device/sequence leaves).
    const groupBranches = cat.children.map((grp) =>
      makeBranch(grp.label, grp.graphs.map(makeLeaf), true));
    panel.appendChild(makeBranch(cat.label, groupBranches));
  }
  refreshTree();
}

function makeLeaf(g) {
  const row = document.createElement("div");
  row.className = "tw-leaf";
  const name = document.createElement("span");
  name.className = "tw-name";
  name.textContent = g.label;
  name.title = g.label;
  name.addEventListener("click", () => {
    if (STATE.hidden.has(g.id)) STATE.hidden.delete(g.id);
    else STATE.hidden.add(g.id);
    refreshTree();
    renderGrid();
  });
  row.appendChild(name);
  row._ids = [g.id];
  STATE.treeRefs.leaves.push({ id: g.id, el: name });
  return row;
}

function makeBranch(label, childRows, collapsed = false) {
  const ids = childRows.flatMap((r) => r._ids);
  const branch = document.createElement("div");
  branch.className = "tw-branch";

  const head = document.createElement("div");
  head.className = "tw-head";
  const toggle = document.createElement("span");
  toggle.className = "tw-toggle";
  toggle.textContent = collapsed ? "+" : "-";
  const name = document.createElement("span");
  name.className = "tw-name tw-group";
  name.textContent = label;
  name.title = label;

  const kids = document.createElement("div");
  kids.className = "tw-children";
  if (collapsed) kids.classList.add("collapsed");
  childRows.forEach((r) => kids.appendChild(r));

  toggle.addEventListener("click", () => {
    const collapsed = kids.classList.toggle("collapsed");
    toggle.textContent = collapsed ? "+" : "-";
  });
  // Clicking the hierarchy name forces every graph under it on or off,
  // overriding any per-graph choices made inside it.
  name.addEventListener("click", () => {
    const anyVisible = ids.some((id) => !STATE.hidden.has(id));
    if (anyVisible) ids.forEach((id) => STATE.hidden.add(id));
    else ids.forEach((id) => STATE.hidden.delete(id));
    refreshTree();
    renderGrid();
  });

  head.append(toggle, name);
  branch.append(head, kids);
  branch._ids = ids;
  STATE.treeRefs.groups.push({ ids, el: name });
  return branch;
}

function refreshTree() {
  for (const { id, el } of STATE.treeRefs.leaves) {
    el.classList.toggle("off", STATE.hidden.has(id));
  }
  for (const { ids, el } of STATE.treeRefs.groups) {
    const hidden = ids.filter((id) => STATE.hidden.has(id)).length;
    el.classList.toggle("off", hidden === ids.length && ids.length > 0);
    el.classList.toggle("partial", hidden > 0 && hidden < ids.length);
  }
}

// --- drawing --------------------------------------------------------------------
function prepCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 300;
  const h = canvas.clientHeight || 120;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function plotRect(w, h) {
  return { x: PAD.l, y: PAD.t, w: w - PAD.l - PAD.r, h: h - PAD.t - PAD.b };
}

function timeToX(t, rect) {
  const span = STATE.t1 - STATE.t0 || 1;
  return rect.x + ((t - STATE.t0) / span) * rect.w;
}

function drawAxes(ctx, rect) {
  ctx.strokeStyle = "#2a2f3d";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(rect.x, rect.y);
  ctx.lineTo(rect.x, rect.y + rect.h);
  ctx.lineTo(rect.x + rect.w, rect.y + rect.h);
  ctx.stroke();
}

function clipBuckets(buckets) {
  return buckets.filter((b) => b[1] > STATE.t0 && b[0] < STATE.t1);
}

function drawGraph(canvas, g) {
  const { ctx, w, h } = prepCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const rect = plotRect(w, h);
  if (g.kind === "value") drawValue(ctx, rect, g);
  else if (g.kind === "stacked") drawStacked(ctx, rect, g);
  else if (g.kind === "discrete") drawDiscrete(ctx, rect, g);
  drawAxes(ctx, rect);
  attachHover(canvas, g, rect);
}

function labelText(ctx, text, x, y, color, align) {
  ctx.fillStyle = color;
  ctx.font = "10px monospace";
  ctx.textAlign = align || "left";
  ctx.textBaseline = "alphabetic";
  ctx.fillText(text, x, y);
}

function drawValue(ctx, rect, g) {
  const buckets = clipBuckets(g.buckets);
  let dataMax = 0;
  for (const b of buckets) dataMax = Math.max(dataMax, b[2]);
  if (dataMax <= 0) dataMax = 1;
  const yOf = (v) => rect.y + rect.h - (v / dataMax) * rect.h;

  ctx.fillStyle = colorFor(g.id) + "";
  ctx.globalAlpha = 0.35;
  for (const [t0, t1, v] of buckets) {
    const x0 = timeToX(Math.max(t0, STATE.t0), rect);
    const x1 = timeToX(Math.min(t1, STATE.t1), rect);
    const y = yOf(v);
    ctx.fillRect(x0, y, Math.max(1, x1 - x0), rect.y + rect.h - y);
  }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = colorFor(g.id);
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  let first = true;
  for (const [t0, t1, v] of buckets) {
    const x0 = timeToX(Math.max(t0, STATE.t0), rect);
    const x1 = timeToX(Math.min(t1, STATE.t1), rect);
    const y = yOf(v);
    if (first) { ctx.moveTo(x0, y); first = false; } else { ctx.lineTo(x0, y); }
    ctx.lineTo(x1, y);
  }
  ctx.stroke();

  // Non-autoscaling max-value line (off-screen when max >> data).
  if (g.max_value) {
    const y = yOf(g.max_value);
    if (y >= rect.y - 1) {
      ctx.strokeStyle = "#e06c75aa";
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(rect.x, y);
      ctx.lineTo(rect.x + rect.w, y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // Left: absolute peak; right: relative to the static max.
  labelText(ctx, formatEng(dataMax), rect.x - 2, rect.y + 8, "#9aa3b2", "right");
  if (g.max_value) {
    labelText(ctx, formatEng(dataMax / g.max_value), rect.x + rect.w + 2,
              rect.y + 8, "#9aa3b2", "left");
  }
}

function drawStacked(ctx, rect, g) {
  const buckets = clipBuckets(g.buckets);
  let dataMax = 0;
  for (const b of buckets) {
    let sum = 0;
    for (const k of g.keys) sum += b[2][k] || 0;
    dataMax = Math.max(dataMax, sum);
  }
  if (dataMax <= 0) dataMax = 1;
  const hOf = (v) => (v / dataMax) * rect.h;

  for (const [t0, t1, content] of buckets) {
    const x0 = timeToX(Math.max(t0, STATE.t0), rect);
    const x1 = timeToX(Math.min(t1, STATE.t1), rect);
    let yBase = rect.y + rect.h;
    for (const k of g.keys) {
      const v = content[k] || 0;
      if (v <= 0) continue;
      const bh = hOf(v);
      ctx.fillStyle = colorFor(k);
      ctx.fillRect(x0, yBase - bh, Math.max(1, x1 - x0), bh);
      yBase -= bh;
    }
  }
  if (g.max_value) {
    const y = rect.y + rect.h - hOf(g.max_value);
    if (y >= rect.y - 1) {
      ctx.strokeStyle = "#e06c75aa";
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(rect.x, y);
      ctx.lineTo(rect.x + rect.w, y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  labelText(ctx, formatEng(dataMax), rect.x - 2, rect.y + 8, "#9aa3b2", "right");
  if (g.max_value) {
    labelText(ctx, formatEng(dataMax / g.max_value), rect.x + rect.w + 2,
              rect.y + 8, "#9aa3b2", "left");
  }
}

function drawDiscrete(ctx, rect, g) {
  ctx.font = "10px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const yMid = rect.y + rect.h / 2;
  for (const [t0, t1, abbrev, , key] of g.segments) {
    if (t1 <= STATE.t0 || t0 >= STATE.t1) continue;
    const x0 = timeToX(Math.max(t0, STATE.t0), rect);
    const x1 = timeToX(Math.min(t1, STATE.t1), rect);
    const wseg = Math.max(1, x1 - x0);
    ctx.fillStyle = colorFor(key);
    ctx.fillRect(x0, rect.y + 2, wseg, rect.h - 4);
    if (wseg > 18 && abbrev) {
      ctx.fillStyle = "#0b0d12";
      ctx.fillText(abbrev, x0 + wseg / 2, yMid);
    }
  }
}

// --- hover tooltips -------------------------------------------------------------
function attachHover(canvas, g, rect) {
  const tip = document.getElementById("tooltip");
  canvas.onmousemove = (e) => {
    const r = canvas.getBoundingClientRect();
    const px = e.clientX - r.left;
    if (px < rect.x || px > rect.x + rect.w) { tip.hidden = true; return; }
    const span = STATE.t1 - STATE.t0 || 1;
    const t = STATE.t0 + ((px - rect.x) / rect.w) * span;
    let text = null;
    if (g.kind === "discrete") {
      const seg = g.segments.find((s) => t >= s[0] && t < s[1]);
      if (seg) {
        const span = `${formatEng(seg[0])}s – ${formatEng(seg[1])}s`;
        text = seg[3] ? `${span}\n${seg[3]}` : span;
      }
    } else {
      const b = g.buckets.find((s) => t >= s[0] && t < s[1]);
      if (b) {
        if (g.kind === "stacked") {
          const parts = g.keys.filter((k) => (b[2][k] || 0) > 0)
            .map((k) => `${k}: ${formatEng(b[2][k])}`);
          text = `t=${formatEng(t)}s\n` + parts.join("\n");
        } else {
          text = `t=${formatEng(t)}s\n${formatEng(b[2])} ${g.unit || ""}`;
        }
      }
    }
    if (!text) { tip.hidden = true; return; }
    tip.textContent = text;
    tip.style.whiteSpace = "pre";
    tip.style.left = `${e.clientX + 12}px`;
    tip.style.top = `${e.clientY + 12}px`;
    tip.hidden = false;
  };
  canvas.onmouseleave = () => { document.getElementById("tooltip").hidden = true; };
}

boot();
