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
const STATE = { vm: null, cols: 4, t0: 0, t1: 1, order: [], drag: null,
                hidden: new Set(), treeRefs: { leaves: [], groups: [] } };
const PAD = { l: 40, r: 42, t: 8, b: 14 };

// Compute-device graph types shown by default; every other graph starts hidden.
const DEFAULT_VISIBLE_TYPES = new Set(["compute", "bandwidth", "reason", "out_tps"]);
function isDefaultVisible(id) {
  if (!id.startsWith("dev:")) return false;
  return DEFAULT_VISIBLE_TYPES.has(id.slice(id.lastIndexOf(":") + 1));
}

async function boot() {
  await loadRun(null);     // fetch + render the default run
  setupTabs();
  setupControls();
  await setupRunPicker();
  window.addEventListener("resize", () => { renderGrid(); renderWorkload(); });
}

// Fetch a run's view model (the default when ``run`` is null) and (re)render.
async function loadRun(run) {
  const url = run ? `api/view-model?run=${encodeURIComponent(run)}` : "api/view-model";
  const resp = await fetch(url);
  STATE.vm = await resp.json();
  STATE.t0 = 0;
  STATE.t1 = STATE.vm.makespan_s || 1;
  STATE.order = STATE.vm.graphs.map((g) => g.id);
  STATE.hidden = new Set(
    STATE.vm.graphs.filter((g) => !isDefaultVisible(g.id)).map((g) => g.id));
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
      if (btn.dataset.tab === "workload") renderWorkload();
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

// --- workload tab ---------------------------------------------------------------
// A node-link graph: each turn is an input (prefill) and output (decode) block,
// tool calls fill the gaps, and edges run from reused KV to the reusing sequence.
const WL = { laneH: 34, padL: 96, padR: 24, padT: 22, padB: 8, rects: [] };
// Darker block palette so the lighter, bolder reuse edges stand out on top.
const WL_FILL = { prefill: "#1f4e9c", decode: "#1f7a3a", tool: "#8a6212" };

function wlTimeToX(t, geom) {
  const span = geom.span || 1;
  return geom.x + (t / span) * geom.w;
}

function renderWorkload() {
  const panel = document.getElementById("workload");
  if (!panel || !panel.classList.contains("active")) return;
  const wg = STATE.vm && STATE.vm.workload_graph;
  const canvas = document.getElementById("workload-canvas");
  if (!canvas || !wg) return;

  // Size lanes so ~16 workloads fill one screen height (double-spaced); taller
  // runs simply scroll within the panel.
  WL.laneH = Math.max(36, Math.round((window.innerHeight - 150) / 16));

  const lanes = Math.max(1, wg.num_lanes || 0);
  const body = canvas.parentElement;
  const cssW = body.clientWidth || 800;
  const cssH = WL.padT + WL.padB + lanes * WL.laneH;
  canvas.style.width = cssW + "px";
  canvas.style.height = cssH + "px";
  const dpr = window.devicePixelRatio || 1;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const geom = {
    x: WL.padL, w: Math.max(20, cssW - WL.padL - WL.padR),
    span: wg.makespan_s || 1,
  };

  // Lane backgrounds and workload labels.
  ctx.font = "10px monospace";
  ctx.textBaseline = "middle";
  const laneLabel = {};
  for (const n of wg.nodes) if (!(n.lane in laneLabel)) laneLabel[n.lane] = n.workload;
  for (let i = 0; i < lanes; i++) {
    const top = WL.padT + i * WL.laneH;
    if (i % 2 === 0) {
      ctx.fillStyle = "#161a23";
      ctx.fillRect(0, top, cssW, WL.laneH);
    }
    ctx.fillStyle = "#9aa3b2";
    ctx.textAlign = "left";
    ctx.fillText(laneLabel[i] || "", 6, top + WL.laneH / 2);
  }

  // Time-axis ticks across the top.
  for (let k = 0; k <= 5; k++) {
    const t = (geom.span * k) / 5;
    const x = wlTimeToX(t, geom);
    ctx.strokeStyle = "#2a2f3d";
    ctx.globalAlpha = 0.25;
    ctx.beginPath();
    ctx.moveTo(x, WL.padT - 4);
    ctx.lineTo(x, cssH - WL.padB);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#6b7280";
    ctx.textAlign = "center";
    ctx.fillText(`${formatEng(t)}s`, x, WL.padT - 11);
  }

  // Node rectangles (remembered for hover and edge endpoints).
  WL.rects = [];
  const byId = {};
  const nodeH = WL.laneH * 0.42;
  for (const n of wg.nodes) {
    const top = WL.padT + n.lane * WL.laneH;
    const cy = top + WL.laneH / 2;
    const x0 = wlTimeToX(n.t0, geom);
    const x1 = wlTimeToX(n.t1, geom);
    const w = Math.max(2, x1 - x0);
    const rect = { x: x0, y: cy - nodeH / 2, w, h: nodeH, node: n, cx: x0 + w / 2, cy };
    WL.rects.push(rect);
    byId[n.id] = rect;
  }

  // Nodes first.
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (const rect of WL.rects) {
    const n = rect.node;
    ctx.fillStyle = WL_FILL[n.kind] || "#6b7280";
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    if (n.group) {
      ctx.strokeStyle = colorFor(`group:${n.group}`);
      ctx.lineWidth = 1.5;
      ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.w - 1, rect.h - 1);
    }
    if (rect.w > 26 && n.text) {
      ctx.fillStyle = "#eef2f8";
      ctx.font = "9px sans-serif";
      ctx.fillText(n.text, rect.cx, rect.cy);
    }
  }

  // Edges on top (in front of the nodes) so they stay visible: existing KV ->
  // reusing sequence. Each edge gets a slightly varied hue/brightness so
  // overlapping links are easier to follow.
  for (const e of wg.edges) {
    const s = byId[e.source];
    const d = byId[e.target];
    if (!s || !d) continue;
    drawWlEdge(ctx, s.x + s.w, s.cy, d.x, d.cy, wlEdgeColor(`${e.source}->${e.target}`));
  }

  attachWlHover(canvas);
}

// Deterministic per-edge variation around the base reuse-edge orange so the
// lines differ slightly in hue and brightness without changing identity.
function wlEdgeColor(key) {
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) >>> 0;
  const hue = 24 + ((h % 31) - 15);          // base 24deg +/- 15
  const light = 66 + (((h >> 5) % 23) - 11); // base 66% +/- 11
  return `hsl(${hue}, 100%, ${light}%)`;
}

function drawWlEdge(ctx, x0, y0, x1, y1, color) {
  const stroke = color || "#ff9d5c";
  ctx.strokeStyle = stroke;
  ctx.fillStyle = stroke;
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  // Horizontal tangents at both ends: every edge leaves the source's right side
  // heading right and approaches the target's left side heading right, so the
  // picture reads consistently regardless of relative position.
  const reach = Math.max(24, Math.abs(x1 - x0) * 0.4);
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.bezierCurveTo(x0 + reach, y0, x1 - reach, y1, x1, y1);
  ctx.stroke();
  // Arrowhead entering the target's left side, pointing right.
  const a = 8;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - a, y1 - a * 0.6);
  ctx.lineTo(x1 - a, y1 + a * 0.6);
  ctx.closePath();
  ctx.fill();
}

function attachWlHover(canvas) {
  const tip = document.getElementById("tooltip");
  canvas.onmousemove = (e) => {
    const r = canvas.getBoundingClientRect();
    const px = e.clientX - r.left;
    const py = e.clientY - r.top;
    let hit = null;
    for (const rect of WL.rects) {
      if (px >= rect.x && px <= rect.x + rect.w && py >= rect.y && py <= rect.y + rect.h) {
        hit = rect.node;
        break;
      }
    }
    if (!hit) { tip.hidden = true; return; }
    const lines = [`${hit.sequence} (${hit.kind})`];
    if (hit.kind === "tool") {
      lines.push(hit.desc || "Tool call");
    } else {
      lines.push(`${formatEng(hit.tokens)} tokens`);
      if (hit.group) lines.push(`group ${hit.group}`);
      if (hit.kind === "decode" && hit.tps != null)
        lines.push(`TPS ${formatEng(hit.tps)} tok/s`);
      if (hit.kind === "prefill" && hit.ttft_s != null)
        lines.push(`TTFT ${formatEng(hit.ttft_s)}s`);
    }
    tip.textContent = lines.join("\n");
    tip.style.whiteSpace = "pre";
    tip.style.left = `${e.clientX + 12}px`;
    tip.style.top = `${e.clientY + 12}px`;
    tip.hidden = false;
  };
  canvas.onmouseleave = () => { document.getElementById("tooltip").hidden = true; };
}

boot();
