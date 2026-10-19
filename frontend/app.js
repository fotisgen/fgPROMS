/* ============ fgPROMS frontend application ============ */
"use strict";

let LANG = localStorage.getItem("fgproms-lang") || "el";
const D = () => I18N[LANG];
const t = (k) => (D()[k] !== undefined ? D()[k] : k);
const tm = (code, args) => {
  const tpl = (MSG[LANG] && MSG[LANG][code]) || MSG.en[code] || code;
  return tpl.replace(/\{(\w+)\}/g, (_, k) =>
    args && args[k] !== undefined ? String(args[k]) : "");
};

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function fmt(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v !== "number" || !isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-4)) return v.toExponential(3);
  return String(+v.toPrecision(6));
}

let editor = null;
let lastRun = null;
let currentFile = "untitled.gPROMS";
let runCount = 0;

/* ---------------- init ---------------- */
function init() {
  editor = CodeMirror.fromTextArea($("code"), {
    mode: "gproms",
    lineNumbers: true,
    indentUnit: 2,
    tabSize: 2,
    lineWrapping: false,
  });
  bindEvents();
  applyI18n();
  setStatus(t("statusReady"));
  loadExamples().then(() => loadExample("cstr.gPROMS"));
  loadProjects();
}

function bindEvents() {
  document.querySelectorAll(".tab").forEach((b) => {
    b.addEventListener("click", () => switchTab(b.dataset.tab));
  });
  $("btnRun").addEventListener("click", runModel);
  $("btnLang").addEventListener("click", toggleLang);
  $("btnOpen").addEventListener("click", () => $("fileInput").click());
  $("btnSave").addEventListener("click", saveFile);
  $("btnSaveAs").addEventListener("click", saveAsProject);
  $("btnDeleteProject").addEventListener("click", deleteProject);
  $("fileInput").addEventListener("change", openFile);
  $("exampleSel").addEventListener("change", (e) => {
    if (e.target.value) loadExample(e.target.value);
  });
  $("projectSel").addEventListener("change", (e) => {
    if (e.target.value) loadProject(e.target.value);
  });
}

/* ---------------- i18n ---------------- */
function toggleLang() {
  LANG = LANG === "el" ? "en" : "el";
  localStorage.setItem("fgproms-lang", LANG);
  applyI18n();
  if (lastRun) rerenderAll();
  const sel = $("exampleSel");
  sel.options[0].textContent = t("examples");
}

function applyI18n() {
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    el.textContent = t(el.getAttribute("data-i18n"));
  });
  $("btnLang").textContent = D().langBtn;
  document.title = "fgPROMS";
  const sel = $("exampleSel");
  if (sel.options.length) sel.options[0].textContent = t("examples");
  const psel = $("projectSel");
  if (psel.options.length) psel.options[0].textContent = t("projects");
}

/* ---------------- tabs ---------------- */
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab-body").forEach((b) =>
    b.classList.toggle("active", b.id === "tab-" + name));
}

/* ---------------- status ---------------- */
function setStatus(text) {
  $("statusBar").textContent = text;
}

/* ---------------- examples / files ---------------- */
async function loadExamples() {
  try {
    const res = await fetch("/api/examples");
    const names = await res.json();
    const sel = $("exampleSel");
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = t("examples");
    sel.appendChild(opt);
    names.forEach((n) => {
      const o = document.createElement("option");
      o.value = n;
      o.textContent = n;
      sel.appendChild(o);
    });
  } catch (e) {
    setStatus(t("loadErr"));
  }
}

async function loadExample(name) {
  try {
    const res = await fetch("/api/examples/" + encodeURIComponent(name));
    const data = await res.json();
    editor.setValue(data.code);
    currentFile = name;
    $("fileName").textContent = name;
    setStatus(t("statusReady"));
  } catch (e) {
    setStatus(t("loadErr"));
  }
}

function openFile(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    editor.setValue(String(reader.result));
    currentFile = file.name;
    $("fileName").textContent = file.name;
    setStatus(t("statusReady"));
  };
  reader.readAsText(file);
  e.target.value = "";
}

function saveFile() {
  const blob = new Blob([editor.getValue()], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = currentFile || "model.gPROMS";
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ---------------- projects ---------------- */
async function loadProjects() {
  try {
    const res = await fetch("/api/projects");
    const names = await res.json();
    const sel = $("projectSel");
    sel.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = t("projects");
    sel.appendChild(opt);
    names.forEach((n) => {
      const o = document.createElement("option");
      o.value = n;
      o.textContent = n;
      sel.appendChild(o);
    });
  } catch (e) {
    setStatus(t("loadErr"));
  }
}

async function loadProject(name) {
  try {
    const res = await fetch("/api/projects/" + encodeURIComponent(name));
    const data = await res.json();
    editor.setValue(data.code);
    currentFile = name;
    $("fileName").textContent = name;
    setStatus(t("statusReady"));
  } catch (e) {
    setStatus(t("loadErr"));
  }
}

async function saveAsProject() {
  const name = prompt(t("promptProjectName"), currentFile || "model.gPROMS");
  if (!name) return;
  try {
    const res = await fetch("/api/projects/" + encodeURIComponent(name), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: editor.getValue() }),
    });
    const data = await res.json();
    if (!data.ok) throw new Error("save failed");
    currentFile = name;
    $("fileName").textContent = name;
    setStatus(t("savedOk") + name);
    await loadProjects();
  } catch (e) {
    setStatus(t("loadErr"));
  }
}

async function deleteProject() {
  const name = $("projectSel").value;
  if (!name) return;
  if (!confirm(t("confirmDelete").replace("{name}", name))) return;
  try {
    await fetch("/api/projects/" + encodeURIComponent(name), { method: "DELETE" });
    setStatus(t("deletedOk") + name);
    await loadProjects();
  } catch (e) {
    setStatus(t("loadErr"));
  }
}

/* ---------------- CSV export ---------------- */
async function exportCsv() {
  if (!lastRun) return;
  try {
    const res = await fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        results: lastRun.results,
        timeSeries: lastRun.timeSeries,
      }),
    });
    const data = await res.json();
    const blob = new Blob([data.csv], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = (currentFile || "model").replace(/\.gPROMS$/i, "") + ".csv";
    a.click();
    URL.revokeObjectURL(a.href);
    setStatus(t("csvDownloaded") + a.download);
  } catch (e) {
    setStatus(t("loadErr"));
  }
}

/* ---------------- flowsheet diagram ---------------- */
function flowsheetDiagram(m) {
  const units = m.units || [];
  if (!units.length) return "";
  const W = 150, H = 54, GX = 110;
  const xs = {};
  units.forEach((u, i) => { xs[u.name] = 30 + i * (W + GX); });
  const edges = [];
  for (const e of m.equations) {
    if (e.kind !== "eq") continue;
    const mm = e.text.match(/([A-Za-z_][\w]*)\.\w[\w.]*\s*=\s*([A-Za-z_][\w]*)\.\w/);
    if (mm && xs[mm[1]] !== undefined && xs[mm[2]] !== undefined) {
      edges.push([mm[2], mm[1]]);
    }
  }
  const y = 34;
  const totalW = 30 + units.length * (W + GX) + 60;
  let svg = `<svg width="${totalW}" height="160" viewBox="0 0 ${totalW} 160" style="max-width:100%">`;
  const cy = 126;
  edges.forEach((pair, i) => {
    const x1 = xs[pair[0]] + W / 2;
    const x2 = xs[pair[1]] + W / 2;
    const mx = (x1 + x2) / 2;
    const my = cy + 22 * (i % 2);
    svg += `<path d="M ${x1} ${y + H} Q ${mx} ${my} ${x2} ${y + H}" fill="none" stroke="#38bdf8" stroke-width="1.5"/>`;
    svg += `<circle cx="${mx}" cy="${my}" r="3" fill="#38bdf8"/>`;
  });
  for (const u of units) {
    const x = xs[u.name];
    svg += `<rect x="${x}" y="${y}" width="${W}" height="${H}" rx="8" fill="#16223a" stroke="#38bdf8" stroke-width="1.5"/>`;
    svg += `<text x="${x + W / 2}" y="${y + 23}" text-anchor="middle" fill="#dbe4f0" font-size="13" font-weight="600">${esc(u.name)}</text>`;
    svg += `<text x="${x + W / 2}" y="${y + 40}" text-anchor="middle" fill="#8ea0bd" font-size="11">${esc(u.type)}</text>`;
  }
  svg += "</svg>";
  return svg;
}

/* ---------------- run ---------------- */
async function runModel() {
  setStatus(t("statusRunning"));
  lastRun = null;
  let data;
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: editor.getValue() }),
    });
    data = await res.json();
  } catch (e) {
    setStatus(t("loadErr"));
    return;
  }
  lastRun = data;
  renderModelTab(data);
  renderResultsTab(data);
  renderMessagesTab(data);
  renderPlotsTab(data);
  const hasErr = (data.messages || []).some((m) => m.level === "error");
  if (hasErr) switchTab("messages");
  else if (data.results) switchTab("results");
  else switchTab("model");
  setStatus(
    t("statusDone") + " · " + (data.messages || []).length + " " +
    t("tab_messages").toLowerCase());
  // completion marker (also used by the GUI test suite to detect that the
  // whole run — renders and tab switch included — has finished)
  document.body.dataset.runCount = String(++runCount);
}

function rerenderAll() {
  if (!lastRun) return;
  renderModelTab(lastRun);
  renderResultsTab(lastRun);
  renderMessagesTab(lastRun);
  renderPlotsTab(lastRun);
}

/* ---------------- model tab ---------------- */
function renderModelTab(d) {
  const el = $("tab-model");
  const m = d.model;
  if (!m) {
    el.innerHTML = `<div class="note">${esc(t("noResults"))}</div>`;
    return;
  }
  const df = m.dof;
  const modeText = m.isProcess ? t("flowsheet")
    : (m.objective ? t("optimization")
      : (m.estimated && m.estimated.length ? t("estimation")
        : (m.isDynamic ? t("dynamic") : t("steady"))));
  let html = `<h3>${t("modelName")}: ${esc(m.name)}</h3>
  <div class="cards">
    <div class="card"><div class="card-num">${df.unknowns}</div><div class="card-label">${t("unknowns")}</div></div>
    <div class="card"><div class="card-num">${df.equations}</div><div class="card-label">${t("eqCount")}</div></div>
    <div class="card ${df.dof === 0 ? "ok" : "bad"}"><div class="card-num">${df.dof}</div><div class="card-label">${t("dof")}</div></div>
    <div class="card"><div class="card-num" style="font-size:15px">${modeText}</div><div class="card-label">${t("mode")}</div></div>
  </div>`;

  if (m.isProcess && m.units && m.units.length) {
    html += `<h3>${t("unitsTitle")} (${m.units.length})</h3>
      <div class="flowsheet">${flowsheetDiagram(m)}</div>
      <table class="grid"><thead><tr><th>${t("name")}</th><th>${t("unitType")}</th></tr></thead><tbody>`;
    for (const u of m.units)
      html += `<tr><td>${esc(u.name)}</td><td class="muted">${esc(u.type)}</td></tr>`;
    html += "</tbody></table>";
  }
  if (m.isProcess && m.streams && m.streams.length) {
    html += `<h3>${t("streamsTitle")} (${m.streams.length})</h3>
      <table class="grid"><thead><tr><th>${t("name")}</th><th>${t("fluid")}</th></tr></thead><tbody>`;
    for (const s of m.streams)
      html += `<tr><td>${esc(s.name)}</td><td class="muted">${esc(s.fluid)}</td></tr>`;
    html += "</tbody></table>";
  }

  html += `<h3>${t("parameters")}</h3>
  <table class="grid"><thead><tr><th>${t("name")}</th><th>${t("type")}</th><th>${t("defaultValue")}</th><th>${t("value")}</th></tr></thead><tbody>`;
  for (const p of m.parameters)
    html += `<tr><td>${esc(p.name)}</td><td class="muted">${esc(p.type)}</td><td class="muted">${esc(p.default ?? "—")}</td><td>${fmt(p.value)}</td></tr>`;
  html += "</tbody></table>";

  html += `<h3>${t("variables")}</h3>
  <table class="grid"><thead><tr><th>${t("name")}</th><th>${t("type")}</th><th>${t("guess")}</th><th>${t("fixed")}</th><th>${t("state")}</th></tr></thead><tbody>`;
  for (const v of m.variables)
    html += `<tr><td>${esc(v.name)}</td><td class="muted">${esc(v.type)}</td><td class="muted">${fmt(v.guess)}</td><td>${v.fixed ? "✓" : ""}</td><td>${v.isState ? t("stateYes") : ""}</td></tr>`;
  html += "</tbody></table>";

  html += `<h3>${t("equations")} <span class="muted">(${m.equations.length})</span></h3><ol class="eqlist">`;
  for (const e of m.equations) {
    const chips = (e.derivatives || [])
      .map((v) => `<span class="chip der">$${esc(v)}</span>`)
      .concat((e.variables || []).map((v) => `<span class="chip">${esc(v)}</span>`))
      .join("");
    const idTag = e.id !== null && e.id !== undefined ? ` <span class="muted">#${e.id}</span>` : "";
    html += `<li class="${esc(e.kind)}" style="margin-left:${(e.indent || 0) * 16}px"><code>${idTag} ${esc(e.text)}</code><div class="chips">${chips}</div></li>`;
  }
  html += "</ol>";

  html += `<h3>${t("schedule")}</h3><div class="schedule">${
    m.schedule
      ? esc(`${m.schedule.var} : ${m.schedule.t0} → ${m.schedule.t1}  (${t("stepLabel")} ${m.schedule.step})`)
      : t("noSchedule")
  }</div>`;

  if (m.objective) {
    html += `<h3>${t("objectiveTitle")}</h3>
      <div class="schedule">${t(m.objective.kind)}: <code>${esc(m.objective.text)}</code></div>`;
  }
  if (m.bounds && m.bounds.length) {
    html += `<h3>${t("boundsTitle")}</h3>
    <table class="grid"><thead><tr><th>${t("name")}</th><th>${t("lower")}</th><th>${t("upper")}</th></tr></thead><tbody>`;
    for (const b of m.bounds)
      html += `<tr><td>${esc(b.name)}</td><td>${fmt(b.lo)}</td><td>${fmt(b.hi)}</td></tr>`;
    html += "</tbody></table>";
  }

  el.innerHTML = html;
}

/* ---------------- results tab ---------------- */
function renderResultsTab(d) {
  const el = $("tab-results");
  const r = d.results;
  if (!r) {
    el.innerHTML = `<div class="note">${esc(t("noResults"))}</div>`;
    return;
  }
  const objText = r.mode === "optimization" && r.objective
    ? ` — ${t("objectiveTitle")} (${t(r.objective.kind)}): ${fmt(r.objective.value)}`
    : "";
  let html = `<div class="summary ${r.success ? "ok" : "bad"}">${
    r.success ? "✓ " + t("success") : "✗ " + t("failure")
  } — ${t("solver")}: ${esc(r.method || "—")}${
    r.maxResidual !== undefined && r.maxResidual !== null
      ? " · " + t("maxResidual") + ": " + fmt(r.maxResidual)
      : ""
  }${objText}</div>`;

    if (r.values) {
    html += `<h3>${t(r.mode === "dynamic" ? "finalValues" : "valuesTitle")}</h3>
    <table class="grid"><thead><tr><th>${t("name")}</th><th>${t("value")}</th></tr></thead><tbody>`;
    const vars = d.model ? d.model.variables : [];
    for (const v of vars) {
      const val = r.values[v.name.toLowerCase()];
      html += `<tr><td>${esc(v.name)}</td><td>${val !== undefined ? fmt(val) : "—"}</td></tr>`;
    }
    html += "</tbody></table>";
  }

  if (r.mode === "estimation" && r.estimated) {
    html += `<h3>${t("estimatedTitle")}</h3>
    <table class="grid"><thead><tr><th>${t("name")}</th><th>${t("value")}</th></tr></thead><tbody>`;
    const pnames = {};
    if (d.model) for (const p of d.model.parameters) pnames[p.name.toLowerCase()] = p.name;
    for (const [k, v] of Object.entries(r.estimated))
      html += `<tr><td>${esc(pnames[k] || k)}</td><td>${fmt(v)}</td></tr>`;
    html += `</tbody></table>`;
    html += `<div class="summary ok">${t("sseLabel")}: ${fmt(r.sse)}</div>`;
  }

  if (r.mode === "estimation" && r.dataFit && r.dataFit.length) {
    html += `<h3>${t("fitTitle")}</h3>
    <table class="grid"><thead><tr><th>${t("variable")}</th><th>${t("time")}</th><th>${t("measured")}</th><th>${t("modelVal")}</th></tr></thead><tbody>`;
    for (const f of r.dataFit)
      html += `<tr><td>${esc(f.var)}</td><td>${fmt(f.t)}</td><td>${fmt(f.measured)}</td><td>${fmt(f.model)}</td></tr>`;
    html += "</tbody></table>";
  }

  if (r.residuals && r.residuals.length) {
    html += `<h3>${t("eqResiduals")}</h3>
    <table class="grid"><thead><tr><th>${t("eqId")}</th><th>${t("residual")}</th></tr></thead><tbody>`;
    for (const rr of r.residuals)
      html += `<tr><td>#${rr.id}</td><td>${fmt(rr.value)}</td></tr>`;
    html += "</tbody></table>";
  }
  html += `<div style="margin-top:14px"><button id="btnCsv" class="btn">${t("exportCsv")}</button></div>`;
  el.innerHTML = html;
  const btnCsv = el.querySelector("#btnCsv");
  if (btnCsv) btnCsv.addEventListener("click", exportCsv);
}

/* ---------------- plots tab ---------------- */
function baseGroups(values) {
  const groups = {};
  for (const k of Object.keys(values || {})) {
    const m = k.match(/^(.+)\((\d+)\)$/);
    if (m) {
      const b = m[1];
      (groups[b] = groups[b] || {})[parseInt(m[2])] = values[k];
    } else {
      groups[k] = values[k];
    }
  }
  return groups;
}

function renderPlotsTab(d) {
  const el = $("tab-plots");
  const ts = d.timeSeries;
  if (ts) {
    const names = Object.keys(ts.variables);
    let html = `<div class="plot-controls"><span class="label">${t("selectVariables")}:</span>
    <div class="checkbox-grid">`;
    for (const n of names)
      html += `<label class="check"><input type="checkbox" class="pv" value="${esc(n)}" checked> ${esc(n)}</label>`;
    html += `</div><button id="btnPlot" class="btn">${t("plot")}</button></div>
    <div id="plotArea" style="height:430px"></div>`;
    el.innerHTML = html;
    $("btnPlot").addEventListener("click", () => drawPlot(ts));
    loadPlotly().then(() => drawPlot(ts)).catch(() => {});
    return;
  }

  // x-y plot for array results (phase envelopes, profiles, ...)
  const groups = baseGroups(d.results ? d.results.values : null);
  const arrayBases = Object.keys(groups).filter(
    (b) => typeof groups[b] === "object");
  if (arrayBases.length >= 2) {
    let html = `<div class="plot-controls"><span class="label">${t("selectVariables")}:</span>
      <div style="display:flex;gap:14px;margin:8px 0;flex-wrap:wrap">
        <label class="check">${t("xyX")}:
          <select id="xyX">${arrayBases.map((b) =>
            `<option value="${esc(b)}">${esc(b)}</option>`).join("")}</select>
        </label>
      </div>
      <div class="checkbox-grid">${arrayBases.map((b) =>
        `<label class="check"><input type="checkbox" class="pvy" value="${esc(b)}" ${b === "Tb" || b === "tb" ? "checked" : ""}> ${esc(b)}</label>`).join("")}
      </div>
      <button id="btnPlotXY" class="btn">${t("plot")}</button></div>
      <div id="plotArea" style="height:430px"></div>`;
    el.innerHTML = html;
    $("btnPlotXY").addEventListener("click", () => drawXY());
    loadPlotly().then(() => drawXY()).catch(() => {});
  } else {
    el.innerHTML = `<div class="note">${esc(t("noDynamic"))}</div>`;
  }
}

function drawXY() {
  const el = $("tab-plots");
  if (!el.querySelector("#plotArea") || !window.Plotly || !lastRun) return;
  const groups = baseGroups(lastRun.results ? lastRun.results.values : null);
  const xB = el.querySelector("#xyX").value;
  const yBs = Array.from(el.querySelectorAll(".pvy:checked"))
    .map((c) => c.value)
    .filter((b) => b !== xB && typeof groups[b] === "object");
  const gx = groups[xB];
  const xIdx = Object.keys(gx).map(Number).sort((a, b) => a - b);
  const traces = yBs.map((b) => {
    const g = groups[b];
    const yIdx = Object.keys(g).map(Number).sort((a, b) => a - b);
    const common = xIdx.filter((i) => yIdx.includes(i));
    return {
      x: common.map((i) => gx[i]),
      y: common.map((i) => g[i]),
      mode: "lines+markers",
      name: b,
      line: { width: 2.2 },
    };
  });
  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#cbd5e1", size: 12 },
    margin: { l: 55, r: 20, t: 30, b: 45 },
    xaxis: { title: xB, gridcolor: "#24324d", zerolinecolor: "#24324d" },
    yaxis: { gridcolor: "#24324d", zerolinecolor: "#24324d" },
    legend: { orientation: "h", font: { size: 11 } },
    hovermode: "x unified",
  };
  Plotly.newPlot("plotArea", traces, layout, { responsive: true, displaylogo: false });
}

function drawPlot(ts) {
  const el = $("tab-plots");
  if (!el.querySelector("#plotArea") || !window.Plotly) return;
  const names = Array.from(el.querySelectorAll(".pv:checked")).map((c) => c.value);
  const traces = names.map((n) => ({
    x: ts.t,
    y: ts.variables[n],
    mode: "lines",
    name: n,
    line: { width: 2.2 },
  }));
  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#cbd5e1", size: 12 },
    margin: { l: 55, r: 20, t: 30, b: 45 },
    xaxis: { title: t("time"), gridcolor: "#24324d", zerolinecolor: "#24324d" },
    yaxis: { gridcolor: "#24324d", zerolinecolor: "#24324d" },
    legend: { orientation: "h", font: { size: 11 } },
    hovermode: "x unified",
  };
  Plotly.newPlot("plotArea", traces, layout, { responsive: true, displaylogo: false });
}

function loadPlotly() {
  return new Promise((resolve, reject) => {
    if (window.Plotly) return resolve();
    const s = document.createElement("script");
    s.src = "/vendor/plotly.min.js";
    s.onload = resolve;
    s.onerror = reject;
    document.head.appendChild(s);
  });
}

/* ---------------- messages tab ---------------- */
function renderMessagesTab(d) {
  const el = $("tab-messages");
  const msgs = d.messages || [];
  if (!msgs.length) {
    el.innerHTML = `<div class="note">—</div>`;
    return;
  }
  let html = '<ul class="msglist">';
  for (const m of msgs) {
    const icon = m.level === "error" ? "✖" : m.level === "warn" ? "⚠" : "ℹ";
    html += `<li class="${esc(m.level)}"><span class="icon">${icon}</span><span>${esc(tm(m.code, m.args))}</span></li>`;
  }
  el.innerHTML = html + "</ul>";
}

document.addEventListener("DOMContentLoaded", init);
