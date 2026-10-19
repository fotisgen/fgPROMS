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
}

function bindEvents() {
  document.querySelectorAll(".tab").forEach((b) => {
    b.addEventListener("click", () => switchTab(b.dataset.tab));
  });
  $("btnRun").addEventListener("click", runModel);
  $("btnLang").addEventListener("click", toggleLang);
  $("btnOpen").addEventListener("click", () => $("fileInput").click());
  $("btnSave").addEventListener("click", saveFile);
  $("fileInput").addEventListener("change", openFile);
  $("exampleSel").addEventListener("change", (e) => {
    if (e.target.value) loadExample(e.target.value);
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
  let html = `<h3>${t("modelName")}: ${esc(m.name)}</h3>
  <div class="cards">
    <div class="card"><div class="card-num">${df.unknowns}</div><div class="card-label">${t("unknowns")}</div></div>
    <div class="card"><div class="card-num">${df.equations}</div><div class="card-label">${t("eqCount")}</div></div>
    <div class="card ${df.dof === 0 ? "ok" : "bad"}"><div class="card-num">${df.dof}</div><div class="card-label">${t("dof")}</div></div>
    <div class="card"><div class="card-num" style="font-size:15px">${m.isDynamic ? t("dynamic") : t("steady")}</div><div class="card-label">${t("mode")}</div></div>
  </div>`;

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
  let html = `<div class="summary ${r.success ? "ok" : "bad"}">${
    r.success ? "✓ " + t("success") : "✗ " + t("failure")
  } — ${t("solver")}: ${esc(r.method || "—")}${
    r.maxResidual !== undefined && r.maxResidual !== null
      ? " · " + t("maxResidual") + ": " + fmt(r.maxResidual)
      : ""
  }</div>`;

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

  if (r.residuals && r.residuals.length) {
    html += `<h3>${t("eqResiduals")}</h3>
    <table class="grid"><thead><tr><th>${t("eqId")}</th><th>${t("residual")}</th></tr></thead><tbody>`;
    for (const rr of r.residuals)
      html += `<tr><td>#${rr.id}</td><td>${fmt(rr.value)}</td></tr>`;
    html += "</tbody></table>";
  }
  el.innerHTML = html;
}

/* ---------------- plots tab ---------------- */
function renderPlotsTab(d) {
  const el = $("tab-plots");
  const ts = d.timeSeries;
  if (!ts) {
    el.innerHTML = `<div class="note">${esc(t("noDynamic"))}</div>`;
    return;
  }
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
