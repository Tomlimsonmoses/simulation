/* ================================================================
   side_effects.js — Side Effect Forecasting
   ================================================================ */

let _seDuration     = 365;
let _seData         = null;
let _selectedBM     = null;
let _diseaseMap     = {};   // name → { indicated_drugs, default_severity }
let _drugDefaults   = {};   // name → default_dose
let _drugOrgFx      = {};   // name → { organ: magnitude } (from database)

// Normalise organ names that differ between the drug-effects DB and our chart
const _ORG_ALIAS = {
  kidneys: "kidney", lungs: "lung", joints: "bone",
  immune_system: "immune", adipose: "endocrine",
};
function _normOrg(o) { return _ORG_ALIAS[o] || o; }

// ── Bootstrap ────────────────────────────────────────────────────
async function initSEPage() {
  const diseaseEl = document.getElementById("se-disease");
  const drugEl    = document.getElementById("se-drug");

  try {
    const [diseases, drugs, orgFx] = await Promise.all([
      fetch("/api/diseases").then(r => { if (!r.ok) throw new Error(r.status); return r.json(); }),
      fetch("/api/drugs").then(r => { if (!r.ok) throw new Error(r.status); return r.json(); }),
      fetch("/api/drug-organ-effects").then(r => { if (!r.ok) throw new Error(r.status); return r.json(); }),
    ]);

    // Build lookup maps
    drugs.forEach(d => { _drugDefaults[d.name] = d.default_dose; });
    diseases.forEach(d => {
      _diseaseMap[d.name] = {
        indicated_drugs:  d.indicated_drugs || [],
        default_severity: d.default_severity ?? 0.5,
      };
    });
    // Normalise organ names in the drug-organ effects DB
    for (const [drug, effects] of Object.entries(orgFx)) {
      _drugOrgFx[drug] = {};
      for (const [org, mag] of Object.entries(effects)) {
        _drugOrgFx[drug][_normOrg(org)] = mag;
      }
    }

    // Populate disease dropdown (sorted)
    diseaseEl.innerHTML = "";
    diseases
      .sort((a, b) => a.name.localeCompare(b.name))
      .forEach(d => {
        const opt = document.createElement("option");
        opt.value = d.name;
        opt.textContent = d.name;
        if (d.name === "Heart Failure (Mild)") opt.selected = true;
        diseaseEl.appendChild(opt);
      });

    diseaseEl.addEventListener("change", () => {
      updateDrugsByDisease();
      runSEForecast();
    });
    drugEl.addEventListener("change", runSEForecast);

    updateDrugsByDisease();
    runSEForecast();

  } catch (err) {
    console.error("Side-effects init failed:", err);
    // Fallback: hardcoded minimal list so the page is still usable
    const FALLBACK = {
      "Hypertension":             ["Lisinopril","Metoprolol","Amlodipine","Atorvastatin"],
      "Heart Failure (Mild)":     ["Lisinopril","Metoprolol","Atorvastatin"],
      "Type 2 Diabetes (Moderate)":["Metformin","Glipizide"],
      "COPD (Moderate)":          ["Prednisone"],
      "Arthritis":                ["Ibuprofen","Naproxen","Prednisone"],
      "Depression":               ["Sertraline"],
      "Anxiety Disorder":         ["Sertraline","Lorazepam"],
      "Hypothyroidism":           ["Levothyroxine"],
      "Bacterial Infection":      ["Amoxicillin","Ciprofloxacin"],
    };
    diseaseEl.innerHTML = "";
    Object.keys(FALLBACK).forEach((name, i) => {
      const opt = document.createElement("option");
      opt.value = opt.textContent = name;
      if (i === 0) opt.selected = true;
      diseaseEl.appendChild(opt);
      _diseaseMap[name] = { indicated_drugs: FALLBACK[name], default_severity: 0.5 };
    });
    diseaseEl.addEventListener("change", () => { updateDrugsByDisease(); runSEForecast(); });
    drugEl.addEventListener("change", runSEForecast);
    updateDrugsByDisease();
    runSEForecast();
  }
}

// ── Drug dropdown ─────────────────────────────────────────────────
function updateDrugsByDisease() {
  const disease = document.getElementById("se-disease").value;
  const drugEl  = document.getElementById("se-drug");
  const drugs   = _diseaseMap[disease]?.indicated_drugs || [];

  drugEl.innerHTML = "";
  if (!drugs.length) {
    const opt = document.createElement("option");
    opt.value = ""; opt.textContent = "No drugs mapped";
    drugEl.appendChild(opt);
    return;
  }
  drugs.forEach((name, i) => {
    const opt = document.createElement("option");
    opt.value = opt.textContent = name;
    if (i === 0) opt.selected = true;
    drugEl.appendChild(opt);
  });
}

function setSEDur(btn, days) {
  document.querySelectorAll("[data-d]").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  _seDuration = days;
  const label = days >= 365 ? `${(days / 365).toFixed(0)} yr` : `${days}d`;
  document.getElementById("se-kpi-dur").textContent = label;
  document.getElementById("se-summary-duration").textContent =
    days >= 365 ? `${(days / 365).toFixed(0)} year${days >= 730 ? "s" : ""}` : `${days} days`;
}

// ── Main forecast ────────────────────────────────────────────────
async function runSEForecast() {
  const drug    = document.getElementById("se-drug").value;
  const disease = document.getElementById("se-disease").value;
  if (!drug || !disease) return;

  const severity = _diseaseMap[disease]?.default_severity ?? 0.5;
  const dose     = _drugDefaults[drug] ?? 20.0;

  document.getElementById("se-kpi-drug").textContent = drug;
  document.querySelectorAll(".ch4-spinner").forEach(el => el.style.display = "block");
  d3.selectAll("#se-trajectory-chart svg, #se-toxicity-chart svg").remove();
  document.getElementById("se-events-list").innerHTML = "";
  document.getElementById("se-bm-btns").innerHTML     = "";

  try {
    const resp = await fetch("/api/side-effects-forecast", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        drug, disease, severity, dose,
        age: 65, weight: 70, sex: 1,
        kidney_function: 0.8, liver_function: 0.9,
        duration_days: _seDuration,
      }),
    });
    if (!resp.ok) throw new Error(resp.status);
    _seData = await resp.json();
  } catch (e) {
    document.querySelectorAll(".ch4-spinner").forEach(el => el.style.display = "none");
    document.getElementById("se-events-list").innerHTML =
      `<div style="color:#ef5350;padding:16px;font-size:12px">Simulation failed — server error.</div>`;
    return;
  }

  document.querySelectorAll(".ch4-spinner").forEach(el => el.style.display = "none");

  // API returns key "events", not "clinical_events"
  const simEvents  = _seData.events          || [];
  const biomarkers = _seData.biomarkers      || {};
  const immMarkers = _seData.immune_markers  || {};
  const summary    = _seData.summary         || {};

  // Augment simulator events with summary milestones not already in the list
  const events = _buildEventList(simEvents, summary, drug);

  document.getElementById("se-kpi-events").textContent = events.length;
  document.getElementById("se-kpi-risk").textContent   = countHighRiskOrgans(biomarkers, immMarkers, drug);
  document.getElementById("se-kpi-drug").textContent   = drug;

  updateSummary(biomarkers, immMarkers, events, drug, disease);
  buildBMButtons(biomarkers, immMarkers);
  drawToxicityChart(biomarkers, immMarkers, drug, disease);
  renderEventsList(events);
}

// ── Organ toxicity (biomarkers + immune + known drug effects) ────
const ORGAN_BM_MAP = {
  liver:     ["alt","ast","bilirubin","alp","albumin","inr"],
  muscle:    ["ck","ldh","myoglobin","creatine"],
  kidney:    ["creatinine","gfr","urea","bun","egfr"],
  bone:      ["bmd","calcium","phosphate","alk_phos"],
  immune:    ["wbc","crp","il6","tnf","neutrophil","lymphocyte","inflammation"],
  endocrine: ["glucose","hba1c","cortisol","insulin","tsh"],
  heart:     ["bnp","troponin","ejection","nt_probnp","hr"],
  brain:     ["cortisol","il6","nmda","neuro"],
};

function _allBiomarkers(bm, imm) {
  // Merge simulation biomarkers with immune/inflammation markers
  const merged = { ...bm };
  for (const [k, v] of Object.entries(imm)) {
    if (Array.isArray(v) && v.length > 1) merged[k] = v;
  }
  return merged;
}

function computeOrganTox(bm, imm) {
  const all = _allBiomarkers(bm, imm);
  const tox = {};
  for (const [organ, keys] of Object.entries(ORGAN_BM_MAP)) {
    let score = 0, n = 0;
    for (const [bmKey, vals] of Object.entries(all)) {
      if (!Array.isArray(vals) || vals.length < 2) continue;
      const kl = bmKey.toLowerCase();
      if (!keys.some(k => kl.includes(k))) continue;
      const base = Math.abs(vals[0]) || 1;
      const peak = Math.max(...vals.map(v => Math.abs(v ?? 0)));
      score += Math.abs(peak / base - 1);
      n++;
    }
    tox[organ] = n > 0 ? score / n : 0;
  }
  return tox;
}

function _drugOrgTox(drug) {
  // Convert known drug-organ effect magnitudes to a toxicity-style score
  // Negative magnitude = harmful; scale to [0, 1]
  const effects = _drugOrgFx[drug] || {};
  const out = {};
  for (const [org, mag] of Object.entries(effects)) {
    const normOrg = _normOrg(org);
    // Only count harmful (negative) effects; scale: -0.20 → 0.20
    const harm = Math.max(0, -mag);
    out[normOrg] = (out[normOrg] || 0) + harm;
  }
  return out;
}

function countHighRiskOrgans(bm, imm, drug) {
  const simTox  = computeOrganTox(bm, imm);
  const drugTox = _drugOrgTox(drug);
  const all     = new Set([...Object.keys(simTox), ...Object.keys(drugTox)]);
  let count = 0;
  for (const org of all) {
    const combined = (simTox[org] || 0) * 0.7 + (drugTox[org] || 0) * 0.3;
    if (combined > 0.15) count++;
  }
  return count;
}

// ── Summary panel ────────────────────────────────────────────────
function updateSummary(bm, imm, events, drug, disease) {
  const tox      = computeOrganTox(bm, imm);
  const drugTox  = _drugOrgTox(drug);
  const combined = {};
  const all      = new Set([...Object.keys(tox), ...Object.keys(drugTox)]);
  for (const o of all) combined[o] = (tox[o] || 0) * 0.7 + (drugTox[o] || 0) * 0.3;

  const worst = Object.entries(combined).sort((a,b) => b[1] - a[1])[0];
  document.getElementById("se-summary-peak-stress").textContent =
    worst ? `${worst[0]} (index ${worst[1].toFixed(2)})` : "None";
  document.getElementById("se-summary-events-count").textContent = events.length;
  document.getElementById("se-summary-risk-organ").textContent   = worst ? worst[0] : "—";

  const all_bm = _allBiomarkers(bm, imm);
  const lines  = Object.entries(all_bm).slice(0, 6).map(([k, vals]) => {
    if (!Array.isArray(vals) || vals.length < 2) return "";
    const base = vals[0] || 1;
    const last = vals[vals.length - 1];
    const pct  = ((last / base - 1) * 100).toFixed(1);
    const col  = Math.abs(pct) > 30 ? "#ef5350" : Math.abs(pct) > 10 ? "#ffab40" : "#00e676";
    return `<span style="margin-right:10px"><strong>${k}:</strong> <span style="color:${col}">${pct > 0 ? "+" : ""}${pct}%</span></span>`;
  }).filter(Boolean);
  document.getElementById("se-summary-biomarkers").innerHTML = lines.join("");
}

// ── Biomarker buttons ────────────────────────────────────────────
function buildBMButtons(bm, imm) {
  const container = document.getElementById("se-bm-btns");
  container.innerHTML = "";
  // Show clinical biomarkers first, then immune markers
  const clinKeys = Object.keys(bm).slice(0, 6);
  const immKeys  = Object.keys(imm).filter(k => ["crp","il6","wbc","glucose"].includes(k.toLowerCase())).slice(0, 3);
  const keys     = [...new Set([...clinKeys, ...immKeys])];

  keys.forEach((key, i) => {
    const src  = bm[key] ? bm : imm;
    const btn  = document.createElement("button");
    btn.className = "ch4-btn" + (i === 0 ? " active" : "");
    btn.textContent = key.toUpperCase();
    btn.style.cssText = "font-size:10px;padding:3px 8px";
    btn.onclick = () => {
      document.querySelectorAll("#se-bm-btns .ch4-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      _selectedBM = key;
      drawTrajectoryChart(bm, imm, _seData.population_envelopes, key);
    };
    container.appendChild(btn);
    if (i === 0) _selectedBM = key;
  });
  if (_selectedBM) drawTrajectoryChart(bm, imm, _seData.population_envelopes, _selectedBM);
}

// ── Trajectory chart ─────────────────────────────────────────────
function drawTrajectoryChart(bm, imm, envelopes, key) {
  d3.select("#se-trajectory-chart").selectAll("svg").remove();
  const all  = _allBiomarkers(bm, imm);
  const vals = all[key];
  const env  = envelopes?.[key];
  if (!vals || vals.length < 2) return;

  const container = document.getElementById("se-trajectory-chart");
  const W = container.clientWidth || 520;
  const H = 260;
  const M = { top: 20, right: 20, bottom: 40, left: 58 };

  const svg = d3.select("#se-trajectory-chart").append("svg")
    .attr("width","100%").attr("height",H).attr("viewBox",`0 0 ${W} ${H}`);

  const n  = vals.length;
  const xs = d3.range(n).map(i => (i / (n - 1)) * _seDuration);

  let allV = [...vals];
  if (env) allV = allV.concat(env.p05 || [], env.p95 || []);
  const yMax = (d3.max(allV) || 1) * 1.15;
  const yMin = Math.min(0, d3.min(allV) || 0) * 1.1;

  const x = d3.scaleLinear().domain([0, _seDuration]).range([M.left, W - M.right]);
  const y = d3.scaleLinear().domain([yMin, yMax]).range([H - M.bottom, M.top]);

  svg.append("g").attr("transform",`translate(0,${H - M.bottom})`)
    .call(d3.axisBottom(x).ticks(6).tickFormat(d =>
      _seDuration >= 365 ? `${(d/365).toFixed(1)}yr` : `${d}d`))
    .selectAll("text").style("fill","#6a7f9e").style("font-size","10px");
  svg.append("g").attr("transform",`translate(${M.left},0)`)
    .call(d3.axisLeft(y).ticks(5))
    .selectAll("text").style("fill","#6a7f9e").style("font-size","10px");

  svg.append("text").attr("x",W/2).attr("y",H-4).attr("text-anchor","middle")
    .style("fill","#4a5e7a").style("font-size","10px").text("Time");
  svg.append("text").attr("transform","rotate(-90)").attr("x",-(H/2)).attr("y",14)
    .attr("text-anchor","middle").style("fill","#4a5e7a").style("font-size","10px").text(key);

  const lineF = d3.line().x((_,i) => x(xs[i])).y(d => y(d)).curve(d3.curveCatmullRom);

  if (env?.p05 && env?.p95) {
    const area = d3.area().x((_,i)=>x(xs[i])).y0((_,i)=>y(env.p05[i])).y1((_,i)=>y(env.p95[i])).curve(d3.curveCatmullRom);
    svg.append("path").datum(env.p50 || vals).attr("fill","rgba(0,188,212,0.10)").attr("d",area);
    svg.append("path").datum(env.p05).attr("fill","none").attr("stroke","rgba(0,188,212,0.3)")
      .attr("stroke-width",1).attr("stroke-dasharray","3,2").attr("d",lineF);
    svg.append("path").datum(env.p95).attr("fill","none").attr("stroke","rgba(0,188,212,0.3)")
      .attr("stroke-width",1).attr("stroke-dasharray","3,2").attr("d",lineF);
  }

  svg.append("line").attr("x1",M.left).attr("y1",y(vals[0])).attr("x2",W-M.right).attr("y2",y(vals[0]))
    .attr("stroke","rgba(255,255,255,0.15)").attr("stroke-dasharray","4,3");
  svg.append("path").datum(vals).attr("fill","none")
    .attr("stroke","#00bcd4").attr("stroke-width",2.2).attr("d",lineF);

  svg.append("text").attr("x",M.left+6).attr("y",M.top+12)
    .style("fill","#00bcd4").style("font-size","10px").text("P50 Median");
  if (env?.p05) svg.append("text").attr("x",M.left+90).attr("y",M.top+12)
    .style("fill","rgba(0,188,212,0.6)").style("font-size","10px").text("P5–P95 Band");
  svg.append("text").attr("x",M.left+6).attr("y",H-M.bottom-6)
    .style("fill","rgba(255,255,255,0.25)").style("font-size","9px").text("— Baseline");
}

// ── Organ Toxicity Index chart ───────────────────────────────────
function drawToxicityChart(bm, imm, drug, disease) {
  d3.select("#se-toxicity-chart").selectAll("svg").remove();

  const simTox  = computeOrganTox(bm, imm);
  const drugTox = _drugOrgTox(drug);

  // Combined score: 70% simulation-derived, 30% known drug effects
  // Ensures organs with no matching biomarkers still show expected drug effects
  const allOrgans = new Set([...Object.keys(simTox), ...Object.keys(drugTox)]);
  const combined  = {};
  for (const o of allOrgans) {
    combined[o] = (simTox[o] || 0) * 0.7 + (drugTox[o] || 0) * 0.3;
  }

  const organs = Object.entries(combined).sort((a,b) => b[1] - a[1]);
  const maxTox = organs[0]?.[1] || 0.01;

  const container = document.getElementById("se-toxicity-chart");
  const W  = container.clientWidth || 260;
  const H  = 280;
  const M  = { top: 30, right: 60, bottom: 30, left: 80 };
  const RH = Math.floor((H - M.top - M.bottom) / organs.length);

  const svg = d3.select("#se-toxicity-chart").append("svg")
    .attr("width","100%").attr("height",H).attr("viewBox",`0 0 ${W} ${H}`);

  svg.append("text").attr("x",W/2).attr("y",18)
    .attr("text-anchor","middle").style("fill","#4a5e7a").style("font-size","9px")
    .text(`${drug} · ${disease.length > 24 ? disease.slice(0,22)+"…" : disease}`);

  const xScale = d3.scaleLinear().domain([0, Math.max(maxTox, 0.05)]).range([M.left, W - M.right]);

  // Risk threshold line at 0.15
  const threshX = xScale(0.15);
  if (threshX < W - M.right) {
    svg.append("line").attr("x1",threshX).attr("x2",threshX)
      .attr("y1",M.top).attr("y2",H-M.bottom)
      .attr("stroke","rgba(255,171,64,0.4)").attr("stroke-dasharray","3,2").attr("stroke-width",1);
    svg.append("text").attr("x",threshX+2).attr("y",M.top+8)
      .style("fill","#ffab40").style("font-size","7.5px").text("risk threshold");
  }

  const tooltip = document.getElementById("se-tooltip");

  organs.forEach(([organ, tox], i) => {
    const y0  = M.top + i * RH + 1;
    const bh  = RH - 4;
    const bw  = Math.max(xScale(tox) - M.left, 0);
    const col = tox > 0.4 ? "#ef5350" : tox > 0.15 ? "#ffab40" : "#00e676";
    const hasDrugFx = (drugTox[organ] || 0) > 0.01;

    // Stripe
    svg.append("rect").attr("x",M.left).attr("y",y0)
      .attr("width",W-M.left-M.right).attr("height",bh)
      .attr("fill", i%2===0 ? "rgba(255,255,255,0.015)" : "transparent");

    // Simulation-derived bar
    svg.append("rect").attr("x",M.left).attr("y",y0+1)
      .attr("width",0).attr("height",bh-2)
      .attr("rx",3).attr("fill",col).attr("opacity",0.85)
      .transition().delay(i*40).duration(500).attr("width",bw);

    // Known-drug-effect overlay (dashed border on top of bar)
    if (hasDrugFx) {
      const drugBw = Math.max(xScale(drugTox[organ]*0.3) - M.left, 0);
      svg.append("rect").attr("x",M.left).attr("y",y0+1)
        .attr("width",Math.min(drugBw, W-M.left-M.right))
        .attr("height",bh-2).attr("rx",3)
        .attr("fill","none").attr("stroke",col)
        .attr("stroke-width","1.5").attr("stroke-dasharray","3,2").attr("opacity",0.5);
    }

    // Organ label
    svg.append("text").attr("x",M.left-4).attr("y",y0+bh/2+1)
      .attr("text-anchor","end").attr("dominant-baseline","middle")
      .style("fill","#9aaabf").style("font-size","10px")
      .text(organ);

    // Value
    svg.append("text").attr("x",M.left+bw+4).attr("y",y0+bh/2+1)
      .attr("dominant-baseline","middle")
      .style("fill",col).style("font-size","9px").style("font-weight","700")
      .text(tox.toFixed(2));

    // Hover
    svg.append("rect").attr("x",M.left).attr("y",y0)
      .attr("width",W-M.left-M.right).attr("height",bh)
      .attr("fill","transparent").style("cursor","default")
      .on("mousemove", evt => {
        const risk  = tox > 0.4 ? "High Risk" : tox > 0.15 ? "Moderate Risk" : "Low Risk";
        const parts = [`<strong style="color:${col}">${organ.toUpperCase()}</strong>`, risk, `Combined index: ${tox.toFixed(3)}`];
        if (simTox[organ] > 0.005) parts.push(`Biomarker-derived: ${(simTox[organ]||0).toFixed(3)}`);
        if (hasDrugFx) parts.push(`Known drug effect: ${(drugTox[organ]||0).toFixed(3)}`);
        tooltip.style.display = "block";
        tooltip.style.left  = (evt.clientX+12)+"px";
        tooltip.style.top   = (evt.clientY-28)+"px";
        tooltip.innerHTML   = parts.join("<br>");
      })
      .on("mouseleave", () => { tooltip.style.display = "none"; });
  });

  // X-axis
  svg.append("g").attr("transform",`translate(0,${H-M.bottom})`)
    .call(d3.axisBottom(xScale).ticks(4).tickFormat(d => d.toFixed(2)))
    .selectAll("text").style("fill","#6a7f9e").style("font-size","9px");
}

// ── Clinical events list ─────────────────────────────────────────
// Merge simulator events + summary milestones into one sorted list
function _buildEventList(simEvents, summary, drug) {
  const list = [...simEvents];
  const existingTitles = new Set(list.map(e => e.title));

  // Myopathy onset (if not already captured as CK event)
  if (summary.myopathy_onset_h != null && !existingTitles.has("Myopathy onset")) {
    list.push({
      time_h: summary.myopathy_onset_h,
      severity: "warning",
      organ_system: "muscle",
      title: "Myopathy onset detected",
      description: `Muscle biomarker changes consistent with drug-induced myopathy detected at ${_fmtTime(summary.myopathy_onset_h)}. Monitor CK levels and report any muscle pain or weakness.`,
    });
  }

  // ACE-inhibitor cough
  if (summary.cough_onset_h != null && !existingTitles.has("ACE inhibitor cough onset")) {
    list.push({
      time_h: summary.cough_onset_h,
      severity: "info",
      organ_system: "respiratory",
      title: "Dry cough onset",
      description: `Bradykinin-mediated dry cough likely to develop around ${_fmtTime(summary.cough_onset_h)}. Common ACE-inhibitor side effect (~15–20% of patients).`,
    });
  }

  // Bone density loss
  if (summary.bone_loss_pct_per_year > 2.0) {
    list.push({
      time_h: _seDuration * 24 * 0.5,   // flag at midpoint
      severity: summary.bone_loss_pct_per_year > 5 ? "warning" : "info",
      organ_system: "bone",
      title: `Bone density declining — ${summary.bone_loss_pct_per_year.toFixed(1)}%/yr`,
      description: `Projected bone mineral density loss of ${summary.bone_loss_pct_per_year.toFixed(1)}% per year. ` +
        (drug === "Prednisone" ? "Corticosteroid-induced osteoporosis risk. Consider bisphosphonate prophylaxis." :
         "Monitor with DEXA scan if therapy continues beyond 1 year."),
    });
  }

  // B12 depletion
  if (summary.b12_depletion_pct > 15.0) {
    list.push({
      time_h: _seDuration * 24 * 0.7,
      severity: summary.b12_depletion_pct > 30 ? "warning" : "info",
      organ_system: "haematological",
      title: `B12 depleted ${summary.b12_depletion_pct.toFixed(0)}% from baseline`,
      description: `Vitamin B12 has fallen ${summary.b12_depletion_pct.toFixed(0)}% from baseline. ` +
        "Metformin impairs B12 absorption at the ileum. Annual monitoring and oral B12 supplementation recommended.",
    });
  }

  // LDL reduction milestone (positive)
  if (summary.ldl_change_pct < -20) {
    list.push({
      time_h: _seDuration * 24 * 0.25,
      severity: "info",
      organ_system: "cardiovascular",
      title: `LDL reduced ${Math.abs(summary.ldl_change_pct).toFixed(0)}% — therapeutic target`,
      description: `LDL cholesterol reduced by ${Math.abs(summary.ldl_change_pct).toFixed(0)}% from baseline, consistent with statin therapeutic effect. Cardiovascular risk reduction achieved.`,
    });
  }

  // Sort by time
  return list.sort((a, b) => (a.time_h ?? 0) - (b.time_h ?? 0));
}

function _fmtTime(h) {
  if (h == null) return "—";
  const d = h / 24;
  if (d < 1)  return `${h.toFixed(0)}h`;
  if (d < 7)  return `Day ${d.toFixed(0)}`;
  if (d < 30) return `Week ${(d / 7).toFixed(0)}`;
  return `Month ${(d / 30).toFixed(0)}`;
}

function renderEventsList(events) {
  const container = document.getElementById("se-events-list");
  if (!events.length) {
    container.innerHTML = `
      <div style="display:flex;align-items:center;gap:10px;padding:16px;color:#4a5e7a;font-size:12px;">
        <span style="font-size:20px;opacity:0.5">✓</span>
        No adverse clinical events detected at this dose and duration.
      </div>`;
    return;
  }
  const SEV_COL = { critical:"#f44336", warning:"#ff9800", info:"#2196f3" };
  events.forEach(ev => {
    const col  = SEV_COL[ev.severity] || "#2196f3";
    const days = ev.time_h != null ? `Day ${(ev.time_h/24).toFixed(1)}` : "";
    const div  = document.createElement("div");
    div.style.cssText = `border-left:3px solid ${col};padding:9px 12px;background:${col}11;border-radius:4px;margin-bottom:4px`;
    div.innerHTML = `
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:3px">
        <span style="background:${col};color:#000;border-radius:3px;padding:1px 6px;font-size:9px;font-weight:800;letter-spacing:.04em">${(ev.severity||"info").toUpperCase()}</span>
        <span style="font-size:12px;color:#c5d8f0;font-weight:600">${ev.title || ev.organ_system || "Event"}</span>
        ${days ? `<span style="margin-left:auto;font-size:10px;color:#4a5e7a">${days}</span>` : ""}
      </div>
      <div style="font-size:11px;color:#6a7f9e;line-height:1.4">${ev.description || ""}</div>`;
    container.appendChild(div);
  });
}

// ── Init ─────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  initSEPage();
  document.querySelectorAll(".nav-link").forEach(a => {
    if (a.getAttribute("href") === window.location.pathname) a.classList.add("active");
    else a.classList.remove("active");
  });
});
