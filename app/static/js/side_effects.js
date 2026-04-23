/* ================================================================
   side_effects.js — Objective 3: Side Effect Forecasting
   ================================================================ */

let _seDuration = 365;
let _seData = null;
let _selectedBM = null;

function setSEDur(btn, days) {
  document.querySelectorAll("[data-d]").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  _seDuration = days;
  document.getElementById("se-kpi-dur").textContent = days + "d";
}

async function runSEForecast() {
  const drug    = document.getElementById("se-drug").value;
  const disease = document.getElementById("se-disease").value;

  document.getElementById("se-kpi-drug").textContent = drug;
  document.querySelectorAll(".ch4-spinner").forEach(el => el.style.display = "block");
  d3.selectAll("#se-trajectory-chart svg, #se-toxicity-chart svg").remove();
  document.getElementById("se-events-list").innerHTML = "";

  const resp = await fetch("/api/side-effects-forecast", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({
      drug, disease, severity: 0.5,
      age: 65, weight: 70, sex: 1,
      kidney_function: 0.8, liver_function: 0.9,
      duration_days: _seDuration,
    }),
  });
  _seData = await resp.json();

  document.querySelectorAll(".ch4-spinner").forEach(el => el.style.display = "none");

  // KPIs
  const events = _seData.clinical_events || [];
  document.getElementById("se-kpi-events").textContent = events.length;

  // Count organs with high toxicity
  const biomarkers = _seData.biomarkers || {};
  const highRisk = countHighRiskOrgans(biomarkers);
  document.getElementById("se-kpi-risk").textContent = highRisk;

  // Build biomarker buttons
  buildBMButtons(biomarkers);

  // Draw charts
  drawToxicityChart(biomarkers);
  renderEventsList(events);
}

function countHighRiskOrgans(bm) {
  const CONCERN_BMS = ["alt", "ast", "ck", "creatinine", "glucose", "bmd", "cortisol"];
  let count = 0;
  for (const [k, vals] of Object.entries(bm)) {
    const kl = k.toLowerCase();
    if (!CONCERN_BMS.some(c => kl.includes(c))) continue;
    const base = vals[0] || 1;
    const peak = Math.max(...vals);
    if (peak / base > 1.5) count++;
  }
  return count;
}

function buildBMButtons(bm) {
  const container = document.getElementById("se-bm-btns");
  container.innerHTML = "";
  const keys = Object.keys(bm).slice(0, 8);
  keys.forEach((key, i) => {
    const btn = document.createElement("button");
    btn.className = "ch4-btn" + (i === 0 ? " active" : "");
    btn.textContent = key;
    btn.style.fontSize = "10px"; btn.style.padding = "3px 8px";
    btn.onclick = () => {
      document.querySelectorAll("#se-bm-btns .ch4-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      _selectedBM = key;
      drawTrajectoryChart(_seData.biomarkers, _seData.population_envelopes, key);
    };
    container.appendChild(btn);
    if (i === 0) { _selectedBM = key; }
  });
  if (_selectedBM) drawTrajectoryChart(bm, _seData.population_envelopes, _selectedBM);
}

/* ── Biomarker Trajectory Chart with P5/P95 bands ── */
function drawTrajectoryChart(bm, envelopes, key) {
  d3.select("#se-trajectory-chart").selectAll("svg").remove();

  const vals = bm[key];
  const env  = envelopes ? envelopes[key] : null;
  if (!vals) return;

  const container = document.getElementById("se-trajectory-chart");
  const W = container.clientWidth || 520;
  const H = 260;
  const M = { top: 20, right: 20, bottom: 40, left: 58 };

  const svg = d3.select("#se-trajectory-chart").append("svg")
    .attr("width","100%").attr("height",H).attr("viewBox",`0 0 ${W} ${H}`);

  const n = vals.length;
  const xs = d3.range(n).map(i => (i / (n-1)) * _seDuration);

  // Compute Y domain
  let allVals = [...vals];
  if (env) { allVals = allVals.concat(env.p05, env.p95); }
  const yMax = d3.max(allVals) * 1.15 || 1;
  const yMin = Math.min(0, d3.min(allVals)) * 1.1;

  const x = d3.scaleLinear().domain([0, _seDuration]).range([M.left, W-M.right]);
  const y = d3.scaleLinear().domain([yMin, yMax]).range([H-M.bottom, M.top]);

  svg.append("g").attr("transform",`translate(0,${H-M.bottom})`)
    .call(d3.axisBottom(x).ticks(6).tickFormat(d=>d+"d"))
    .selectAll("text").style("fill","#6a7f9e").style("font-size","10px");
  svg.append("g").attr("transform",`translate(${M.left},0)`)
    .call(d3.axisLeft(y).ticks(5))
    .selectAll("text").style("fill","#6a7f9e").style("font-size","10px");

  svg.append("text").attr("x",W/2).attr("y",H-4)
    .attr("text-anchor","middle").style("fill","#4a5e7a").style("font-size","10px").text("Time (days)");
  svg.append("text").attr("transform","rotate(-90)").attr("x",-(H/2)).attr("y",14)
    .attr("text-anchor","middle").style("fill","#4a5e7a").style("font-size","10px").text(key);

  const lineF = d3.line().x((_,i)=>x(xs[i])).y(d=>y(d)).curve(d3.curveCatmullRom);

  // P5–P95 band
  if (env) {
    const area = d3.area().x((_,i)=>x(xs[i])).y0((_,i)=>y(env.p05[i])).y1((_,i)=>y(env.p95[i])).curve(d3.curveCatmullRom);
    svg.append("path").datum(env.p50).attr("fill","rgba(0,188,212,0.1)").attr("d", area);
    svg.append("path").datum(env.p05).attr("fill","none").attr("stroke","rgba(0,188,212,0.3)").attr("stroke-width",1).attr("stroke-dasharray","3,2").attr("d",lineF);
    svg.append("path").datum(env.p95).attr("fill","none").attr("stroke","rgba(0,188,212,0.3)").attr("stroke-width",1).attr("stroke-dasharray","3,2").attr("d",lineF);
  }

  // Baseline reference
  const base = vals[0];
  svg.append("line").attr("x1",M.left).attr("y1",y(base)).attr("x2",W-M.right).attr("y2",y(base))
    .attr("stroke","rgba(255,255,255,0.15)").attr("stroke-dasharray","4,3");

  // P50 median line
  svg.append("path").datum(vals).attr("fill","none")
    .attr("stroke","#00bcd4").attr("stroke-width",2.2).attr("d",lineF);

  // Legend
  svg.append("text").attr("x",M.left+6).attr("y",M.top+12)
    .style("fill","#00bcd4").style("font-size","10px").text("P50 Median");
  if (env) {
    svg.append("text").attr("x",M.left+90).attr("y",M.top+12)
      .style("fill","rgba(0,188,212,0.6)").style("font-size","10px").text("P5–P95 Band");
  }
  svg.append("text").attr("x",M.left+6).attr("y",H-M.bottom-6)
    .style("fill","rgba(255,255,255,0.25)").style("font-size","9px").text("— Baseline");
}

/* ── Toxicity Heatmap ── */
function drawToxicityChart(bm) {
  d3.select("#se-toxicity-chart").selectAll("svg").remove();

  const ORGAN_BM_MAP = {
    "liver":    ["alt","ast","bilirubin"],
    "muscle":   ["ck","ldh"],
    "kidney":   ["creatinine","gfr","urea"],
    "bone":     ["bmd","calcium","phosphate"],
    "immune":   ["wbc","crp","il6"],
    "endocrine":["glucose","hba1c","cortisol"],
    "heart":    ["bnp","troponin","hr"],
  };

  // Compute toxicity index per organ
  const organTox = {};
  for (const [organ, bmKeys] of Object.entries(ORGAN_BM_MAP)) {
    let score = 0; let n = 0;
    for (const key of Object.keys(bm)) {
      const kl = key.toLowerCase();
      if (!bmKeys.some(b => kl.includes(b))) continue;
      const vals = bm[key];
      const base = vals[0] || 1;
      const peak = Math.max(...vals);
      score += Math.abs(peak / base - 1);
      n++;
    }
    organTox[organ] = n > 0 ? score / n : 0;
  }

  const container = document.getElementById("se-toxicity-chart");
  const W = container.clientWidth || 260;
  const H = 260;
  const M = { top: 20, right: 70, bottom: 30, left: 70 };

  const svg = d3.select("#se-toxicity-chart").append("svg")
    .attr("width","100%").attr("height",H).attr("viewBox",`0 0 ${W} ${H}`);

  const organs = Object.keys(organTox);
  const maxTox = d3.max(Object.values(organTox)) || 1;
  const colour = d3.scaleSequential(d3.interpolateRdYlGn).domain([maxTox, 0]);
  const y = d3.scaleBand().domain(organs).range([M.top, H-M.bottom]).padding(0.2);
  const x = d3.scaleLinear().domain([0, maxTox]).range([M.left, W-M.right]);

  svg.append("g").attr("transform",`translate(${M.left},0)`)
    .call(d3.axisLeft(y).tickSize(0))
    .selectAll("text").style("fill","#c5d8f0").style("font-size","10px");

  const tooltip = document.getElementById("se-tooltip");

  organs.forEach(org => {
    const tox = organTox[org];
    svg.append("rect")
      .attr("x",M.left).attr("y",y(org))
      .attr("width", x(tox) - M.left).attr("height", y.bandwidth())
      .attr("rx",4).attr("fill",colour(tox)).attr("opacity",0.85)
      .on("mousemove",(event)=>{
        tooltip.style.display="block";
        tooltip.style.left=(event.clientX+12)+"px";
        tooltip.style.top=(event.clientY-28)+"px";
        tooltip.innerHTML=`<strong>${org}</strong><br>Toxicity index: ${tox.toFixed(3)}`;
      }).on("mouseleave",()=>{ tooltip.style.display="none"; });

    svg.append("text").attr("x", x(tox)+4).attr("y",y(org)+y.bandwidth()/2+4)
      .style("fill","#c5d8f0").style("font-size","9px").text(tox.toFixed(2));
  });
}

/* ── Events List ── */
function renderEventsList(events) {
  const container = document.getElementById("se-events-list");
  if (!events.length) {
    container.innerHTML = "<div style='color:#4a5e7a;font-size:12px;padding:16px'>No clinical events detected at this dose/duration.</div>";
    return;
  }
  events.forEach(ev => {
    const col = ev.severity === "critical" ? "#f44336" : ev.severity === "warning" ? "#ff9800" : "#2196f3";
    const div = document.createElement("div");
    div.style.cssText = `border-left:3px solid ${col};padding:8px 12px;background:rgba(255,255,255,0.02);border-radius:4px`;
    div.innerHTML = `
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:3px">
        <span style="font-size:11px;font-weight:600;color:${col}">${ev.severity?.toUpperCase()}</span>
        <span style="font-size:12px;color:#c5d8f0">${ev.title || ev.organ_system}</span>
        <span style="margin-left:auto;font-size:10px;color:#4a5e7a">t = ${(ev.time_h/24).toFixed(1)} days</span>
      </div>
      <div style="font-size:11px;color:#6a7f9e">${ev.description || ""}</div>
    `;
    container.appendChild(div);
  });
}

window.addEventListener("DOMContentLoaded", () => {
  runSEForecast();
  document.querySelectorAll(".nav-link").forEach(a => {
    if (a.getAttribute("href") === window.location.pathname) a.classList.add("active");
    else a.classList.remove("active");
  });
});
