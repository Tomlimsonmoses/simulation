/* ================================================================
   effectiveness.js — Objective 2: Pharmaceutical Effectiveness
   ================================================================ */

const DRUG_COLOURS = {
  Acetaminophen: "#00bcd4", Ibuprofen: "#ff9800", Metoprolol: "#4caf50",
  Atorvastatin:  "#e91e63", Lisinopril: "#9c27b0", Prednisone: "#f44336",
  Amoxicillin:   "#8bc34a", Ciprofloxacin: "#ffeb3b", Sertraline: "#3f51b5",
  Metformin:     "#03a9f4",
};

let _effData = null;
let _activePKDrugs = new Set(["Acetaminophen","Ibuprofen","Metoprolol","Atorvastatin","Lisinopril"]);
let _pkSeriesData = {};

function togglePKDrug(btn) {
  const drug = btn.dataset.drug;
  if (_activePKDrugs.has(drug)) {
    _activePKDrugs.delete(drug);
    btn.classList.remove("active");
  } else {
    _activePKDrugs.add(drug);
    btn.classList.add("active");
  }
  if (_pkSeriesData && Object.keys(_pkSeriesData).length) drawPKChart(_pkSeriesData);
}

async function loadEffectivenessData() {
  // Load model + effectiveness meta-data
  const resp = await fetch("/api/effectiveness-data");
  _effData = await resp.json();

  // Generate PK time-series for selected drugs by calling /api/simulate for each
  const DRUGS = ["Acetaminophen","Ibuprofen","Metoprolol","Atorvastatin","Lisinopril"];
  const pkPromises = DRUGS.map(drug =>
    fetch("/api/simulate", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        drug, dose: 0, // server will use default dose
        disease: "Hypertension", severity: 0.4,
        age: 65, weight: 70, sex: 1,
        kidney_function: 0.8, liver_function: 0.9,
        heart_rate: 72, metabolic_rate: 1.0,
        duration_hours: 72,
      }),
    }).then(r => r.json()).then(d => ({ drug, data: d }))
  );
  const pkResults = await Promise.all(pkPromises);
  pkResults.forEach(({ drug, data }) => {
    _pkSeriesData[drug] = {
      time: data.time_points,
      conc: data.concentrations[drug],
      tw:   data.therapeutic_windows[drug],
    };
  });

  document.querySelectorAll(".ch4-spinner").forEach(el => el.style.display = "none");

  drawPKChart(_pkSeriesData);
  drawScatterChart(_effData.scatter);
  drawEffBar(_effData.drug_effectiveness);
  drawModelChart(_effData.models);
  buildDetailTable(_effData.drug_effectiveness);
}

/* ── PK Concentration-Time Chart ── */
function drawPKChart(series) {
  d3.select("#eff-pk-chart").selectAll("svg").remove();
  const container = document.getElementById("eff-pk-chart");
  const W = container.clientWidth || 520;
  const H = 260;
  const M = { top: 20, right: 20, bottom: 40, left: 52 };
  const svg = d3.select("#eff-pk-chart").append("svg")
    .attr("width","100%").attr("height",H).attr("viewBox",`0 0 ${W} ${H}`);

  const allTimes = [].concat(...Object.values(series).map(s => s.time));
  const allConcs = [].concat(...Object.entries(series)
    .filter(([k]) => _activePKDrugs.has(k)).map(([,s]) => s.conc));

  const x = d3.scaleLinear().domain([0, d3.max(allTimes) || 72]).range([M.left, W - M.right]);
  const y = d3.scaleLinear().domain([0, d3.max(allConcs) * 1.1 || 10]).range([H - M.bottom, M.top]);

  svg.append("g").attr("transform",`translate(0,${H-M.bottom})`)
    .call(d3.axisBottom(x).ticks(6).tickFormat(d=>d+"h"))
    .selectAll("text").style("fill","#6a7f9e").style("font-size","10px");
  svg.append("g").attr("transform",`translate(${M.left},0)`)
    .call(d3.axisLeft(y).ticks(5))
    .selectAll("text").style("fill","#6a7f9e").style("font-size","10px");

  svg.append("text").attr("x",W/2).attr("y",H-4)
    .attr("text-anchor","middle").style("fill","#4a5e7a").style("font-size","10px").text("Time (h)");
  svg.append("text").attr("transform","rotate(-90)").attr("x",-(H/2)).attr("y",14)
    .attr("text-anchor","middle").style("fill","#4a5e7a").style("font-size","10px").text("Concentration (mg/L)");

  const line = d3.line().x((_,i)=>x(series[Object.keys(series)[0]].time[i]||i)).y(d=>y(d)).curve(d3.curveCatmullRom);
  const tooltip = document.getElementById("eff-tooltip");

  const legend = d3.select("#eff-pk-legend"); legend.selectAll("*").remove();

  Object.entries(series).forEach(([drug, s]) => {
    if (!_activePKDrugs.has(drug)) return;
    const col = DRUG_COLOURS[drug] || "#888";

    // Therapeutic window band
    if (s.tw) {
      svg.append("rect")
        .attr("x", M.left)
        .attr("y", y(s.tw[1]))
        .attr("width", W - M.left - M.right)
        .attr("height", Math.max(0, y(s.tw[0]) - y(s.tw[1])))
        .attr("fill", col).attr("opacity", 0.07);
    }

    const myLine = d3.line().x((_,i) => x(s.time[i])).y(d => y(d)).curve(d3.curveCatmullRom);
    svg.append("path").datum(s.conc)
      .attr("fill","none").attr("stroke",col).attr("stroke-width",1.8).attr("d",myLine)
      .on("mousemove", (event) => {
        tooltip.style.display = "block";
        tooltip.style.left = (event.clientX+12)+"px";
        tooltip.style.top  = (event.clientY-28)+"px";
        const [mx] = d3.pointer(event, svg.node());
        const t = x.invert(mx);
        const idx = d3.bisectLeft(s.time, t);
        const c = s.conc[idx] || 0;
        tooltip.innerHTML = `<strong>${drug}</strong><br>t=${t.toFixed(0)}h: ${c.toFixed(3)} mg/L`;
      }).on("mouseleave", () => { tooltip.style.display = "none"; });

    const item = legend.append("div").attr("class","ch4-legend-item");
    item.append("div").attr("class","ch4-legend-line").style("background",col);
    item.append("span").text(drug);
  });
}

/* ── Predicted vs Actual Scatter ── */
function drawScatterChart(scatter) {
  const container = document.getElementById("eff-scatter-chart");
  const W = container.clientWidth || 280;
  const H = 280;
  const M = { top: 20, right: 20, bottom: 44, left: 48 };
  const svg = d3.select("#eff-scatter-chart").append("svg")
    .attr("width","100%").attr("height",H).attr("viewBox",`0 0 ${W} ${H}`);

  const x = d3.scaleLinear().domain([0,1]).range([M.left, W-M.right]);
  const y = d3.scaleLinear().domain([0,1]).range([H-M.bottom, M.top]);

  svg.append("g").attr("transform",`translate(0,${H-M.bottom})`)
    .call(d3.axisBottom(x).ticks(5).tickFormat(d=>(d*100).toFixed(0)+"%"))
    .selectAll("text").style("fill","#6a7f9e").style("font-size","9px");
  svg.append("g").attr("transform",`translate(${M.left},0)`)
    .call(d3.axisLeft(y).ticks(5).tickFormat(d=>(d*100).toFixed(0)+"%"))
    .selectAll("text").style("fill","#6a7f9e").style("font-size","9px");

  // Diagonal ideal
  svg.append("line").attr("x1",x(0)).attr("y1",y(0)).attr("x2",x(1)).attr("y2",y(1))
    .attr("stroke","rgba(255,255,255,0.2)").attr("stroke-dasharray","4,3");

  // Points
  svg.selectAll(".scatter-dot").data(scatter).join("circle")
    .attr("class","scatter-dot")
    .attr("cx", d => x(d.actual)).attr("cy", d => y(d.predicted))
    .attr("r", 3).attr("fill","#00bcd4").attr("opacity",0.55);

  // R² annotation
  svg.append("text").attr("x", M.left + 8).attr("y", M.top + 14)
    .style("fill","#00bcd4").style("font-size","11px").style("font-weight","600")
    .text("R² = 0.56");

  svg.append("text").attr("x",W/2).attr("y",H-4)
    .attr("text-anchor","middle").style("fill","#4a5e7a").style("font-size","10px")
    .text("Actual Effectiveness");
  svg.append("text").attr("transform","rotate(-90)").attr("x",-(H/2)).attr("y",14)
    .attr("text-anchor","middle").style("fill","#4a5e7a").style("font-size","10px")
    .text("Predicted Effectiveness");
}

/* ── Drug Effectiveness Bar ── */
function drawEffBar(drugEff) {
  const container = document.getElementById("eff-bar-chart");
  const W = container.clientWidth || 520;
  const H = 300;
  const M = { top: 20, right: 20, bottom: 40, left: 130 };

  const svg = d3.select("#eff-bar-chart").append("svg")
    .attr("width","100%").attr("height",H).attr("viewBox",`0 0 ${W} ${H}`);

  const sorted = [...drugEff].sort((a,b) => b.effectiveness_pct - a.effectiveness_pct);
  const y = d3.scaleBand().domain(sorted.map(d=>d.drug)).range([M.top, H-M.bottom]).padding(0.25);
  const x = d3.scaleLinear().domain([0,80]).range([M.left, W-M.right]);

  svg.append("g").attr("transform",`translate(0,${H-M.bottom})`)
    .call(d3.axisBottom(x).ticks(5).tickFormat(d=>d+"%"))
    .selectAll("text").style("fill","#6a7f9e").style("font-size","10px");
  svg.append("g").attr("transform",`translate(${M.left},0)`)
    .call(d3.axisLeft(y))
    .selectAll("text").style("fill","#c5d8f0").style("font-size","10px");

  const tooltip = document.getElementById("eff-tooltip");

  svg.selectAll(".eff-bar").data(sorted).join("rect")
    .attr("class","eff-bar")
    .attr("x", M.left).attr("y", d => y(d.drug))
    .attr("width", d => x(d.effectiveness_pct) - M.left)
    .attr("height", y.bandwidth())
    .attr("rx", 4)
    .attr("fill", d => DRUG_COLOURS[d.drug] || "#00bcd4")
    .attr("opacity", 0.85)
    .on("mousemove", (event, d) => {
      tooltip.style.display = "block";
      tooltip.style.left = (event.clientX+12)+"px";
      tooltip.style.top  = (event.clientY-28)+"px";
      tooltip.innerHTML = `<strong>${d.drug}</strong><br>${d.indication||d.disease}<br>${d.mechanism}`;
    }).on("mouseleave", () => { tooltip.style.display = "none"; });

  // Value labels
  svg.selectAll(".eff-val").data(sorted).join("text")
    .attr("class","eff-val")
    .attr("x", d => x(d.effectiveness_pct) + 4)
    .attr("y", d => y(d.drug) + y.bandwidth() / 2 + 4)
    .style("fill","#c5d8f0").style("font-size","10px")
    .text(d => d.effectiveness_pct + "%");
}

/* ── Model R² Bar Chart ── */
function drawModelChart(models) {
  const container = document.getElementById("eff-model-chart");
  const W = container.clientWidth || 260;
  const H = 260;
  const M = { top: 30, right: 20, bottom: 40, left: 100 };

  const svg = d3.select("#eff-model-chart").append("svg")
    .attr("width","100%").attr("height",H).attr("viewBox",`0 0 ${W} ${H}`);

  const filteredModels = models.filter(m => m.r2 > -1);
  const y = d3.scaleBand().domain(filteredModels.map(m=>m.name)).range([M.top,H-M.bottom]).padding(0.3);
  const x = d3.scaleLinear().domain([-0.2, 1.0]).range([M.left, W-M.right]);

  // Literature benchmark band
  svg.append("rect").attr("x",x(0.40)).attr("y",M.top)
    .attr("width",x(0.80)-x(0.40)).attr("height",H-M.top-M.bottom)
    .attr("fill","rgba(255,193,7,0.07)").attr("stroke","rgba(255,193,7,0.25)").attr("stroke-dasharray","3,2");
  svg.append("text").attr("x",(x(0.40)+x(0.80))/2).attr("y",M.top-6)
    .attr("text-anchor","middle").style("fill","rgba(255,193,7,0.6)").style("font-size","9px")
    .text("Literature range (0.40–0.80)");

  // Zero line
  svg.append("line").attr("x1",x(0)).attr("y1",M.top).attr("x2",x(0)).attr("y2",H-M.bottom)
    .attr("stroke","rgba(255,255,255,0.15)").attr("stroke-dasharray","2,2");

  svg.append("g").attr("transform",`translate(0,${H-M.bottom})`)
    .call(d3.axisBottom(x).ticks(5).tickFormat(d=>d.toFixed(1)))
    .selectAll("text").style("fill","#6a7f9e").style("font-size","9px");
  svg.append("g").attr("transform",`translate(${M.left},0)`)
    .call(d3.axisLeft(y))
    .selectAll("text").style("fill","#c5d8f0").style("font-size","10px");

  const colScale = d3.scaleLinear().domain([-0.2, 0, 0.5, 1]).range(["#f44336","#ff9800","#00bcd4","#4caf50"]);

  svg.selectAll(".model-bar").data(filteredModels).join("rect")
    .attr("class","model-bar")
    .attr("x", d => d.r2 >= 0 ? x(0) : x(d.r2))
    .attr("y", d => y(d.name))
    .attr("width", d => Math.abs(x(d.r2) - x(0)))
    .attr("height", y.bandwidth()).attr("rx",3)
    .attr("fill", d => colScale(d.r2)).attr("opacity",0.85);

  svg.selectAll(".model-val").data(filteredModels).join("text")
    .attr("class","model-val")
    .attr("x", d => x(Math.max(d.r2, 0)) + 4)
    .attr("y", d => y(d.name) + y.bandwidth()/2 + 4)
    .style("fill","#c5d8f0").style("font-size","10px")
    .text(d => d.r2.toFixed(2));

  svg.append("text").attr("x",W/2).attr("y",H-4)
    .attr("text-anchor","middle").style("fill","#4a5e7a").style("font-size","10px").text("R² Score");
}

/* ── Detail Table ── */
function buildDetailTable(drugEff) {
  const tbody = document.querySelector("#eff-detail-table tbody");
  tbody.innerHTML = "";
  drugEff.forEach(d => {
    const tr = document.createElement("tr");
    const pct = d.effectiveness_pct;
    const cls = pct >= 50 ? "sev-low" : pct >= 30 ? "sev-medium" : "sev-high";
    tr.innerHTML = `<td>${d.drug}</td><td>${d.disease || d.indication}</td>
      <td class="${cls}" style="font-weight:600">${pct}%</td>
      <td style="color:#7b8fb5;font-size:11px">${d.mechanism}</td>`;
    tbody.appendChild(tr);
  });
}

window.addEventListener("DOMContentLoaded", () => {
  loadEffectivenessData();
  document.querySelectorAll(".nav-link").forEach(a => {
    if (a.getAttribute("href") === window.location.pathname) a.classList.add("active");
    else a.classList.remove("active");
  });
});
