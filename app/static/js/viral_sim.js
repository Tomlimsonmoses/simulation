/* ================================================================
   viral_sim.js — Objective 1: Viral Pathogen Simulation charts
   ================================================================ */

const ORGAN_COLOURS = {
  lungs:   "#00bcd4",
  gut:     "#8bc34a",
  heart:   "#f44336",
  kidneys: "#ff9800",
  liver:   "#9c27b0",
  brain:   "#3f51b5",
};

let _vsDays = 30;
let _vsData = null;

function setDays(btn, days) {
  document.querySelectorAll("[data-days]").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  _vsDays = days;
}

document.getElementById("vs-immune").addEventListener("input", function () {
  document.getElementById("vs-immune-lbl").textContent = (this.value / 100).toFixed(1) + "×";
});

async function runViralSim() {
  const archetype     = document.getElementById("vs-archetype").value;
  const immuneStrength = parseFloat(document.getElementById("vs-immune").value) / 100;

  ["vs-spinner1","vs-spinner2","vs-spinner3"].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.style.display = "block"; }
  });
  ["vs-kinetics-chart","vs-heatmap-chart","vs-immune-chart"].forEach(id => {
    d3.select("#" + id).selectAll("svg").remove();
  });

  const resp = await fetch("/api/viral-sim", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ archetype, immune_strength: immuneStrength, duration_days: _vsDays }),
  });
  _vsData = await resp.json();

  ["vs-spinner1","vs-spinner2","vs-spinner3"].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.style.display = "none"; }
  });

  // KPIs
  document.getElementById("vs-kpi-row").style.display = "flex";
  document.getElementById("kpi-peak-day").textContent = "Day " + _vsData.peak_viral_day;
  document.getElementById("kpi-lung-exp").textContent = ((_vsData.organ_peak_exposure.lungs || 0) * 100).toFixed(0) + "%";
  document.getElementById("kpi-max-il6").textContent  = (d3.max(_vsData.il6) * 100).toFixed(0) + "%";
  document.getElementById("kpi-archetype").textContent = archetype === "respiratory" ? "Respiratory" : "Systemic";

  drawKineticsChart(_vsData);
  drawHeatmap(_vsData);
  drawImmuneChart(_vsData);
}

/* ── Kinetics line chart ── */
function drawKineticsChart(data) {
  const container = document.getElementById("vs-kinetics-chart");
  const W = container.clientWidth || 560;
  const H = 280;
  const M = { top: 20, right: 20, bottom: 40, left: 50 };

  const svg = d3.select("#vs-kinetics-chart").append("svg")
    .attr("width", "100%").attr("height", H)
    .attr("viewBox", `0 0 ${W} ${H}`);

  const x = d3.scaleLinear().domain([0, d3.max(data.time_days)]).range([M.left, W - M.right]);
  const y = d3.scaleLinear().domain([0, 1]).range([H - M.bottom, M.top]);

  // Gridlines
  svg.append("g").attr("class", "grid")
    .attr("transform", `translate(0,${H - M.bottom})`)
    .call(d3.axisBottom(x).tickSize(-(H - M.top - M.bottom)).tickFormat(""))
    .selectAll("line").style("stroke", "rgba(255,255,255,0.05)");
  svg.select(".grid .domain").remove();

  // Axes
  svg.append("g").attr("transform", `translate(0,${H - M.bottom})`)
    .call(d3.axisBottom(x).ticks(6).tickFormat(d => d + "d"))
    .selectAll("text").style("fill", "#6a7f9e").style("font-size", "10px");
  svg.append("g").attr("transform", `translate(${M.left},0)`)
    .call(d3.axisLeft(y).ticks(5).tickFormat(d => (d * 100).toFixed(0) + "%"))
    .selectAll("text").style("fill", "#6a7f9e").style("font-size", "10px");

  // Axis labels
  svg.append("text").attr("x", W / 2).attr("y", H - 4)
    .attr("text-anchor", "middle").style("fill", "#4a5e7a").style("font-size", "10px")
    .text("Time (days)");
  svg.append("text").attr("transform", "rotate(-90)")
    .attr("x", -(H / 2)).attr("y", 14)
    .attr("text-anchor", "middle").style("fill", "#4a5e7a").style("font-size", "10px")
    .text("Normalised Viral Load");

  const line = d3.line()
    .x((_, i) => x(data.time_days[i]))
    .y(d => y(d))
    .curve(d3.curveCatmullRom);

  const organs = data.organs || Object.keys(data.organ_viral_load);
  const tooltip = document.getElementById("vs-tooltip");

  organs.forEach(org => {
    const vals = data.organ_viral_load[org];
    if (!vals) return;
    const col = ORGAN_COLOURS[org] || "#888";
    const path = svg.append("path")
      .datum(vals).attr("fill", "none")
      .attr("stroke", col).attr("stroke-width", 1.8)
      .attr("d", line)
      .style("cursor", "pointer");

    path.on("mousemove", (event) => {
      tooltip.style.display = "block";
      tooltip.style.left = (event.clientX + 12) + "px";
      tooltip.style.top  = (event.clientY - 28) + "px";
      // Find closest point
      const [mx] = d3.pointer(event, svg.node());
      const day = x.invert(mx);
      const idx = d3.bisectLeft(data.time_days, day);
      const v = (vals[idx] || 0) * 100;
      tooltip.innerHTML = `<strong>${org}</strong><br>Day ${day.toFixed(1)}: ${v.toFixed(1)}%`;
    }).on("mouseleave", () => { tooltip.style.display = "none"; });
  });

  // Legend
  const legend = d3.select("#vs-kinetics-legend");
  legend.selectAll("*").remove();
  organs.forEach(org => {
    const item = legend.append("div").attr("class", "ch4-legend-item");
    item.append("div").attr("class", "ch4-legend-line")
      .style("background", ORGAN_COLOURS[org] || "#888");
    item.append("span").text(org.charAt(0).toUpperCase() + org.slice(1));
  });
}

/* ── Organ Exposure Heatmap ── */
function drawHeatmap(data) {
  const container = document.getElementById("vs-heatmap-chart");
  const W = container.clientWidth || 280;
  const H = 280;
  const M = { top: 20, right: 60, bottom: 40, left: 70 };

  const organs = data.organs || Object.keys(data.organ_viral_load);
  // Downsample time to 20 columns
  const nCols = 20;
  const totalT = data.time_days.length;
  const step = Math.max(1, Math.floor(totalT / nCols));
  const colIndices = d3.range(0, totalT, step).slice(0, nCols);
  const colDays = colIndices.map(i => data.time_days[i]);

  const svg = d3.select("#vs-heatmap-chart").append("svg")
    .attr("width", "100%").attr("height", H)
    .attr("viewBox", `0 0 ${W} ${H}`);

  const cellW = (W - M.left - M.right) / nCols;
  const cellH = (H - M.top - M.bottom) / organs.length;

  const colour = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, 1]);
  const tooltip = document.getElementById("vs-tooltip");

  organs.forEach((org, ri) => {
    colIndices.forEach((tIdx, ci) => {
      const val = (data.organ_viral_load[org] || [])[tIdx] || 0;
      svg.append("rect")
        .attr("x", M.left + ci * cellW)
        .attr("y", M.top + ri * cellH)
        .attr("width", cellW - 1).attr("height", cellH - 1)
        .attr("fill", colour(val))
        .attr("rx", 1)
        .on("mousemove", (event) => {
          tooltip.style.display = "block";
          tooltip.style.left = (event.clientX + 12) + "px";
          tooltip.style.top  = (event.clientY - 28) + "px";
          tooltip.innerHTML = `<strong>${org}</strong><br>Day ${colDays[ci].toFixed(1)}: ${(val*100).toFixed(1)}%`;
        }).on("mouseleave", () => { tooltip.style.display = "none"; });
    });
    // Row label
    svg.append("text").attr("x", M.left - 6).attr("y", M.top + ri * cellH + cellH / 2 + 4)
      .attr("text-anchor", "end").style("fill", "#7b8fb5").style("font-size", "10px")
      .text(org.charAt(0).toUpperCase() + org.slice(1));
  });

  // X axis labels (every 5 cols)
  colDays.forEach((d, i) => {
    if (i % 5 === 0 || i === colDays.length - 1) {
      svg.append("text")
        .attr("x", M.left + i * cellW + cellW / 2)
        .attr("y", M.top + organs.length * cellH + 14)
        .attr("text-anchor", "middle").style("fill", "#6a7f9e").style("font-size", "9px")
        .text(d.toFixed(0) + "d");
    }
  });

  // Colour scale bar
  const barW = 10, barH = H - M.top - M.bottom;
  const barX = W - M.right + 12;
  const defs = svg.append("defs");
  const grad = defs.append("linearGradient").attr("id","hm-grad")
    .attr("gradientUnits","userSpaceOnUse")
    .attr("x1",0).attr("y1", M.top + barH).attr("x2",0).attr("y2", M.top);
  [0,0.25,0.5,0.75,1].forEach(s => {
    grad.append("stop").attr("offset", s).attr("stop-color", colour(s));
  });
  svg.append("rect").attr("x", barX).attr("y", M.top)
    .attr("width", barW).attr("height", barH)
    .attr("fill","url(#hm-grad)").attr("rx", 2);
  svg.append("text").attr("x", barX + barW + 2).attr("y", M.top + 4)
    .style("fill","#6a7f9e").style("font-size","9px").text("High");
  svg.append("text").attr("x", barX + barW + 2).attr("y", M.top + barH)
    .style("fill","#6a7f9e").style("font-size","9px").text("Low");
}

/* ── Immune response chart ── */
function drawImmuneChart(data) {
  const container = document.getElementById("vs-immune-chart");
  const W = container.clientWidth || 900;
  const H = 240;
  const M = { top: 20, right: 20, bottom: 40, left: 50 };

  const svg = d3.select("#vs-immune-chart").append("svg")
    .attr("width", "100%").attr("height", H)
    .attr("viewBox", `0 0 ${W} ${H}`);

  const x = d3.scaleLinear().domain([0, d3.max(data.time_days)]).range([M.left, W - M.right]);
  const y = d3.scaleLinear().domain([0, 1]).range([H - M.bottom, M.top]);

  svg.append("g").attr("transform", `translate(0,${H - M.bottom})`)
    .call(d3.axisBottom(x).ticks(8).tickFormat(d => d + "d"))
    .selectAll("text").style("fill", "#6a7f9e").style("font-size", "10px");
  svg.append("g").attr("transform", `translate(${M.left},0)`)
    .call(d3.axisLeft(y).ticks(5).tickFormat(d => (d * 100).toFixed(0) + "%"))
    .selectAll("text").style("fill", "#6a7f9e").style("font-size", "10px");

  const series = [
    { key: "viral_load_total", label: "Viral Load",       colour: "#f44336", dash: null },
    { key: "infected_cells",   label: "Infected Cells",   colour: "#ff9800", dash: "4,3" },
    { key: "immune_effectors", label: "Immune Effectors", colour: "#4caf50", dash: null },
    { key: "il6",              label: "IL-6 (proxy)",     colour: "#e91e63", dash: "8,4" },
    { key: "tnf_alpha",        label: "TNF-α (proxy)",    colour: "#9c27b0", dash: "4,4" },
    { key: "neutrophils",      label: "Neutrophils",      colour: "#03a9f4", dash: null },
  ];

  const line = d3.line().x((_, i) => x(data.time_days[i])).y(d => y(d)).curve(d3.curveCatmullRom);
  const tooltip = document.getElementById("vs-tooltip");

  series.forEach(s => {
    const vals = data[s.key];
    if (!vals) return;
    const path = svg.append("path").datum(vals)
      .attr("fill", "none").attr("stroke", s.colour).attr("stroke-width", 1.8)
      .attr("d", line);
    if (s.dash) path.attr("stroke-dasharray", s.dash);
    path.on("mousemove", (event) => {
      tooltip.style.display = "block";
      tooltip.style.left = (event.clientX + 12) + "px";
      tooltip.style.top  = (event.clientY - 28) + "px";
      const [mx] = d3.pointer(event, svg.node());
      const day = x.invert(mx);
      const idx = Math.max(0, Math.min(vals.length - 1, Math.round(day / data.time_days[1])));
      tooltip.innerHTML = `<strong>${s.label}</strong><br>Day ${day.toFixed(1)}: ${(vals[idx] * 100).toFixed(1)}%`;
    }).on("mouseleave", () => { tooltip.style.display = "none"; });
  });

  // Legend
  const legend = d3.select("#vs-immune-legend");
  legend.selectAll("*").remove();
  series.forEach(s => {
    const item = legend.append("div").attr("class", "ch4-legend-item");
    item.append("div").attr("class", "ch4-legend-line")
      .style("background", s.colour)
      .style("border-top", s.dash ? `2px dashed ${s.colour}` : "none")
      .style("height", s.dash ? "0" : "2px");
    item.append("span").text(s.label);
  });
}

// Auto-run on load
window.addEventListener("DOMContentLoaded", () => {
  runViralSim();
  // highlight active nav
  document.querySelectorAll(".nav-link").forEach(a => {
    if (a.getAttribute("href") === window.location.pathname) a.classList.add("active");
    else a.classList.remove("active");
  });
});
