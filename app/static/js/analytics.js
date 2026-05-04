/* analytics.js — PK-PD BioSim Analytics page (D3 v7) */

"use strict";

// ─── Colour palette ───────────────────────────────────────────────────────────
const AN = {
  bg:      "#060c1a",
  surface: "#0d1b2e",
  accent:  "#00bcd4",
  text:    "#c5d8f0",
  muted:   "#4a5e7a",
  muted2:  "#6a7f9e",
  pos:     "#4caf50",
  warn:    "#ff9800",
  danger:  "#f44336",
};

// ─── Tooltip helper ───────────────────────────────────────────────────────────
const tooltip = d3.select("#an-tooltip");
function showTip(event, html) {
  tooltip.style("display", "block").html(html)
    .style("left", (event.clientX + 14) + "px")
    .style("top",  (event.clientY - 10) + "px");
}
function hideTip() { tooltip.style("display", "none"); }

// ─── Spinner helpers ──────────────────────────────────────────────────────────
function hideSpin(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = "none";
}

// ─── 1. Histogram ─────────────────────────────────────────────────────────────
function drawHistogram(containerId, spinnerId, distObj) {
  hideSpin(spinnerId);
  const host = document.getElementById(containerId);
  if (!host) return;

  const W = host.clientWidth  || 260;
  const H = host.clientHeight || 220;
  const m = { top: 16, right: 14, bottom: 36, left: 44 };
  const w = W - m.left - m.right;
  const h = H - m.top  - m.bottom;

  const svg = d3.select(`#${containerId}`).append("svg")
    .attr("width", W).attr("height", H);
  const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);

  const edges  = distObj.hist_edges;
  const counts = distObj.hist_counts;
  const bins   = counts.map((c, i) => ({ x0: edges[i], x1: edges[i + 1], count: c }));

  const xScale = d3.scaleLinear().domain([edges[0], edges[edges.length - 1]]).range([0, w]);
  const yScale = d3.scaleLinear().domain([0, d3.max(counts)]).nice().range([h, 0]);

  // bars
  g.selectAll("rect").data(bins).join("rect")
    .attr("x",      d => xScale(d.x0) + 1)
    .attr("width",  d => Math.max(0, xScale(d.x1) - xScale(d.x0) - 2))
    .attr("y",      d => yScale(d.count))
    .attr("height", d => h - yScale(d.count))
    .attr("fill",   AN.accent)
    .attr("opacity", 0.78)
    .on("mouseover", (ev, d) =>
      showTip(ev, `<b>${d.x0.toFixed(3)}–${d.x1.toFixed(3)}</b><br/>${d.count} drug(s)`))
    .on("mousemove", (ev) =>
      tooltip.style("left", (ev.clientX + 14) + "px").style("top", (ev.clientY - 10) + "px"))
    .on("mouseout", hideTip);

  // mean line
  g.append("line")
    .attr("x1", xScale(distObj.mean)).attr("x2", xScale(distObj.mean))
    .attr("y1", 0).attr("y2", h)
    .attr("stroke", AN.warn).attr("stroke-width", 1.6)
    .attr("stroke-dasharray", "4,3");
  g.append("text")
    .attr("x", xScale(distObj.mean) + 4).attr("y", 12)
    .attr("fill", AN.warn).attr("font-size", "10px").text(`μ=${distObj.mean.toFixed(2)}`);

  // axes
  g.append("g").attr("transform", `translate(0,${h})`)
    .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format(".2g")))
    .call(a => a.selectAll("text").attr("fill", AN.muted2).attr("font-size", "10px"))
    .call(a => a.select(".domain").attr("stroke", AN.muted))
    .call(a => a.selectAll(".tick line").attr("stroke", AN.muted));

  g.append("g")
    .call(d3.axisLeft(yScale).ticks(4))
    .call(a => a.selectAll("text").attr("fill", AN.muted2).attr("font-size", "10px"))
    .call(a => a.select(".domain").attr("stroke", AN.muted))
    .call(a => a.selectAll(".tick line").attr("stroke", AN.muted));
}

// ─── 2. Correlation Heatmap ───────────────────────────────────────────────────
function drawCorrHeatmap(data) {
  hideSpin("an-sp5");
  const host = document.getElementById("an-corr-heatmap");
  if (!host) return;

  const labels = data.labels;
  const matrix = data.matrix;
  const N = labels.length;

  const W = host.clientWidth  || 500;
  const H = host.clientHeight || 340;
  const m = { top: 60, right: 30, bottom: 30, left: 110 };
  const w = W - m.left - m.right;
  const h = H - m.top  - m.bottom;
  const cellW = w / N;
  const cellH = h / N;

  const svg = d3.select("#an-corr-heatmap").append("svg")
    .attr("width", W).attr("height", H);
  const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);

  const colorScale = d3.scaleDiverging(t => d3.interpolateRdBu(1 - t))
    .domain([-1, 0, 1]);

  // flatten cells
  const cells = [];
  matrix.forEach((row, ri) => row.forEach((val, ci) => cells.push({ ri, ci, val })));

  g.selectAll("rect").data(cells).join("rect")
    .attr("x",      d => d.ci * cellW)
    .attr("y",      d => d.ri * cellH)
    .attr("width",  cellW - 1)
    .attr("height", cellH - 1)
    .attr("fill",   d => colorScale(d.val))
    .on("mouseover", (ev, d) =>
      showTip(ev, `<b>${labels[d.ri]}</b> × <b>${labels[d.ci]}</b><br/>r = ${d.val.toFixed(3)}`))
    .on("mousemove", (ev) =>
      tooltip.style("left", (ev.clientX + 14) + "px").style("top", (ev.clientY - 10) + "px"))
    .on("mouseout", hideTip);

  // cell values (only if cell is big enough)
  if (cellW > 28) {
    g.selectAll("text.cell-val").data(cells).join("text")
      .attr("class", "cell-val")
      .attr("x",     d => d.ci * cellW + cellW / 2)
      .attr("y",     d => d.ri * cellH + cellH / 2 + 4)
      .attr("text-anchor", "middle")
      .attr("font-size",   Math.min(9, cellW * 0.32) + "px")
      .attr("fill",        d => Math.abs(d.val) > 0.5 ? "#fff" : "#333")
      .text(d => d.val.toFixed(2));
  }

  // col labels (top, rotated)
  g.selectAll("text.col-lbl").data(labels).join("text")
    .attr("class", "col-lbl")
    .attr("x",     (_, i) => i * cellW + cellW / 2)
    .attr("y",     -6)
    .attr("transform", (_, i) => `rotate(-45,${i * cellW + cellW / 2},-6)`)
    .attr("text-anchor", "start")
    .attr("font-size", "9px")
    .attr("fill", AN.muted2)
    .text(d => d.replace(/_/g, " "));

  // row labels (left)
  g.selectAll("text.row-lbl").data(labels).join("text")
    .attr("class", "row-lbl")
    .attr("x", -6)
    .attr("y", (_, i) => i * cellH + cellH / 2 + 4)
    .attr("text-anchor", "end")
    .attr("font-size", "9px")
    .attr("fill", AN.muted2)
    .text(d => d.replace(/_/g, " "));
}

// ─── 3. Feature Importance (horizontal bars) ──────────────────────────────────
function drawFeatureImportance(data) {
  hideSpin("an-sp6");
  const host = document.getElementById("an-feat-importance");
  if (!host) return;

  const W = host.clientWidth  || 280;
  const H = host.clientHeight || 340;
  const m = { top: 14, right: 30, bottom: 30, left: 130 };
  const w = W - m.left - m.right;
  const h = H - m.top  - m.bottom;

  const svg = d3.select("#an-feat-importance").append("svg")
    .attr("width", W).attr("height", H);
  const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);

  const xScale = d3.scaleLinear().domain([-1, 1]).range([0, w]);
  const yScale = d3.scaleBand()
    .domain(data.map(d => d.feature))
    .range([0, h]).padding(0.25);

  // zero line
  g.append("line")
    .attr("x1", xScale(0)).attr("x2", xScale(0))
    .attr("y1", 0).attr("y2", h)
    .attr("stroke", AN.muted2).attr("stroke-width", 1);

  g.selectAll("rect").data(data).join("rect")
    .attr("x",      d => d.correlation >= 0 ? xScale(0) : xScale(d.correlation))
    .attr("y",      d => yScale(d.feature))
    .attr("width",  d => Math.abs(xScale(d.correlation) - xScale(0)))
    .attr("height", yScale.bandwidth())
    .attr("fill",   d => d.correlation >= 0 ? AN.accent : AN.danger)
    .attr("opacity", 0.85)
    .on("mouseover", (ev, d) =>
      showTip(ev,
        `<b>${d.feature.replace(/_/g, " ")}</b><br/>r = ${d.correlation.toFixed(4)}`
        + `<br/>p = ${d.pvalue < 0.001 ? "&lt;0.001" : d.pvalue.toFixed(4)}`))
    .on("mousemove", (ev) =>
      tooltip.style("left", (ev.clientX + 14) + "px").style("top", (ev.clientY - 10) + "px"))
    .on("mouseout", hideTip);

  // r value labels
  g.selectAll("text.bar-val").data(data).join("text")
    .attr("class", "bar-val")
    .attr("x", d => d.correlation >= 0
      ? xScale(d.correlation) + 3
      : xScale(d.correlation) - 3)
    .attr("y", d => yScale(d.feature) + yScale.bandwidth() / 2 + 4)
    .attr("text-anchor", d => d.correlation >= 0 ? "start" : "end")
    .attr("font-size", "9px")
    .attr("fill", AN.text)
    .text(d => d.correlation.toFixed(3));

  // y axis
  g.append("g")
    .call(d3.axisLeft(yScale).tickFormat(d => d.replace(/_/g, " ")))
    .call(a => a.selectAll("text").attr("fill", AN.muted2).attr("font-size", "9px"))
    .call(a => a.select(".domain").remove())
    .call(a => a.selectAll(".tick line").remove());

  // x axis
  g.append("g").attr("transform", `translate(0,${h})`)
    .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format(".1f")))
    .call(a => a.selectAll("text").attr("fill", AN.muted2).attr("font-size", "10px"))
    .call(a => a.select(".domain").attr("stroke", AN.muted))
    .call(a => a.selectAll(".tick line").attr("stroke", AN.muted));
}

// ─── 4. Model Performance ─────────────────────────────────────────────────────
function drawModelPerformance(models) {
  hideSpin("an-sp7");
  const host = document.getElementById("an-model-perf");
  if (!host) return;

  const W = host.clientWidth  || 900;
  const H = host.clientHeight || 280;
  const m = { top: 20, right: 120, bottom: 50, left: 130 };
  const w = W - m.left - m.right;
  const h = H - m.top  - m.bottom;

  const svg = d3.select("#an-model-perf").append("svg")
    .attr("width", W).attr("height", H);
  const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);

  const names = models.map(m => m.name);
  const yScale = d3.scaleBand().domain(names).range([0, h]).padding(0.28);
  const xScale = d3.scaleLinear().domain([-0.5, 1]).range([0, w]);

  // literature band 0.40–0.80
  g.append("rect")
    .attr("x",      xScale(0.40))
    .attr("width",  xScale(0.80) - xScale(0.40))
    .attr("y",      0).attr("height", h)
    .attr("fill",   "#ffeb3b").attr("opacity", 0.08);
  g.append("text")
    .attr("x", xScale(0.60)).attr("y", -6)
    .attr("text-anchor", "middle").attr("font-size", "9px")
    .attr("fill", "#ffeb3b").attr("opacity", 0.7)
    .text("Literature range 0.40–0.80");

  const BARS = [
    { key: "r2",    label: "R² Overall",  color: AN.accent,  offset: 0 },
    { key: "pk_r2", label: "R² PK",       color: AN.pos,     offset: 1 },
    { key: "pd_r2", label: "R² PD",       color: "#9c27b0",  offset: 2 },
  ];
  const sub = d3.scaleBand().domain(BARS.map(b => b.key))
    .range([0, yScale.bandwidth()]).padding(0.1);

  BARS.forEach(({ key, color, offset }) => {
    g.selectAll(`rect.${key}`).data(models).join("rect")
      .attr("class", key)
      .attr("x",      d => xScale(Math.min(0, d[key])))
      .attr("y",      d => yScale(d.name) + sub(key))
      .attr("width",  d => Math.abs(xScale(d[key]) - xScale(0)))
      .attr("height", sub.bandwidth())
      .attr("fill",   color).attr("opacity", 0.84)
      .on("mouseover", (ev, d) =>
        showTip(ev, `<b>${d.name}</b><br/>${key.toUpperCase()}: ${d[key].toFixed(4)}`))
      .on("mousemove", (ev) =>
        tooltip.style("left", (ev.clientX + 14) + "px").style("top", (ev.clientY - 10) + "px"))
      .on("mouseout", hideTip);
  });

  // RMSE dots
  const rmseScale = d3.scaleLinear().domain([0, 0.16]).range([0, w]);
  g.selectAll("circle.rmse").data(models).join("circle")
    .attr("class", "rmse")
    .attr("cx",  d => rmseScale(d.rmse))
    .attr("cy",  d => yScale(d.name) + yScale.bandwidth() / 2)
    .attr("r",   5)
    .attr("fill", AN.warn).attr("opacity", 0.9)
    .on("mouseover", (ev, d) =>
      showTip(ev, `<b>${d.name}</b><br/>RMSE: ${d.rmse.toFixed(4)}`))
    .on("mousemove", (ev) =>
      tooltip.style("left", (ev.clientX + 14) + "px").style("top", (ev.clientY - 10) + "px"))
    .on("mouseout", hideTip);

  // axes
  g.append("g").attr("transform", `translate(0,${h})`)
    .call(d3.axisBottom(xScale).ticks(7).tickFormat(d3.format(".1f")))
    .call(a => a.selectAll("text").attr("fill", AN.muted2).attr("font-size", "10px"))
    .call(a => a.select(".domain").attr("stroke", AN.muted))
    .call(a => a.selectAll(".tick line").attr("stroke", AN.muted));

  g.append("g")
    .call(d3.axisLeft(yScale))
    .call(a => a.selectAll("text").attr("fill", AN.muted2).attr("font-size", "10px"))
    .call(a => a.select(".domain").remove())
    .call(a => a.selectAll(".tick line").remove());

  // legend
  const lg = svg.append("g").attr("transform", `translate(${W - m.right + 10},${m.top + 10})`);
  const legend = [
    { color: AN.accent, label: "R² Overall" },
    { color: AN.pos,    label: "R² PK" },
    { color: "#9c27b0", label: "R² PD" },
    { color: AN.warn,   label: "RMSE (●)" },
  ];
  legend.forEach(({ color, label }, i) => {
    lg.append("rect").attr("x", 0).attr("y", i * 20).attr("width", 10).attr("height", 10)
      .attr("fill", color).attr("opacity", 0.85);
    lg.append("text").attr("x", 14).attr("y", i * 20 + 9)
      .attr("fill", AN.muted2).attr("font-size", "9px").text(label);
  });
}

// ─── 5. Box Plots ─────────────────────────────────────────────────────────────
function drawBoxPlots(boxData) {
  hideSpin("an-sp8");
  const host = document.getElementById("an-boxplot");
  if (!host) return;

  const W = host.clientWidth  || 500;
  const H = host.clientHeight || 300;
  const m = { top: 20, right: 20, bottom: 60, left: 52 };
  const w = W - m.left - m.right;
  const h = H - m.top  - m.bottom;

  const svg = d3.select("#an-boxplot").append("svg")
    .attr("width", W).attr("height", H);
  const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);

  const allVals = boxData.flatMap(d => [d.min, d.max]);
  const yScale = d3.scaleLinear().domain([0, d3.max(allVals) * 1.08]).nice().range([h, 0]);
  const xScale = d3.scaleBand()
    .domain(boxData.map(d => d.group))
    .range([0, w]).padding(0.3);

  const bw = xScale.bandwidth();

  boxData.forEach(d => {
    const cx = xScale(d.group) + bw / 2;

    // whiskers
    g.append("line")
      .attr("x1", cx).attr("x2", cx)
      .attr("y1", yScale(d.min)).attr("y2", yScale(d.q1))
      .attr("stroke", AN.muted2).attr("stroke-width", 1.4);
    g.append("line")
      .attr("x1", cx).attr("x2", cx)
      .attr("y1", yScale(d.q3)).attr("y2", yScale(d.max))
      .attr("stroke", AN.muted2).attr("stroke-width", 1.4);

    // whisker caps
    [d.min, d.max].forEach(v => {
      g.append("line")
        .attr("x1", cx - bw * 0.18).attr("x2", cx + bw * 0.18)
        .attr("y1", yScale(v)).attr("y2", yScale(v))
        .attr("stroke", AN.muted2).attr("stroke-width", 1.2);
    });

    // IQR box
    g.append("rect")
      .attr("x",      xScale(d.group))
      .attr("y",      yScale(d.q3))
      .attr("width",  bw)
      .attr("height", yScale(d.q1) - yScale(d.q3))
      .attr("fill",   AN.accent).attr("opacity", 0.22)
      .attr("stroke", AN.accent).attr("stroke-width", 1.4)
      .on("mouseover", (ev) =>
        showTip(ev,
          `<b>${d.group.replace(/_/g, " ")}</b>`
          + `<br/>Min ${d.min} · Q1 ${d.q1}`
          + `<br/>Median ${d.median} · Mean ${d.mean}`
          + `<br/>Q3 ${d.q3} · Max ${d.max}`))
      .on("mousemove", (ev) =>
        tooltip.style("left", (ev.clientX + 14) + "px").style("top", (ev.clientY - 10) + "px"))
      .on("mouseout", hideTip);

    // median line (white)
    g.append("line")
      .attr("x1", xScale(d.group)).attr("x2", xScale(d.group) + bw)
      .attr("y1", yScale(d.median)).attr("y2", yScale(d.median))
      .attr("stroke", "#fff").attr("stroke-width", 2);

    // mean diamond (orange)
    const my = yScale(d.mean);
    const ds = bw * 0.16;
    g.append("path")
      .attr("d", `M${cx},${my - ds} L${cx + ds},${my} L${cx},${my + ds} L${cx - ds},${my} Z`)
      .attr("fill", AN.warn);
  });

  // axes
  g.append("g").attr("transform", `translate(0,${h})`)
    .call(d3.axisBottom(xScale).tickFormat(d => d.replace(/_/g, " ")))
    .call(a => a.selectAll("text")
      .attr("fill", AN.muted2).attr("font-size", "9px")
      .attr("transform", "rotate(-30)").attr("text-anchor", "end"))
    .call(a => a.select(".domain").attr("stroke", AN.muted))
    .call(a => a.selectAll(".tick line").attr("stroke", AN.muted));

  g.append("g")
    .call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format(".1f")))
    .call(a => a.selectAll("text").attr("fill", AN.muted2).attr("font-size", "10px"))
    .call(a => a.select(".domain").attr("stroke", AN.muted))
    .call(a => a.selectAll(".tick line").attr("stroke", AN.muted));

  // y label
  svg.append("text")
    .attr("transform", "rotate(-90)")
    .attr("x", -(m.top + h / 2)).attr("y", 12)
    .attr("text-anchor", "middle").attr("font-size", "10px")
    .attr("fill", AN.muted2).text("Half-Life (h)");
}

// ─── 6. Stats Table ───────────────────────────────────────────────────────────
function sigStars(p) {
  if (p < 0.001) return "***";
  if (p < 0.01)  return "**";
  if (p < 0.05)  return "*";
  return "ns";
}

function buildStatsTable(inferential) {
  const host = document.getElementById("an-stats-table");
  if (!host) return;

  const { anova, ttest_ha_vs_elderly, boxplot } = inferential;
  const aSig = sigStars(anova.p_value);
  const tSig = sigStars(ttest_ha_vs_elderly.p_value);

  // group mean bars
  const maxMean = d3.max(boxplot, d => d.mean) || 1;
  const rows = boxplot.map(d => {
    const barW = Math.round((d.mean / maxMean) * 80);
    return `<tr>
      <td style="white-space:nowrap">${d.group.replace(/_/g, " ")}</td>
      <td>${d.mean.toFixed(2)}</td>
      <td>
        <div style="background:${AN.accent};width:${barW}px;height:8px;border-radius:2px;opacity:.7;display:inline-block"></div>
      </td>
    </tr>`;
  }).join("");

  host.innerHTML = `
    <table class="ch4-table" style="font-size:0.8rem">
      <thead><tr><th>Group</th><th>Mean HL (h)</th><th></th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div style="margin-top:12px;color:${AN.muted2};font-size:0.78rem;line-height:1.8">
      <div>
        <span style="color:${AN.text}">One-way ANOVA</span> (6 groups)<br/>
        F&nbsp;=&nbsp;<b style="color:${AN.accent}">${anova.f_stat}</b>&nbsp;&nbsp;
        p&nbsp;=&nbsp;<b style="color:${anova.p_value < 0.05 ? AN.pos : AN.warn}">${anova.p_value.toFixed(5)}</b>
        <b style="color:${anova.p_value < 0.05 ? AN.pos : AN.warn}">&nbsp;${aSig}</b>
      </div>
      <div style="margin-top:8px">
        <span style="color:${AN.text}">Welch t-test</span> (healthy&nbsp;vs&nbsp;elderly)<br/>
        t&nbsp;=&nbsp;<b style="color:${AN.accent}">${ttest_ha_vs_elderly.t_stat}</b>&nbsp;&nbsp;
        p&nbsp;=&nbsp;<b style="color:${ttest_ha_vs_elderly.p_value < 0.05 ? AN.pos : AN.warn}">${ttest_ha_vs_elderly.p_value.toFixed(5)}</b>
        <b style="color:${ttest_ha_vs_elderly.p_value < 0.05 ? AN.pos : AN.warn}">&nbsp;${tSig}</b>
      </div>
    </div>`;
}

// ─── 7. Outlier Scatter ───────────────────────────────────────────────────────
function drawOutlierScatter(outliers) {
  hideSpin("an-sp9");
  const host = document.getElementById("an-outlier-scatter");
  if (!host) return;

  const W = host.clientWidth  || 400;
  const H = host.clientHeight || 300;
  const m = { top: 20, right: 20, bottom: 46, left: 56 };
  const w = W - m.left - m.right;
  const h = H - m.top  - m.bottom;

  const svg = d3.select("#an-outlier-scatter").append("svg")
    .attr("width", W).attr("height", H);
  const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);

  const xScale = d3.scaleLinear()
    .domain(d3.extent(outliers, d => d.half_life)).nice().range([0, w]);
  const yScale = d3.scaleLinear()
    .domain(d3.extent(outliers, d => d.clearance)).nice().range([h, 0]);

  // grid
  g.append("g").attr("class", "grid")
    .call(d3.axisLeft(yScale).ticks(4).tickSize(-w).tickFormat(""))
    .call(a => a.select(".domain").remove())
    .call(a => a.selectAll(".tick line").attr("stroke", AN.muted).attr("opacity", 0.3));

  g.selectAll("circle").data(outliers).join("circle")
    .attr("cx", d => xScale(d.half_life))
    .attr("cy", d => yScale(d.clearance))
    .attr("r",  d => d.is_outlier ? 7 : 4.5)
    .attr("fill",    d => d.is_outlier ? AN.danger : AN.accent)
    .attr("opacity", d => d.is_outlier ? 0.92 : 0.62)
    .attr("stroke",  d => d.is_outlier ? "#fff" : "none")
    .attr("stroke-width", 1)
    .on("mouseover", (ev, d) =>
      showTip(ev,
        `<b>${d.drug}</b><br/>`
        + `Half-Life: ${d.half_life}<br/>`
        + `Clearance: ${d.clearance}<br/>`
        + `z(HL): ${d.z_hl} · z(CL): ${d.z_cl}`
        + (d.is_outlier ? `<br/><span style="color:${AN.danger}">⚠ Outlier</span>` : "")))
    .on("mousemove", (ev) =>
      tooltip.style("left", (ev.clientX + 14) + "px").style("top", (ev.clientY - 10) + "px"))
    .on("mouseout", hideTip);

  // labels for outliers
  outliers.filter(d => d.is_outlier).forEach(d => {
    g.append("text")
      .attr("x", xScale(d.half_life) + 8)
      .attr("y", yScale(d.clearance) + 4)
      .attr("font-size", "8px").attr("fill", AN.danger)
      .text(d.drug.length > 12 ? d.drug.slice(0, 11) + "…" : d.drug);
  });

  // axes
  g.append("g").attr("transform", `translate(0,${h})`)
    .call(d3.axisBottom(xScale).ticks(5))
    .call(a => a.selectAll("text").attr("fill", AN.muted2).attr("font-size", "10px"))
    .call(a => a.select(".domain").attr("stroke", AN.muted))
    .call(a => a.selectAll(".tick line").attr("stroke", AN.muted));
  g.append("g")
    .call(d3.axisLeft(yScale).ticks(5))
    .call(a => a.selectAll("text").attr("fill", AN.muted2).attr("font-size", "10px"))
    .call(a => a.select(".domain").attr("stroke", AN.muted))
    .call(a => a.selectAll(".tick line").attr("stroke", AN.muted));

  // axis labels
  svg.append("text")
    .attr("x", m.left + w / 2).attr("y", H - 4)
    .attr("text-anchor", "middle").attr("font-size", "10px")
    .attr("fill", AN.muted2).text("Half-Life (h)");
  svg.append("text")
    .attr("transform", "rotate(-90)")
    .attr("x", -(m.top + h / 2)).attr("y", 14)
    .attr("text-anchor", "middle").attr("font-size", "10px")
    .attr("fill", AN.muted2).text("Clearance Rate");
}

// ─── 8. Tornado / Sensitivity ─────────────────────────────────────────────────
function drawTornado(sensitivity) {
  hideSpin("an-sp10");
  const host = document.getElementById("an-tornado");
  if (!host) return;

  const W = host.clientWidth  || 400;
  const H = host.clientHeight || 300;
  const m = { top: 16, right: 40, bottom: 36, left: 140 };
  const w = W - m.left - m.right;
  const h = H - m.top  - m.bottom;

  const svg = d3.select("#an-tornado").append("svg")
    .attr("width", W).attr("height", H);
  const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);

  const allVals = sensitivity.flatMap(d => [d.auc_plus10, d.auc_minus10]);
  const absMax  = d3.max(allVals.map(Math.abs));
  const xScale  = d3.scaleLinear().domain([-absMax * 1.15, absMax * 1.15]).range([0, w]);
  const yScale  = d3.scaleBand()
    .domain(sensitivity.map(d => d.param))
    .range([0, h]).padding(0.28);

  // zero line
  g.append("line")
    .attr("x1", xScale(0)).attr("x2", xScale(0))
    .attr("y1", 0).attr("y2", h)
    .attr("stroke", AN.muted2).attr("stroke-width", 1);

  // +10% bars (cyan)
  g.selectAll("rect.plus").data(sensitivity).join("rect")
    .attr("class", "plus")
    .attr("x",      d => d.auc_plus10 >= 0 ? xScale(0) : xScale(d.auc_plus10))
    .attr("y",      d => yScale(d.param))
    .attr("width",  d => Math.abs(xScale(d.auc_plus10) - xScale(0)))
    .attr("height", yScale.bandwidth() / 2 - 1)
    .attr("fill",   AN.accent).attr("opacity", 0.78)
    .on("mouseover", (ev, d) =>
      showTip(ev, `<b>${d.param}</b> +10%<br/>ΔAUC = ${d.auc_plus10.toFixed(2)}%`))
    .on("mousemove", (ev) =>
      tooltip.style("left", (ev.clientX + 14) + "px").style("top", (ev.clientY - 10) + "px"))
    .on("mouseout", hideTip);

  // −10% bars (red)
  g.selectAll("rect.minus").data(sensitivity).join("rect")
    .attr("class", "minus")
    .attr("x",      d => d.auc_minus10 >= 0 ? xScale(0) : xScale(d.auc_minus10))
    .attr("y",      d => yScale(d.param) + yScale.bandwidth() / 2 + 1)
    .attr("width",  d => Math.abs(xScale(d.auc_minus10) - xScale(0)))
    .attr("height", yScale.bandwidth() / 2 - 1)
    .attr("fill",   AN.danger).attr("opacity", 0.72)
    .on("mouseover", (ev, d) =>
      showTip(ev, `<b>${d.param}</b> −10%<br/>ΔAUC = ${d.auc_minus10.toFixed(2)}%`))
    .on("mousemove", (ev) =>
      tooltip.style("left", (ev.clientX + 14) + "px").style("top", (ev.clientY - 10) + "px"))
    .on("mouseout", hideTip);

  // y axis
  g.append("g")
    .call(d3.axisLeft(yScale).tickFormat(d => d))
    .call(a => a.selectAll("text").attr("fill", AN.muted2).attr("font-size", "9px"))
    .call(a => a.select(".domain").remove())
    .call(a => a.selectAll(".tick line").remove());

  // x axis
  g.append("g").attr("transform", `translate(0,${h})`)
    .call(d3.axisBottom(xScale).ticks(5).tickFormat(d => d + "%"))
    .call(a => a.selectAll("text").attr("fill", AN.muted2).attr("font-size", "10px"))
    .call(a => a.select(".domain").attr("stroke", AN.muted))
    .call(a => a.selectAll(".tick line").attr("stroke", AN.muted));

  // legend
  const lg = svg.append("g").attr("transform", `translate(${W - m.right - 80},${m.top})`);
  [{ color: AN.accent, label: "+10%" }, { color: AN.danger, label: "−10%" }].forEach(({ color, label }, i) => {
    lg.append("rect").attr("x", 0).attr("y", i * 16).attr("width", 10).attr("height", 9)
      .attr("fill", color).attr("opacity", 0.8);
    lg.append("text").attr("x", 13).attr("y", i * 16 + 8)
      .attr("fill", AN.muted2).attr("font-size", "9px").text(label);
  });
}

// ─── 9. Key Findings ──────────────────────────────────────────────────────────
function buildFindings(findings) {
  const ul = document.getElementById("an-findings");
  if (!ul) return;
  ul.innerHTML = findings.map(f => `<li>${f}</li>`).join("");
}

// ─── 10. Summary Stats Table (§4.6) ──────────────────────────────────────────
function buildSummaryTable(descriptive) {
  const host = document.getElementById("an-summary-table");
  if (!host) return;

  const propLabels = {
    bioavailability: "Bioavailability (F)",
    half_life:       "Half-Life (h)",
    clearance_rate:  "Clearance Rate",
    logp:            "LogP",
    protein_binding: "Protein Binding (%)",
  };

  const rows = Object.entries(propLabels).map(([key, label]) => {
    const d = descriptive[key];
    if (!d) return "";
    const sig = (v, dp = 3) => (v == null ? "—" : Number(v).toFixed(dp));
    const skewAbs = Math.abs(d.skewness ?? 0);
    const skewColor = skewAbs > 1 ? AN.warn : skewAbs > 0.5 ? "#aaa" : AN.pos;
    const skewCol = d.skewness != null
      ? `<span style="color:${skewColor}">${sig(d.skewness, 2)}</span>`
      : "—";
    return `<tr>
      <td style="white-space:nowrap;font-weight:500;color:#e0eaf8">${label}</td>
      <td>${sig(d.mean)}</td>
      <td>${sig(d.std)}</td>
      <td>${sig(d.median)}</td>
      <td>${sig(d.min)}</td>
      <td>${sig(d.max)}</td>
      <td>${skewCol}</td>
    </tr>`;
  }).join("");

  host.innerHTML = `
    <table class="ch4-table" style="font-size:0.82rem">
      <thead>
        <tr>
          <th>Property</th>
          <th>Mean</th>
          <th>SD</th>
          <th>Median</th>
          <th>Min</th>
          <th>Max</th>
          <th>Skewness</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    <p style="font-size:0.76rem;color:#4a5e7a;margin:8px 0 0 2px">
      Skewness: <span style="color:${AN.pos}">|s|≤0.5 symmetric</span> ·
      <span style="color:#aaa">0.5–1 moderate</span> ·
      <span style="color:${AN.warn}">|s|>1 skewed</span>
    </p>`;
}

// ─── Main loader ──────────────────────────────────────────────────────────────
async function loadAnalytics() {
  let data;
  try {
    const resp = await fetch("/api/analytics-data");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    data = await resp.json();
  } catch (err) {
    console.error("Analytics data load failed:", err);
    // Hide all spinners and show error in first card
    ["an-sp1","an-sp2","an-sp3","an-sp4","an-sp5","an-sp6","an-sp7","an-sp8","an-sp9","an-sp10"]
      .forEach(hideSpin);
    const errMsg = `<div style="color:${AN.danger};padding:12px;font-size:.85rem">
      Failed to load analytics data. Check server logs.
    </div>`;
    document.getElementById("an-corr-heatmap").innerHTML = errMsg;
    return;
  }

  // KPIs
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  set("an-kpi-drugs",    data.n_drugs);
  set("an-kpi-features", data.n_features);
  const best = data.model_performance.reduce((a, b) => a.r2 > b.r2 ? a : b);
  set("an-kpi-best-r2",  best.r2.toFixed(3));
  set("an-kpi-outliers", data.outliers.filter(o => o.is_outlier).length);
  set("an-kpi-anova-p",  data.inferential.anova.p_value < 0.001
    ? "<0.001"
    : data.inferential.anova.p_value.toFixed(4));

  // Row 1 — histograms
  drawHistogram("an-hist-bav",  "an-sp1", data.descriptive.bioavailability);
  drawHistogram("an-hist-hl",   "an-sp2", data.descriptive.half_life);
  drawHistogram("an-hist-cl",   "an-sp3", data.descriptive.clearance_rate);
  drawHistogram("an-hist-logp", "an-sp4", data.descriptive.logp);

  // Row 2 — correlation
  drawCorrHeatmap(data.correlation);

  // Row 3 — model performance
  drawModelPerformance(data.model_performance);

  // Row 4 — box plots & stats table (§4.5)
  drawBoxPlots(data.inferential.boxplot);
  buildStatsTable(data.inferential);

  // §4.6 — summary table
  buildSummaryTable(data.descriptive);

  // §4.7 — feature importance
  drawFeatureImportance(data.importance);

  // §4.8 — outlier scatter
  drawOutlierScatter(data.outliers);

  // §4.9 — tornado
  drawTornado(data.sensitivity);

  // §4.10 — findings
  buildFindings(data.key_findings);
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  // Highlight active nav link
  document.querySelectorAll(".nav-link").forEach(a => {
    if (a.getAttribute("href") === window.location.pathname) {
      a.classList.add("active");
    } else {
      a.classList.remove("active");
    }
  });

  loadAnalytics();
});
