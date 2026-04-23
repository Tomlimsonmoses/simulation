/* ================================================================
   benchmarks.js — Objective 5: Accuracy vs Established Benchmarks
   ================================================================ */

let _bmData = null;
let _selectedParam = null;

async function loadBenchmarkData() {
  const resp = await fetch("/api/benchmarks-data");
  _bmData = await resp.json();

  document.querySelectorAll(".ch4-spinner").forEach(el => el.style.display = "none");

  // KPIs
  document.getElementById("bm-kpi-drugs").textContent = _bmData.drug_count;
  document.getElementById("bm-kpi-params").textContent = _bmData.param_names.length;
  document.getElementById("bm-kpi-r2").textContent = _bmData.best_r2.toFixed(2);
  const mape = calcMAPE(_bmData.records);
  document.getElementById("bm-kpi-mape").textContent = mape.toFixed(1) + "%";

  // Build param buttons
  buildParamButtons(_bmData.param_names);
  drawErrorHeatmap(_bmData.records, _bmData.param_names, _bmData.drugs);
  drawModelChart(_bmData.model_performance, _bmData.lit_benchmark);
  buildDetailTable(_bmData.records);
}

function calcMAPE(records) {
  const vals = records.map(r => Math.abs(r.pct_error));
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
}

function buildParamButtons(params) {
  const container = document.getElementById("bm-param-btns");
  container.innerHTML = "";
  params.forEach((p, i) => {
    const btn = document.createElement("button");
    btn.className = "ch4-btn" + (i === 0 ? " active" : "");
    btn.textContent = p;
    btn.style.fontSize = "10px"; btn.style.padding = "3px 8px";
    btn.onclick = () => {
      document.querySelectorAll("#bm-param-btns .ch4-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      _selectedParam = p;
      drawPKGroupedBar(_bmData.records.filter(r => r.param === p), p);
    };
    container.appendChild(btn);
  });
  _selectedParam = params[0];
  drawPKGroupedBar(_bmData.records.filter(r => r.param === params[0]), params[0]);
}

/* ── Grouped Bar: Simulated vs Published ── */
function drawPKGroupedBar(records, param) {
  d3.select("#bm-pk-chart").selectAll("svg").remove();

  const container = document.getElementById("bm-pk-chart");
  const W = container.clientWidth || 500;
  const H = 280;
  const M = { top: 20, right: 20, bottom: 60, left: 64 };

  const svg = d3.select("#bm-pk-chart").append("svg")
    .attr("width","100%").attr("height",H).attr("viewBox",`0 0 ${W} ${H}`);

  const drugs = records.map(r => r.drug);
  const maxVal = d3.max(records, r => Math.max(r.published, r.simulated)) * 1.15 || 1;

  const x0 = d3.scaleBand().domain(drugs).range([M.left, W-M.right]).padding(0.25);
  const x1 = d3.scaleBand().domain(["Published","Simulated"]).range([0, x0.bandwidth()]).padding(0.1);
  const y = d3.scaleLinear().domain([0, maxVal]).range([H-M.bottom, M.top]);

  svg.append("g").attr("transform",`translate(0,${H-M.bottom})`)
    .call(d3.axisBottom(x0))
    .selectAll("text").attr("transform","rotate(-30)").style("text-anchor","end").style("fill","#7b8fb5").style("font-size","10px");
  svg.append("g").attr("transform",`translate(${M.left},0)`)
    .call(d3.axisLeft(y).ticks(5))
    .selectAll("text").style("fill","#6a7f9e").style("font-size","10px");

  svg.append("text").attr("transform","rotate(-90)").attr("x",-(H/2)).attr("y",16)
    .attr("text-anchor","middle").style("fill","#4a5e7a").style("font-size","10px").text(param);

  const COL = { Published: "#4a90e2", Simulated: "#00bcd4" };
  const tooltip = document.getElementById("bm-tooltip");

  ["Published","Simulated"].forEach(type => {
    svg.selectAll(".bar-"+type).data(records).join("rect")
      .attr("class","bar-"+type)
      .attr("x", d => x0(d.drug) + x1(type))
      .attr("y", d => y(d[type.toLowerCase()]))
      .attr("width", x1.bandwidth())
      .attr("height", d => H - M.bottom - y(d[type.toLowerCase()]))
      .attr("rx",3).attr("fill",COL[type]).attr("opacity",0.8)
      .on("mousemove",(event,d)=>{
        tooltip.style.display="block";
        tooltip.style.left=(event.clientX+12)+"px";
        tooltip.style.top=(event.clientY-28)+"px";
        tooltip.innerHTML=`<strong>${d.drug}</strong><br>${param} (${type}): ${d[type.toLowerCase()].toFixed(2)}<br>Error: ${d.pct_error.toFixed(1)}%`;
      }).on("mouseleave",()=>{ tooltip.style.display="none"; });
  });

  // Legend
  ["Published","Simulated"].forEach((t, i) => {
    svg.append("rect").attr("x",M.left + i*80).attr("y",8).attr("width",12).attr("height",8).attr("rx",2).attr("fill",COL[t]);
    svg.append("text").attr("x",M.left+14+i*80).attr("y",16).style("fill","#c5d8f0").style("font-size","10px").text(t);
  });
}

/* ── % Error Heatmap ── */
function drawErrorHeatmap(records, params, drugs) {
  const container = document.getElementById("bm-error-heatmap");
  const W = container.clientWidth || 280;
  const H = 280;
  const M = { top: 30, right: 20, bottom: 20, left: 100 };

  const svg = d3.select("#bm-error-heatmap").append("svg")
    .attr("width","100%").attr("height",H).attr("viewBox",`0 0 ${W} ${H}`);

  const cellW = (W - M.left - M.right) / params.length;
  const cellH = (H - M.top - M.bottom) / drugs.length;
  const colour = d3.scaleSequential(d3.interpolateRdYlGn).domain([30, 0]);
  const tooltip = document.getElementById("bm-tooltip");

  drugs.forEach((drug, ri) => {
    params.forEach((param, ci) => {
      const rec = records.find(r => r.drug === drug && r.param === param);
      const err = rec ? Math.abs(rec.pct_error) : 0;
      svg.append("rect")
        .attr("x", M.left + ci * cellW)
        .attr("y", M.top + ri * cellH)
        .attr("width", cellW - 1).attr("height", cellH - 1)
        .attr("fill", colour(err)).attr("rx",2)
        .on("mousemove",(event)=>{
          tooltip.style.display="block";
          tooltip.style.left=(event.clientX+12)+"px";
          tooltip.style.top=(event.clientY-28)+"px";
          tooltip.innerHTML=`<strong>${drug}</strong><br>${param}: ${err.toFixed(1)}% error`;
        }).on("mouseleave",()=>{ tooltip.style.display="none"; });

      if (cellH > 14 && cellW > 22) {
        svg.append("text")
          .attr("x", M.left + ci * cellW + cellW / 2)
          .attr("y", M.top + ri * cellH + cellH / 2 + 4)
          .attr("text-anchor","middle").style("fill","rgba(0,0,0,0.75)").style("font-size","8px").style("font-weight","600")
          .text(err.toFixed(0)+"%");
      }
    });
    svg.append("text").attr("x",M.left-4).attr("y",M.top + ri*cellH + cellH/2+4)
      .attr("text-anchor","end").style("fill","#c5d8f0").style("font-size","9px").text(drug);
  });

  // Column labels
  params.forEach((p, ci) => {
    svg.append("text")
      .attr("transform","rotate(-30)")
      .attr("x", -(M.top - 6))
      .attr("y", M.left + ci * cellW + cellW / 2 + 4)
      .attr("text-anchor","end").style("fill","#7b8fb5").style("font-size","8px")
      .text(p);
  });

  // Colour legend
  svg.append("text").attr("x",M.left).attr("y",H).style("fill","rgba(76,175,80,0.7)").style("font-size","9px").text("■ Green = Low error");
  svg.append("text").attr("x",M.left+130).attr("y",H).style("fill","rgba(244,67,54,0.7)").style("font-size","9px").text("■ Red = High error");
}

/* ── Model R² Chart ── */
function drawModelChart(models, litBench) {
  const container = document.getElementById("bm-model-chart");
  const W = container.clientWidth || 320;
  const H = 240;
  const M = { top: 30, right: 20, bottom: 40, left: 110 };

  const svg = d3.select("#bm-model-chart").append("svg")
    .attr("width","100%").attr("height",H).attr("viewBox",`0 0 ${W} ${H}`);

  const filtered = models.filter(m => m.r2 > -2);
  const y = d3.scaleBand().domain(filtered.map(m=>m.name)).range([M.top,H-M.bottom]).padding(0.3);
  const x = d3.scaleLinear().domain([-0.15, 1.0]).range([M.left, W-M.right]);

  // Literature band
  svg.append("rect").attr("x",x(litBench.r2_lo)).attr("y",M.top)
    .attr("width",x(litBench.r2_hi)-x(litBench.r2_lo)).attr("height",H-M.top-M.bottom)
    .attr("fill","rgba(255,193,7,0.08)").attr("stroke","rgba(255,193,7,0.3)").attr("stroke-dasharray","3,2");
  svg.append("text").attr("x",(x(litBench.r2_lo)+x(litBench.r2_hi))/2).attr("y",M.top-6)
    .attr("text-anchor","middle").style("fill","rgba(255,193,7,0.6)").style("font-size","9px")
    .text("Lit. range");

  // Zero line
  svg.append("line").attr("x1",x(0)).attr("y1",M.top).attr("x2",x(0)).attr("y2",H-M.bottom)
    .attr("stroke","rgba(255,255,255,0.15)").attr("stroke-dasharray","2,2");

  svg.append("g").attr("transform",`translate(0,${H-M.bottom})`)
    .call(d3.axisBottom(x).ticks(5).tickFormat(d=>d.toFixed(1)))
    .selectAll("text").style("fill","#6a7f9e").style("font-size","9px");
  svg.append("g").attr("transform",`translate(${M.left},0)`)
    .call(d3.axisLeft(y).tickSize(0))
    .selectAll("text").style("fill","#c5d8f0").style("font-size","10px");

  const colScale = d3.scaleLinear().domain([-0.2, 0, 0.5, 1]).range(["#f44336","#ff9800","#00bcd4","#4caf50"]);
  const tooltip = document.getElementById("bm-tooltip");

  svg.selectAll(".bm-bar").data(filtered).join("rect")
    .attr("class","bm-bar")
    .attr("x", d => d.r2 >= 0 ? x(0) : x(d.r2))
    .attr("y", d => y(d.name))
    .attr("width", d => Math.abs(x(d.r2) - x(0)))
    .attr("height", y.bandwidth()).attr("rx",3)
    .attr("fill", d => colScale(d.r2)).attr("opacity",0.85)
    .on("mousemove",(event,d)=>{
      tooltip.style.display="block";
      tooltip.style.left=(event.clientX+12)+"px";
      tooltip.style.top=(event.clientY-28)+"px";
      const inRange = d.r2 >= litBench.r2_lo && d.r2 <= litBench.r2_hi;
      tooltip.innerHTML=`<strong>${d.name}</strong><br>R² = ${d.r2.toFixed(4)}<br>${inRange?"✓ Within":"✗ Outside"} literature range`;
    }).on("mouseleave",()=>{ tooltip.style.display="none"; });

  svg.selectAll(".bm-val").data(filtered).join("text")
    .attr("class","bm-val")
    .attr("x", d => x(Math.max(d.r2, 0)) + 4)
    .attr("y", d => y(d.name) + y.bandwidth()/2 + 4)
    .style("fill","#c5d8f0").style("font-size","10px")
    .text(d => d.r2.toFixed(2));

  svg.append("text").attr("x",W/2).attr("y",H-4)
    .attr("text-anchor","middle").style("fill","#4a5e7a").style("font-size","10px").text("R² Score");
}

/* ── Detail Table ── */
function buildDetailTable(records) {
  const tbody = document.querySelector("#bm-detail-table tbody");
  tbody.innerHTML = "";
  [...records].sort((a,b) => Math.abs(a.pct_error)-Math.abs(b.pct_error)).forEach(r => {
    const err = Math.abs(r.pct_error);
    const cls = err <= 10 ? "sev-low" : err <= 25 ? "sev-medium" : "sev-high";
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.drug}</td><td>${r.param}</td>
      <td>${r.published.toFixed(2)} ${r.unit||""}</td>
      <td>${r.simulated.toFixed(2)} ${r.unit||""}</td>
      <td class="${cls}" style="font-weight:600">${r.pct_error.toFixed(1)}%</td>`;
    tbody.appendChild(tr);
  });
}

window.addEventListener("DOMContentLoaded", () => {
  loadBenchmarkData();
  document.querySelectorAll(".nav-link").forEach(a => {
    if (a.getAttribute("href") === window.location.pathname) a.classList.add("active");
    else a.classList.remove("active");
  });
});
