/* ================================================================
   pathogen_organs.js — Objective 4: Pathogen-Organ Interaction Viz
   ================================================================ */

let _poData = null;
let _currentPathogenIdx = 0;

function setPathogen(btn, type) {
  document.querySelectorAll(".ch4-btn[onclick*='setPathogen']").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  _currentPathogenIdx = type === "systemic" ? 1 : 0;
  if (_poData) drawNetworkGraph(_poData, _currentPathogenIdx);
}

async function loadPOData() {
  const resp = await fetch("/api/pathogen-organs-data");
  _poData = await resp.json();

  document.querySelectorAll(".ch4-spinner").forEach(el => el.style.display = "none");

  drawNetworkGraph(_poData, 0);
  drawInteractionMatrix(_poData);
  buildBiomarkerPanel(_poData.biomarker_refs);
  drawDrugOrganHeatmap(_poData.drug_organ_matrix);
}

/* ── Force-directed Organ Network ── */
function drawNetworkGraph(data, pathogenIdx) {
  d3.select("#po-network-chart").selectAll("svg").remove();

  const container = document.getElementById("po-network-chart");
  const W = container.clientWidth || 400;
  const H = 340;

  const svg = d3.select("#po-network-chart").append("svg")
    .attr("width","100%").attr("height",H).attr("viewBox",`0 0 ${W} ${H}`);

  const tooltip = document.getElementById("po-tooltip");

  // Organ exposure from interaction matrix
  const exposureMap = {};
  (data.organ_names || []).forEach((org, i) => {
    exposureMap[org] = (data.interaction_matrix[pathogenIdx] || [])[i] || 0;
  });

  // Nodes
  const nodes = data.organ_nodes.map(n => ({
    ...n,
    exposure: exposureMap[n.name] || 0,
  }));

  // Edges from organ_edges filtered to our nodes
  const nodeNames = new Set(nodes.map(n => n.name));
  const links = (data.organ_edges || [])
    .filter(e => nodeNames.has(e.source) && nodeNames.has(e.target))
    .map(e => ({
      source: e.source, target: e.target, flow: e.flow_rate, type: e.type,
    }));

  const minBF = d3.min(nodes, d => d.blood_flow) || 1;
  const maxBF = d3.max(nodes, d => d.blood_flow) || 1000;
  const rScale = d3.scaleSqrt().domain([minBF, maxBF]).range([8, 28]);
  const colour = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, 1]);

  // Simulate
  const sim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.name).distance(70).strength(0.5))
    .force("charge", d3.forceManyBody().strength(-180))
    .force("center", d3.forceCenter(W / 2, H / 2))
    .force("collision", d3.forceCollide(d => rScale(d.blood_flow) + 6))
    .stop();

  // Run ticks synchronously for static render
  for (let i = 0; i < 150; i++) sim.tick();

  // Edge link width ∝ flow
  const flowScale = d3.scaleLinear().domain([0, d3.max(links, l=>l.flow)||1]).range([0.8, 3.5]);

  svg.append("defs").append("marker")
    .attr("id","arrow").attr("viewBox","0 -4 8 8").attr("refX",8)
    .attr("markerWidth",5).attr("markerHeight",5).attr("orient","auto")
    .append("path").attr("d","M0,-4L8,0L0,4").attr("fill","rgba(0,188,212,0.4)");

  svg.selectAll(".link").data(links).join("line")
    .attr("class","link")
    .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x).attr("y2", d => d.target.y)
    .attr("stroke", "rgba(0,188,212,0.25)")
    .attr("stroke-width", d => flowScale(d.flow))
    .attr("marker-end","url(#arrow)");

  const nodeG = svg.selectAll(".organ-node").data(nodes).join("g")
    .attr("class","organ-node")
    .attr("transform", d => `translate(${d.x},${d.y})`)
    .style("cursor","pointer");

  nodeG.append("circle")
    .attr("r", d => rScale(d.blood_flow))
    .attr("fill", d => colour(d.exposure))
    .attr("stroke", "rgba(255,255,255,0.2)")
    .attr("stroke-width", 1.5)
    .on("mousemove", (event, d) => {
      tooltip.style.display = "block";
      tooltip.style.left = (event.clientX + 12) + "px";
      tooltip.style.top  = (event.clientY - 28) + "px";
      tooltip.innerHTML = `<strong>${d.name}</strong><br>
        Blood flow: ${d.blood_flow} mL/min<br>
        Viral exposure: ${(d.exposure * 100).toFixed(0)}%<br>
        Health: ${(d.health * 100).toFixed(0)}%`;
    }).on("mouseleave", () => { tooltip.style.display = "none"; });

  nodeG.append("text")
    .attr("dy", d => rScale(d.blood_flow) + 12)
    .attr("text-anchor","middle")
    .style("fill","#c5d8f0")
    .style("font-size","9px")
    .text(d => d.name.charAt(0).toUpperCase() + d.name.slice(1));

  // Colour bar
  const barH = 100, barW = 10, barX = W - 20, barY = H / 2 - barH / 2;
  const defs2 = svg.append("defs");
  const grad = defs2.append("linearGradient").attr("id","net-grad")
    .attr("gradientUnits","userSpaceOnUse").attr("x1",0).attr("y1",barY+barH).attr("x2",0).attr("y2",barY);
  [0,0.25,0.5,0.75,1].forEach(s => grad.append("stop").attr("offset",s).attr("stop-color",colour(s)));
  svg.append("rect").attr("x",barX).attr("y",barY).attr("width",barW).attr("height",barH)
    .attr("fill","url(#net-grad)").attr("rx",2);
  svg.append("text").attr("x",barX+barW+2).attr("y",barY+8).style("fill","#6a7f9e").style("font-size","8px").text("High");
  svg.append("text").attr("x",barX+barW+2).attr("y",barY+barH).style("fill","#6a7f9e").style("font-size","8px").text("Low");
  svg.append("text").attr("x",barX+barW/2).attr("y",barY-4)
    .attr("text-anchor","middle").style("fill","#4a5e7a").style("font-size","8px").text("Exposure");
}

/* ── Pathogen × Organ Matrix ── */
function drawInteractionMatrix(data) {
  const container = document.getElementById("po-matrix-chart");
  const W = container.clientWidth || 300;
  const H = 340;
  const M = { top: 30, right: 20, bottom: 80, left: 160 };

  const svg = d3.select("#po-matrix-chart").append("svg")
    .attr("width","100%").attr("height",H).attr("viewBox",`0 0 ${W} ${H}`);

  const organs = data.organ_names;
  const pathogens = data.pathogens.map(p => p.name);
  const matrix = data.interaction_matrix;

  const cellW = (W - M.left - M.right) / organs.length;
  const cellH = (H - M.top - M.bottom) / pathogens.length;
  const colour = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, 1]);
  const tooltip = document.getElementById("po-tooltip");

  pathogens.forEach((pth, ri) => {
    organs.forEach((org, ci) => {
      const val = matrix[ri][ci];
      svg.append("rect")
        .attr("x", M.left + ci * cellW)
        .attr("y", M.top + ri * cellH)
        .attr("width", cellW - 1).attr("height", cellH - 1)
        .attr("fill", colour(val)).attr("rx",2)
        .on("mousemove",(event)=>{
          tooltip.style.display="block";
          tooltip.style.left=(event.clientX+12)+"px";
          tooltip.style.top=(event.clientY-28)+"px";
          tooltip.innerHTML=`<strong>${pth}</strong><br>${org}: ${(val*100).toFixed(0)}%`;
        }).on("mouseleave",()=>{ tooltip.style.display="none"; });

      // Value text if cell big enough
      if (cellW > 20 && cellH > 16) {
        svg.append("text")
          .attr("x", M.left + ci * cellW + cellW / 2)
          .attr("y", M.top + ri * cellH + cellH / 2 + 4)
          .attr("text-anchor","middle").style("fill","rgba(0,0,0,0.75)").style("font-size","9px").style("font-weight","600")
          .text((val * 100).toFixed(0));
      }
    });
    // Row label
    svg.append("text").attr("x", M.left - 6).attr("y", M.top + ri * cellH + cellH / 2 + 4)
      .attr("text-anchor","end").style("fill","#c5d8f0").style("font-size","9px")
      .text(pth.length > 25 ? pth.substring(0,24)+"…" : pth);
  });

  // Column labels
  organs.forEach((org, ci) => {
    svg.append("text")
      .attr("transform", `rotate(-45)`)
      .attr("x", -(M.top + pathogens.length * cellH + 8))
      .attr("y", M.left + ci * cellW + cellW / 2 + 4)
      .attr("text-anchor","end").style("fill","#7b8fb5").style("font-size","9px")
      .text(org);
  });
}

/* ── Biomarker Clinical Panel ── */
function buildBiomarkerPanel(refs) {
  const container = document.getElementById("po-biomarker-panel");
  container.innerHTML = "";
  const ORGAN_MAP = {
    ALT:"Liver",AST:"Liver",CK:"Muscle",GFR:"Kidney",
    WBC:"Immune",CRP:"Systemic",Troponin:"Heart","IL-6":"Immune"
  };
  Object.entries(refs).forEach(([bm, r]) => {
    const div = document.createElement("div");
    div.style.cssText = "background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:8px;padding:12px 14px";
    div.innerHTML = `
      <div style="font-size:13px;font-weight:600;color:#c5d8f0;margin-bottom:2px">${bm}</div>
      <div style="font-size:10px;color:#4a5e7a;margin-bottom:8px">${ORGAN_MAP[bm]||"—"} · ${r.unit}</div>
      <div style="height:6px;background:rgba(255,255,255,0.06);border-radius:3px;position:relative;margin-bottom:8px">
        <div style="position:absolute;left:${20}%;width:${40}%;height:100%;background:rgba(76,175,80,0.5);border-radius:3px"></div>
        <div style="position:absolute;left:${70}%;width:${15}%;height:100%;background:rgba(255,152,0,0.5);border-radius:3px"></div>
        <div style="position:absolute;left:${85}%;right:0;height:100%;background:rgba(244,67,54,0.5);border-radius:3px"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;color:#6a7f9e">
        <span>Normal: ${r.lo}–${r.hi}</span><span style="color:#f44336">Crit: >${r.critical}</span>
      </div>`;
    container.appendChild(div);
  });
}

/* ── Drug × Organ Heatmap ── */
function drawDrugOrganHeatmap(matrix) {
  const container = document.getElementById("po-drug-organ-chart");
  const W = container.clientWidth || 900;
  const H = 300;
  const M = { top: 40, right: 20, bottom: 30, left: 120 };

  const drugs  = Object.keys(matrix);
  const orgSet = new Set();
  drugs.forEach(d => Object.keys(matrix[d]).forEach(o => orgSet.add(o)));
  const organs = Array.from(orgSet).sort();

  const svg = d3.select("#po-drug-organ-chart").append("svg")
    .attr("width","100%").attr("height",H).attr("viewBox",`0 0 ${W} ${H}`);

  const cellW = (W - M.left - M.right) / organs.length;
  const cellH = (H - M.top - M.bottom) / drugs.length;
  const colour = d3.scaleDiverging(d3.interpolateRdYlGn).domain([-0.25, 0, 0.25]);
  const tooltip = document.getElementById("po-tooltip");

  drugs.forEach((drug, ri) => {
    organs.forEach((org, ci) => {
      const val = (matrix[drug] || {})[org] || 0;
      svg.append("rect")
        .attr("x", M.left + ci * cellW)
        .attr("y", M.top + ri * cellH)
        .attr("width", cellW - 1).attr("height", cellH - 1)
        .attr("fill", colour(val)).attr("rx",2)
        .on("mousemove",(event)=>{
          tooltip.style.display="block";
          tooltip.style.left=(event.clientX+12)+"px";
          tooltip.style.top=(event.clientY-28)+"px";
          const sign = val > 0 ? "✚ Protective" : val < 0 ? "✖ Harmful" : "Neutral";
          tooltip.innerHTML=`<strong>${drug} → ${org}</strong><br>${sign}: ${(val*100).toFixed(0)}%`;
        }).on("mouseleave",()=>{ tooltip.style.display="none"; });
    });

    // Drug label
    svg.append("text").attr("x", M.left - 6).attr("y", M.top + ri * cellH + cellH / 2 + 4)
      .attr("text-anchor","end").style("fill","#c5d8f0").style("font-size","10px").text(drug);
  });

  // Organ column labels
  organs.forEach((org, ci) => {
    svg.append("text")
      .attr("transform","rotate(-40)")
      .attr("x", -(M.top - 14))
      .attr("y", M.left + ci * cellW + cellW / 2 + 4)
      .attr("text-anchor","end").style("fill","#7b8fb5").style("font-size","9px")
      .text(org);
  });

  // Legend
  svg.append("text").attr("x",M.left).attr("y",H-4)
    .style("fill","rgba(76,175,80,0.7)").style("font-size","10px").text("■ Green = Protective");
  svg.append("text").attr("x",M.left+160).attr("y",H-4)
    .style("fill","rgba(244,67,54,0.7)").style("font-size","10px").text("■ Red = Harmful");
}

window.addEventListener("DOMContentLoaded", () => {
  loadPOData();
  document.querySelectorAll(".nav-link").forEach(a => {
    if (a.getAttribute("href") === window.location.pathname) a.classList.add("active");
    else a.classList.remove("active");
  });
});
