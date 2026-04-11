/* ================================================================
   cell_view.js — D3.js cell cross-section with live drug animation
   ================================================================ */

const CellView = (() => {
  let _organName = "", _drugMolecules = [], _animFrame = null;

  // ── Open / close drawer ────────────────────────────────────────────
  function open(organId, organHealth, simData, drugNames) {
    _organName = organId;
    const drawer  = document.getElementById("cell-drawer");
    const overlay = document.getElementById("drawer-overlay");
    const nameEl  = document.getElementById("cell-organ-name");
    nameEl.textContent = organId.replace(/_/g," ").replace(/\b\w/g,c=>c.toUpperCase());
    drawer.classList.add("open");
    overlay.classList.add("visible");
    _fetchAndRender(organId, organHealth, simData, drugNames);
  }

  function close() {
    document.getElementById("cell-drawer").classList.remove("open");
    document.getElementById("drawer-overlay").classList.remove("visible");
    if (_animFrame) { cancelAnimationFrame(_animFrame); _animFrame = null; }
  }

  // ── Fetch /api/cell then render ───────────────────────────────────
  async function _fetchAndRender(organId, organHealth, simData, drugNames) {
    const drug1  = (drugNames && drugNames[0]) || "Lisinopril";
    const drug2  = (drugNames && drugNames[1]) || null;
    const dose1  = simData?._doses?.[drug1]  || 10;
    const dose2  = simData?._doses?.[drug2]  || 10;

    let cellData = null;
    try {
      const resp = await fetch("/api/cell", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({
          organ: organId, organ_health: organHealth,
          drug: drug1, dose: dose1,
          drug2: drug2 || null, dose2: dose2 || null,
        }),
      });
      cellData = await resp.json();
    } catch(e) {
      cellData = _mockCellData(organId, organHealth, drug1);
    }
    _renderCellPanel(cellData);
  }

  // ── Render cell panel ──────────────────────────────────────────────
  function _renderCellPanel(data) {
    const cd = data.cell_data;

    // Health ring
    _drawHealthRing(
      document.getElementById("cell-health-ring"),
      data.health, 60, 10
    );
    document.getElementById("cell-health-pct").textContent = Math.round(data.health * 100) + "%";

    // Cell SVG cross-section
    _drawCellSVG(data);

    // Metric bars
    _setBar("membrane", cd.membrane_integrity);
    _setBar("atp",      cd.atp_level > cd.mitochondrial_activity ? cd.atp_level : cd.mitochondrial_activity);
    _setBar("receptor", cd.receptor_occupancy);
    _setBar("apoptosis",cd.apoptosis_risk);

    // Recovery indicator
    const ri = document.getElementById("recovery-indicator");
    if (cd.recovery_rate > 0.001) {
      ri.className  = "recovery-indicator";
      ri.textContent = `🔄 Recovering +${(cd.recovery_rate*100).toFixed(2)}%/hr`;
    } else if (cd.recovery_rate < -0.001) {
      ri.className  = "recovery-indicator declining";
      ri.textContent = `⚠ Declining ${(cd.recovery_rate*100).toFixed(2)}%/hr`;
    } else {
      ri.className  = "recovery-indicator";
      ri.textContent = "⚖ Stable — no significant change";
    }

    // Binding table
    const tbody = document.querySelector("#binding-table tbody");
    tbody.innerHTML = "";
    (data.drug_binding_events || []).forEach(e => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${e.receptor}</td><td>${e.drug}</td><td>${(e.occupancy*100).toFixed(1)}%</td><td>${e.effect}</td>`;
      tbody.appendChild(tr);
    });
    if (!(data.drug_binding_events||[]).length) {
      tbody.innerHTML = `<tr><td colspan="4" style="color:#3d4e6b;text-align:center">No binding events detected</td></tr>`;
    }
  }

  function _setBar(id, val) {
    const fill = document.getElementById(`bar-${id}`);
    const valEl = document.getElementById(`val-${id}`);
    if (!fill) return;
    const pct = Math.round(Math.max(0,Math.min(1,val||0)) * 100);
    fill.style.width = pct + "%";
    if (valEl) valEl.textContent = pct + "%";
  }

  // ── Cell cross-section SVG ─────────────────────────────────────────
  function _drawCellSVG(data) {
    const cd  = data.cell_data;
    const svg = d3.select("#cell-svg");
    svg.selectAll("*").remove();
    if (_animFrame) { cancelAnimationFrame(_animFrame); _animFrame = null; }

    const W = 360, H = 360, cx = W/2, cy = H/2;
    const defs = svg.append("defs");

    // ── Glow filters
    const glowFn = (id, col, std=4) => {
      const f = defs.append("filter").attr("id",id).attr("x","-50%").attr("y","-50%").attr("width","200%").attr("height","200%");
      f.append("feGaussianBlur").attr("stdDeviation",std).attr("result","blur");
      const m = f.append("feMerge");
      m.append("feMergeNode").attr("in","blur");
      m.append("feMergeNode").attr("in","SourceGraphic");
    };
    glowFn("cell-glow","#00bcd4",5);
    glowFn("nucleus-glow","#7c4dff",6);
    glowFn("mito-glow","#ff6e40",3);

    // ── Recovery radial gradient
    if (cd.recovery_rate > 0.001) {
      const rg = defs.append("radialGradient").attr("id","recover-grad").attr("cx","50%").attr("cy","50%").attr("r","50%");
      rg.append("stop").attr("offset","0%").attr("stop-color","#00e676").attr("stop-opacity","0.18");
      rg.append("stop").attr("offset","100%").attr("stop-color","#00e676").attr("stop-opacity","0");
    }

    // ─ Outer region (extracellular space)
    svg.append("rect").attr("width",W).attr("height",H).attr("fill","#040b16");

    // ─ Cell membrane
    const memCol = d3.scaleLinear().domain([0,0.5,1]).range(["#ef5350","#ffab40","#00bcd4"])(cd.membrane_integrity);
    svg.append("ellipse")
      .attr("cx",cx).attr("cy",cy).attr("rx",140).attr("ry",130)
      .attr("fill","rgba(0,188,212,0.06)")
      .attr("stroke", memCol).attr("stroke-width","3.5")
      .attr("filter","url(#cell-glow)");

    // Recovery glow
    if (cd.recovery_rate > 0.001) {
      svg.append("ellipse")
        .attr("cx",cx).attr("cy",cy).attr("rx",140).attr("ry",130)
        .attr("fill","url(#recover-grad)").attr("opacity","0.7");
    }

    // ─ Phospholipid heads (dots around the membrane)
    const lipidCount = 48;
    for (let i = 0; i < lipidCount; i++) {
      const angle = (2 * Math.PI * i) / lipidCount;
      const rx = 140, ry = 130;
      const lx = cx + rx * Math.cos(angle);
      const ly = cy + ry * Math.sin(angle);
      svg.append("circle")
        .attr("cx", lx).attr("cy", ly).attr("r", 3.5)
        .attr("fill", memCol).attr("opacity","0.65");
    }

    // ─ Cytoplasm background
    svg.append("ellipse")
      .attr("cx",cx).attr("cy",cy).attr("rx",136).attr("ry",126)
      .attr("fill","rgba(0,20,50,0.7)");

    // ─ Nucleus
    const nucCol = d3.scaleLinear().domain([0,1]).range(["#9c27b0","#7c4dff"])(cd.atp_level);
    svg.append("ellipse")
      .attr("cx",cx).attr("cy",cy-10).attr("rx",50).attr("ry",44)
      .attr("fill","rgba(80,0,120,0.45)")
      .attr("stroke",nucCol).attr("stroke-width","2")
      .attr("filter","url(#nucleus-glow)");

    // DNA double-helix strands inside nucleus (simplified)
    for (let side of [-1,1]) {
      const pts = d3.range(-44, 44, 5).map(yy => {
        const xx = side * 14 * Math.sin((yy / 44) * Math.PI * 2.5);
        return [cx + xx, cy - 10 + yy];
      });
      svg.append("path")
        .attr("d", d3.line().curve(d3.curveCatmullRom)(pts))
        .attr("fill","none").attr("stroke",nucCol).attr("stroke-width","1.5").attr("opacity","0.6");
    }
    // Rungs
    for (let y = -40; y < 44; y += 8) {
      const x1 = cx - 14 * Math.sin((y/44)*Math.PI*2.5);
      const x2 = cx + 14 * Math.sin((y/44)*Math.PI*2.5);
      svg.append("line").attr("x1",x1).attr("y1",cy-10+y).attr("x2",x2).attr("y2",cy-10+y)
        .attr("stroke",nucCol).attr("stroke-width","1").attr("opacity","0.4");
    }

    // ─ Mitochondria (3 beans)
    const mitoPositions = [
      {x: cx - 82, y: cy + 30, rot: 30},
      {x: cx + 75, y: cy + 25, rot: -20},
      {x: cx - 30, y: cy + 80, rot: 5},
    ];
    const mitoCol = d3.scaleLinear().domain([0,1]).range(["#b71c1c","#ff6d00","#ff6e40"])(cd.mitochondrial_activity);
    mitoPositions.forEach(m => {
      const mg = svg.append("g").attr("transform",`translate(${m.x},${m.y}) rotate(${m.rot})`);
      mg.append("ellipse").attr("rx",22).attr("ry",12).attr("fill","rgba(255,110,64,0.18)")
        .attr("stroke",mitoCol).attr("stroke-width","1.5").attr("filter","url(#mito-glow)");
      // Cristae
      for (let ci = -12; ci <= 12; ci += 5) {
        mg.append("line").attr("x1",ci).attr("y1",-12).attr("x2",ci).attr("y2",12)
          .attr("stroke",mitoCol).attr("stroke-width","0.8").attr("opacity","0.5");
      }
    });

    // ─ Damage particles (apoptosis)
    if (cd.apoptosis_risk > 0.15) {
      const nPart = Math.floor(cd.apoptosis_risk * 14);
      const rng = (min,max) => min + Math.random()*(max-min);
      for (let p = 0; p < nPart; p++) {
        const angle = Math.random() * Math.PI * 2;
        const dr = 30 + Math.random() * 90;
        const px = cx + dr * Math.cos(angle);
        const py = cy + dr * Math.sin(angle);
        svg.append("polygon")
          .attr("points", _starPoints(px, py, 4, 4))
          .attr("fill","#ef5350").attr("opacity", cd.apoptosis_risk * 0.8);
      }
    }

    // ─ Receptor sites on membrane
    const receptorsData = data.drug_binding_events || [];
    const recAngles = receptorsData.map((_,i) => (2*Math.PI*i)/Math.max(receptorsData.length,1) - Math.PI/2);
    const recGroup = svg.append("g").attr("class","receptors");
    recAngles.forEach((angle, i) => {
      const rx = 140, ry = 130;
      const recX = cx + (rx - 6) * Math.cos(angle);
      const recY = cy + (ry - 5) * Math.sin(angle);
      const occ  = receptorsData[i].occupancy || 0;
      const recCol = occ > 0.01 ? "#00bcd4" : "#3d4e6b";
      recGroup.append("polygon")
        .attr("points", _starPoints(recX, recY, 6, 4))
        .attr("fill", recCol)
        .attr("opacity", 0.5 + occ * 0.5)
        .attr("class","receptor-site")
        .attr("data-idx", i);
    });

    // ─ Animate drug molecules floating to receptors
    _drugMolecules = [];
    if (receptorsData.length > 0) {
      _animateDrugMolecules(svg, cx, cy, receptorsData, recAngles, W, H);
    }
  }

  function _animateDrugMolecules(svg, cx, cy, receptorsData, recAngles, W, H) {
    const DRUG_COLORS = { Lisinopril:"#00bcd4", Metoprolol:"#7c4dff", Ibuprofen:"#ffab40",
      Atorvastatin:"#00e676", Metformin:"#e040fb", default:"#90a4ae" };

    const molecules = receptorsData.filter(e => e.occupancy > 0.02).slice(0,8);
    const molGroup  = svg.append("g").attr("class","drug-molecules");

    molecules.forEach((e, mi) => {
      const angle  = recAngles[mi % recAngles.length];
      const endX   = cx + 134 * Math.cos(angle);
      const endY   = cy + 124 * Math.sin(angle);
      // Start outside the cell
      const startX = cx + (170 + Math.random()*20) * Math.cos(angle + (Math.random()-0.5)*0.5);
      const startY = cy + (158 + Math.random()*20) * Math.sin(angle + (Math.random()-0.5)*0.5);

      const col = DRUG_COLORS[e.drug] || DRUG_COLORS.default;
      const mol = molGroup.append("circle")
        .attr("cx", startX).attr("cy", startY).attr("r", 4.5)
        .attr("fill", col).attr("opacity","0.9")
        .attr("filter","url(#cell-glow)");

      // Oscillating path towards receptor site, looping
      function moveMolecule() {
        mol.attr("cx", startX).attr("cy", startY).attr("opacity","0.9")
          .transition()
          .duration(1800 + mi * 200).ease(d3.easeCubicInOut)
          .attr("cx", endX).attr("cy", endY)
          .on("end", () => {
            mol.transition().duration(400).attr("opacity","0")
              .on("end", () => {
                mol.attr("opacity","0");
                setTimeout(moveMolecule, 600 + mi * 150);
              });
          });
      }
      setTimeout(moveMolecule, mi * 320);
      _drugMolecules.push(mol);
    });
  }

  // ── Health ring (D3 arc) ───────────────────────────────────────────
  function _drawHealthRing(svgEl, health, outerR, strokeW) {
    d3.select(svgEl).selectAll("*").remove();
    const W = svgEl.viewBox?.baseVal?.width || 120;
    const H = svgEl.viewBox?.baseVal?.height || 120;
    const cx = W/2, cy = H/2;
    const arc = d3.arc().innerRadius(outerR-strokeW).outerRadius(outerR).startAngle(0);
    const col  = d3.scaleLinear().domain([0,0.5,1]).range(["#ef5350","#ffab40","#00e676"])(health);
    const svg  = d3.select(svgEl);
    svg.append("path").datum({endAngle: 2*Math.PI})
      .attr("d", arc).attr("fill","rgba(255,255,255,0.05)")
      .attr("transform",`translate(${cx},${cy})`);
    svg.append("path").datum({endAngle: 0})
      .attr("d", arc).attr("fill", col)
      .attr("transform",`translate(${cx},${cy})`)
      .transition().duration(800)
      .attrTween("d", function(d) {
        const i = d3.interpolate(d.endAngle, 2 * Math.PI * health);
        return t => { d.endAngle = i(t); return arc(d); };
      });
  }

  function _starPoints(cx, cy, outerR, innerR, points=5) {
    const step = Math.PI / points;
    return d3.range(2*points).map(i => {
      const r = i%2===0 ? outerR : innerR;
      const a = i * step - Math.PI/2;
      return `${cx + r*Math.cos(a)},${cy + r*Math.sin(a)}`;
    }).join(" ");
  }

  function _mockCellData(organ, health, drug) {
    return {
      organ, health,
      cell_data: {
        membrane_integrity: health * 0.9,
        atp_level: health * 0.85,
        receptor_occupancy: 0.45,
        drug_concentration_cellular: 8.5,
        mitochondrial_activity: health * 0.88,
        inflammatory_markers: 1 - health * 0.7,
        apoptosis_risk: (1-health)*0.6,
        recovery_rate: health > 0.6 ? 0.012 : -0.008,
      },
      drug_effects: {[drug]: 0.1},
      drug_binding_events: [{receptor:"ACE", drug, occupancy:0.45, effect:"inhibition", drug_conc:8.5}],
    };
  }

  // ── Update after new sim result ────────────────────────────────────
  function update(organId, organHealth, simData, drugNames) {
    const drawer = document.getElementById("cell-drawer");
    if (!drawer.classList.contains("open") || _organName !== organId) return;
    _fetchAndRender(organId, organHealth, simData, drugNames);
  }

  // ── Init event listeners ───────────────────────────────────────────
  function init() {
    document.getElementById("drawer-close")?.addEventListener("click", close);
    document.getElementById("drawer-overlay")?.addEventListener("click", close);
  }

  return { init, open, close, update };
})();
