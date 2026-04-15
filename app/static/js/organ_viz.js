/* ================================================================
   organ_viz.js  — D3.js virtual human body with 12 organ nodes
   ================================================================ */

const OrganViz = (() => {
  const ORGANS = [
    { id: "brain",        label: "Brain",         x: 150, y: 42,  r: 20 },
    { id: "heart",        label: "Heart",         x: 118, y: 152, r: 16 },
    { id: "lungs",        label: "Lungs",         x: 182, y: 148, r: 15 },
    { id: "liver",        label: "Liver",         x: 120, y: 210, r: 16 },
    { id: "gut",          label: "Gut",           x: 154, y: 262, r: 14 },
    { id: "kidneys",      label: "Kidneys",       x: 188, y: 218, r: 13 },
    { id: "pancreas",     label: "Pancreas",      x: 100, y: 250, r: 11 },
    { id: "immune_system",label: "Immune",        x: 220, y: 270, r: 12 },
    { id: "muscle",       label: "Muscle",        x: 76,  y: 310, r: 12 },
    { id: "adipose",      label: "Adipose",       x: 228, y: 310, r: 11 },
    { id: "joints",       label: "Joints",        x: 150, y: 360, r: 12 },
    { id: "skin",         label: "Skin",          x: 150, y: 420, r: 12 },
  ];

  // Edges: [from_id, to_id]
  const EDGES = [
    ["brain","heart"],["heart","lungs"],["heart","liver"],
    ["liver","gut"],["liver","kidneys"],["gut","pancreas"],
    ["heart","muscle"],["lungs","immune_system"],["kidneys","immune_system"],
    ["muscle","joints"],["adipose","gut"],["skin","immune_system"],
  ];

  const COLOR = d3.scaleLinear()
    .domain([0, 0.5, 0.75, 1.0])
    .range(["#ef5350","#ef5350","#ffab40","#00e676"])
    .clamp(true);

  let svg, organMap = {}, healthData = {}, heartbeatInterval, bpm = 72;
  let _onOrganClick = () => {};
  let _seriesData = null, _seriesTime = null;
  let _animFrame = null, _scrubberEl = null, _scrubLabelEl = null;

  function init(onOrganClick) {
    _onOrganClick = onOrganClick || _onOrganClick;
    svg = d3.select("#organ-svg");
    svg.selectAll("*").remove();

    const defs = svg.append("defs");
    // Glow filter
    const f = defs.append("filter").attr("id","organ-glow").attr("x","-50%").attr("y","-50%").attr("width","200%").attr("height","200%");
    f.append("feGaussianBlur").attr("stdDeviation","3").attr("result","blur");
    const fmrg = f.append("feMerge");
    fmrg.append("feMergeNode").attr("in","blur");
    fmrg.append("feMergeNode").attr("in","SourceGraphic");

    // Body silhouette (simple SVG path)
    svg.append("path")
      .attr("d", _bodySilhouette())
      .attr("fill","rgba(255,255,255,0.03)")
      .attr("stroke","rgba(255,255,255,0.08)")
      .attr("stroke-width","1.5");

    // Edges
    const edgeGroup = svg.append("g").attr("class","edges");
    EDGES.forEach(([a, b]) => {
      const na = ORGANS.find(o => o.id === a);
      const nb = ORGANS.find(o => o.id === b);
      if (!na || !nb) return;
      edgeGroup.append("line")
        .attr("class","drug-flow-path")
        .attr("data-a", a).attr("data-b", b)
        .attr("x1", na.x).attr("y1", na.y)
        .attr("x2", nb.x).attr("y2", nb.y)
        .attr("stroke","rgba(0,188,212,0.25)")
        .attr("stroke-dashoffset","0");
    });

    // Organ nodes
    const nodeGroup = svg.append("g").attr("class","nodes");
    ORGANS.forEach(org => {
      const g = nodeGroup.append("g")
        .attr("class","organ-node")
        .attr("transform",`translate(${org.x},${org.y})`)
        .attr("data-id", org.id)
        .style("cursor","pointer")
        .on("click", () => _onOrganClick(org.id, organMap[org.id] || 0.85));

      // Pulse ring
      g.append("circle")
        .attr("class","pulse-ring")
        .attr("r", org.r)
        .attr("fill","none")
        .attr("stroke", COLOR(1))
        .attr("stroke-width","1.5")
        .attr("opacity","0");

      // Main circle
      g.append("circle")
        .attr("class","organ-circle")
        .attr("r", org.r)
        .attr("fill", COLOR(0.85))
        .attr("stroke","rgba(255,255,255,0.15)")
        .attr("stroke-width","1")
        .attr("filter","url(#organ-glow)");

      // Label
      g.append("text")
        .attr("class","organ-label")
        .attr("dy","3.8em")
        .attr("text-anchor","middle")
        .attr("font-size","8.5")
        .attr("fill","rgba(255,255,255,0.55)")
        .attr("font-family","Inter,sans-serif")
        .text(org.label);

      organMap[org.id] = 0.85;
    });

    _startHeartbeat();
  }

  function update(organHealth, drugConcs) {
    if (!svg) return;
    const drugs = drugConcs ? Object.keys(drugConcs) : [];
    const hasDrug = drugs.length > 0;

    // Update organ colours with transition
    ORGANS.forEach(org => {
      const h = organHealth ? (organHealth[org.id] ?? 0.85) : 0.85;
      organMap[org.id] = h;
      const col = COLOR(h);

      svg.select(`[data-id="${org.id}"] .organ-circle`)
        .transition().duration(800)
        .attr("fill", col)
        .attr("stroke", hasDrug ? "rgba(0,188,212,0.6)" : "rgba(255,255,255,0.15)");

      // Update pulse ring colour
      svg.select(`[data-id="${org.id}"] .pulse-ring`)
        .attr("stroke", col);
    });

    // Update edge colours based on drug transport
    svg.selectAll(".drug-flow-path").each(function() {
      const el = d3.select(this);
      const a = el.attr("data-a"), b = el.attr("data-b");
      const avgHealth = ((organMap[a] || 0.85) + (organMap[b] || 0.85)) / 2;
      el.attr("stroke", d3.interpolate("rgba(239,83,80,0.3)","rgba(0,188,212,0.5)")(avgHealth));
    });
  }

  function _startHeartbeat() {
    if (heartbeatInterval) clearInterval(heartbeatInterval);
    const ms = Math.round(60000 / bpm);
    heartbeatInterval = setInterval(() => {
      svg && svg.select('[data-id="heart"] .organ-circle')
        .transition().duration(ms * 0.15).attr("r", 20)
        .transition().duration(ms * 0.25).attr("r", 16);
    }, ms);
  }

  function setHeartRate(hr) {
    bpm = Math.max(30, Math.min(200, hr || 72));
    _startHeartbeat();
  }

  /**
   * Load organ health time-series and start playback animation.
   * @param {Array<Object>} series  - Array of organ_health dicts (one per thinned timestep)
   * @param {Array<number>} times   - Corresponding time values (h)
   */
  function animateSeries(series, times) {
    if (!series || series.length < 2) return;
    _seriesData = series;
    _seriesTime = times;

    // Update scrubber range
    if (_scrubberEl) {
      _scrubberEl.min = 0;
      _scrubberEl.max = series.length - 1;
      _scrubberEl.value = 0;
    }

    // Animate over 4 seconds
    const totalMs = 4000;
    const stepMs  = totalMs / series.length;
    let idx = 0;

    if (_animFrame) cancelAnimationFrame(_animFrame);

    const step = () => {
      if (idx >= series.length) {
        // Show final frame
        _applyFrame(series.length - 1);
        return;
      }
      _applyFrame(idx);
      if (_scrubberEl) _scrubberEl.value = idx;
      if (_scrubLabelEl && times) {
        _scrubLabelEl.textContent = `t = ${(times[idx] || 0).toFixed(1)} h`;
      }
      idx++;
      _animFrame = setTimeout(step, stepMs);
    };
    step();
  }

  /**
   * Apply a single time-series frame (instant, no transition).
   */
  function _applyFrame(idx) {
    if (!_seriesData || !svg) return;
    const frame = _seriesData[Math.min(idx, _seriesData.length - 1)];
    ORGANS.forEach(org => {
      const h = frame[org.id] ?? 0.85;
      organMap[org.id] = h;
      svg.select(`[data-id="${org.id}"] .organ-circle`).attr("fill", COLOR(h));
      svg.select(`[data-id="${org.id}"] .pulse-ring`).attr("stroke", COLOR(h));
    });
    svg.selectAll(".drug-flow-path").each(function() {
      const el = d3.select(this);
      const a = el.attr("data-a"), b = el.attr("data-b");
      const avg = ((organMap[a] || 0.85) + (organMap[b] || 0.85)) / 2;
      el.attr("stroke", d3.interpolate("rgba(239,83,80,0.3)","rgba(0,188,212,0.5)")(avg));
    });
  }

  /**
   * Wire an existing range input as the time-series scrubber.
   * @param {HTMLInputElement} scrubberEl  - The <input type="range"> element
   * @param {HTMLElement}      labelEl     - Element to show current time label
   */
  function setScrubber(scrubberEl, labelEl) {
    _scrubberEl   = scrubberEl;
    _scrubLabelEl = labelEl;
    scrubberEl.addEventListener("input", () => {
      if (_animFrame) { clearTimeout(_animFrame); _animFrame = null; }
      const idx = parseInt(scrubberEl.value);
      _applyFrame(idx);
      if (labelEl && _seriesTime) {
        labelEl.textContent = `t = ${(_seriesTime[idx] || 0).toFixed(1)} h`;
      }
    });
  }

  function _bodySilhouette() {
    // Simplified human body outline
    return `M 150 16
      C 146 16 138 20 136 30
      C 132 40 134 52 136 60
      C 128 62 116 68 110 80
      C 104 92 104 108 106 120
      C 100 122 92 126 88 138
      C 84 150 86 170 90 190
      C 82 192 76 200 74 216
      C 72 240 76 280 80 310
      C 78 330 76 360 74 390
      C 72 420 76 450 80 460
      C 84 462 90 462 94 460
      C 96 430 98 390 100 370
      C 110 380 120 390 130 390
      C 140 390 148 380 150 370
      C 152 380 160 390 170 390
      C 180 390 190 380 200 370
      C 202 390 204 430 206 460
      C 210 462 216 462 220 460
      C 224 450 228 420 226 390
      C 224 360 222 330 220 310
      C 224 280 228 240 226 216
      C 224 200 218 192 210 190
      C 214 170 216 150 212 138
      C 208 126 200 122 194 120
      C 196 108 196 92 190 80
      C 184 68 172 62 164 60
      C 166 52 168 40 164 30
      C 162 20 154 16 150 16 Z`;
  }

  return { init, update, setHeartRate, animateSeries, setScrubber };
})();
