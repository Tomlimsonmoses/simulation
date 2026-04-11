/* ================================================================
   radar_chart.js — D3.js organ health spider chart
   ================================================================ */

const RadarChart = (() => {
  const ORGAN_ORDER = [
    "brain","heart","lungs","liver","kidneys",
    "gut","pancreas","immune_system","muscle","adipose","joints","skin"
  ];
  const LABELS = {
    brain:"Brain", heart:"Heart", lungs:"Lungs", liver:"Liver",
    kidneys:"Kidneys", gut:"Gut", pancreas:"Pancreas",
    immune_system:"Immune", muscle:"Muscle", adipose:"Adipose",
    joints:"Joints", skin:"Skin"
  };

  let _base = null; // baseline polygon (before drug)

  function render(organHealth, baselineHealth) {
    const container = document.getElementById("radar-chart-container");
    if (!container || !organHealth) return;

    const cw = container.clientWidth  || 340;
    const ch = Math.max(container.clientHeight || 240, 240);
    const cx = cw / 2, cy = ch / 2;
    const radius = Math.min(cw, ch) / 2 - 44;
    const N = ORGAN_ORDER.length;
    const angleSlice = (2 * Math.PI) / N;

    d3.select(container).select("svg").remove();

    const svg = d3.select(container).append("svg")
        .attr("width", cw).attr("height", ch);
    const g = svg.append("g").attr("transform",`translate(${cx},${cy})`);

    // ── Concentric rings (0.25, 0.5, 0.75, 1.0)
    [0.25, 0.5, 0.75, 1.0].forEach(level => {
      g.append("circle")
        .attr("r", radius * level)
        .attr("fill","none")
        .attr("stroke","rgba(255,255,255,0.06)")
        .attr("stroke-width","1");
      if (level < 1) {
        g.append("text")
          .attr("x", 4).attr("y", -(radius * level) + 3)
          .attr("font-size","8").attr("fill","rgba(255,255,255,0.2)")
          .attr("font-family","Inter,sans-serif")
          .text(`${Math.round(level*100)}%`);
      }
    });

    // ── Spokes
    ORGAN_ORDER.forEach((_, i) => {
      const angle = angleSlice * i - Math.PI / 2;
      g.append("line")
        .attr("x1",0).attr("y1",0)
        .attr("x2", radius * Math.cos(angle))
        .attr("y2", radius * Math.sin(angle))
        .attr("stroke","rgba(255,255,255,0.07)")
        .attr("stroke-width","1");
    });

    // ── Spoke labels
    ORGAN_ORDER.forEach((key, i) => {
      const angle = angleSlice * i - Math.PI / 2;
      const lx = (radius + 18) * Math.cos(angle);
      const ly = (radius + 18) * Math.sin(angle);
      const h = parseFloat((organHealth[key] ?? 0.85).toFixed(2));
      const col = d3.scaleLinear().domain([0,0.5,1]).range(["#ef5350","#ffab40","#00e676"])(h);
      g.append("text")
        .attr("x", lx).attr("y", ly + 3)
        .attr("text-anchor","middle")
        .attr("font-size","8.5").attr("font-family","Inter,sans-serif")
        .attr("fill", col)
        .text(LABELS[key]);
    });

    // ── Polygon helper
    const polygon = (values, rad) => ORGAN_ORDER.map((key, i) => {
      const v = parseFloat(values[key] ?? 0);
      const a = angleSlice * i - Math.PI / 2;
      return [rad * v * Math.cos(a), rad * v * Math.sin(a)];
    });

    const lineClose = d3.line().curve(d3.curveLinearClosed);

    // ── Baseline polygon (previous state or grey)
    if (baselineHealth) {
      g.append("path")
        .datum(polygon(baselineHealth, radius))
        .attr("d", lineClose)
        .attr("fill","rgba(123,143,181,0.08)")
        .attr("stroke","rgba(123,143,181,0.3)")
        .attr("stroke-width","1.5");
    }

    // ── Current polygon
    const currentPts = polygon(organHealth, radius);
    g.append("path")
      .datum(currentPts)
      .attr("d", lineClose)
      .attr("fill","rgba(0,188,212,0.12)")
      .attr("stroke","rgba(0,188,212,0.7)")
      .attr("stroke-width","2.2")
      .attr("opacity", 0)
      .transition().duration(750)
      .attr("opacity", 1);

    // ── Dots on each spoke vertex
    ORGAN_ORDER.forEach((key, i) => {
      const v = parseFloat(organHealth[key] ?? 0.85);
      const a = angleSlice * i - Math.PI / 2;
      const col = d3.scaleLinear().domain([0,0.5,1]).range(["#ef5350","#ffab40","#00e676"])(v);
      g.append("circle")
        .attr("cx", radius * v * Math.cos(a))
        .attr("cy", radius * v * Math.sin(a))
        .attr("r", 3.5)
        .attr("fill", col)
        .attr("stroke","#070b14")
        .attr("stroke-width","1.5");
    });

    // Store for next render as baseline
    _base = { ...organHealth };
  }

  return { render, getBaseline: () => _base };
})();
