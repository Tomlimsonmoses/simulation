/* ================================================================
   timeline.js — D3.js Gantt-style side effects timeline
   ================================================================ */

const Timeline = (() => {
  const SEV_COLOR = { high: "#ef5350", medium: "#ffab40", low: "#00e676" };

  function render(sideEffects, durationHours) {
    const container = document.getElementById("timeline-container");
    if (!container) return;

    if (!sideEffects || sideEffects.length === 0) {
      container.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:120px;color:#3d4e6b;font-size:12px;">No side effects detected in simulation period</div>`;
      return;
    }

    // Deduplicate, keep one bar per unique side-effect name
    const seen = new Map();
    sideEffects.forEach(e => {
      if (!seen.has(e.name)) seen.set(e.name, e);
    });
    const effects = Array.from(seen.values());

    const margin = { top: 12, right: 20, bottom: 34, left: 148 };
    const cw = container.clientWidth  || 520;
    const rowH = 22;
    const ch = effects.length * rowH + margin.top + margin.bottom + 8;

    d3.select(container).select("svg").remove();

    const svg = d3.select(container).append("svg")
        .attr("width", cw).attr("height", ch);
    const g = svg.append("g").attr("transform",`translate(${margin.left},${margin.top})`);
    const W = cw - margin.left - margin.right;

    const xScale = d3.scaleLinear().domain([0, durationHours || 72]).range([0, W]);

    // Grid
    g.append("g").attr("class","chart-grid")
        .call(d3.axisBottom(xScale).tickSize(effects.length * rowH).ticks(8).tickFormat(""))
        .selectAll("line").attr("stroke","rgba(255,255,255,0.04)");
    g.select(".chart-grid .domain").remove();

    // Bars
    effects.forEach((e, i) => {
      const y = i * rowH + 2;
      const col = SEV_COLOR[e.severity] || "#7b8fb5";
      const x0 = xScale(Math.min(e.onset_time, durationHours));
      const x1 = xScale(Math.min(e.onset_time + e.duration, durationHours));
      const bw = Math.max(x1 - x0, 4);

      // Background row
      g.append("rect")
        .attr("x",0).attr("y", y)
        .attr("width", W).attr("height", rowH - 3)
        .attr("fill", i % 2 === 0 ? "rgba(255,255,255,0.01)" : "transparent");

      // Bar
      g.append("rect")
        .attr("x", x0).attr("y", y + 2)
        .attr("width", 0).attr("height", rowH - 7)
        .attr("rx","3")
        .attr("fill", col).attr("opacity","0.75")
        .transition().delay(i * 40).duration(600)
        .attr("width", bw);

      // Probability overlay
      g.append("rect")
        .attr("x", x0).attr("y", y + 2)
        .attr("width", bw * Math.min(e.probability * 4, 1))
        .attr("height", rowH - 7)
        .attr("rx","3")
        .attr("fill", col).attr("opacity","0.30");

      // Label
      g.append("text")
        .attr("x",-6).attr("y", y + rowH/2 + 1)
        .attr("text-anchor","end").attr("alignment-baseline","middle")
        .attr("font-size","10").attr("fill","#7b8fb5")
        .attr("font-family","Inter,sans-serif")
        .text(e.name);

      // Severity badge
      g.append("text")
        .attr("x", x0 + bw / 2).attr("y", y + rowH/2 + 1)
        .attr("text-anchor","middle").attr("alignment-baseline","middle")
        .attr("font-size","8.5").attr("fill","rgba(0,0,0,0.75)").attr("font-weight","700")
        .attr("font-family","Inter,sans-serif")
        .text(e.severity.toUpperCase());
    });

    // X axis
    g.append("g").attr("class","chart-axis")
        .attr("transform",`translate(0,${effects.length * rowH})`)
        .call(d3.axisBottom(xScale).ticks(8).tickFormat(d => `${d}h`))
        .selectAll("text").attr("fill","#7b8fb5").attr("font-size","10");

    // Legend
    const legG = svg.append("g").attr("transform",`translate(${margin.left},${ch - 14})`);
    [{s:"high",l:"High"},{s:"medium",l:"Medium"},{s:"low",l:"Low"}].forEach((item, i) => {
      legG.append("rect").attr("x",i * 72).attr("y",0).attr("width",10).attr("height",10).attr("rx","2").attr("fill",SEV_COLOR[item.s]);
      legG.append("text").attr("x",i*72+14).attr("y",9).attr("font-size","9").attr("fill","#7b8fb5").attr("font-family","Inter,sans-serif").text(item.l+" Severity");
    });
  }

  return { render };
})();
