/* ================================================================
   pk_chart.js — D3.js PK concentration line chart
   ================================================================ */

const PKChart = (() => {
  const COLORS = ["#00bcd4","#ffab40","#e040fb","#00e676"];
  let _container, _tooltip, _svg, _g, _w, _h, _margin;

  _margin = { top: 14, right: 22, bottom: 42, left: 54 };

  function init() {
    _container = document.getElementById("pk-chart-container");
    _tooltip   = document.createElement("div");
    _tooltip.className = "tooltip-box";
    _tooltip.style.display = "none";
    _container.style.position = "relative";
    _container.appendChild(_tooltip);
  }

  function render(data) {
    if (!_container || !data || !data.time_points) return;

    const cw = _container.clientWidth  || 600;
    const ch = Math.max(_container.clientHeight || 240, 240);
    _w = cw - _margin.left - _margin.right;
    _h = ch - _margin.top  - _margin.bottom;

    d3.select(_container).select("svg").remove();

    _svg = d3.select(_container).append("svg")
        .attr("width", cw).attr("height", ch);
    _g = _svg.append("g").attr("transform",`translate(${_margin.left},${_margin.top})`);

    const drugs = Object.keys(data.concentrations);
    const times = data.time_points;

    // ── Scales
    const allConcs = drugs.flatMap(d => data.concentrations[d]);
    const allThs   = drugs.flatMap(d => data.therapeutic_windows?.[d] || [0,50]);
    const yMax = Math.max(...allConcs, ...allThs) * 1.18;

    const xScale = d3.scaleLinear().domain([0, times[times.length-1]]).range([0, _w]);
    const yScale = d3.scaleLinear().domain([0, yMax]).range([_h, 0]);

    // ── Grid
    _g.append("g").attr("class","chart-grid")
        .call(d3.axisLeft(yScale).tickSize(-_w).ticks(5).tickFormat(""))
        .selectAll("line").attr("stroke","rgba(255,255,255,0.05)");
    _g.select(".chart-grid .domain").remove();

    // ── Therapeutic windows
    drugs.forEach(dname => {
      const tw = data.therapeutic_windows?.[dname];
      if (!tw) return;
      const [lo, hi] = tw;

      _g.append("rect")
        .attr("x",0).attr("y", yScale(hi))
        .attr("width",_w).attr("height", yScale(lo) - yScale(hi))
        .attr("fill","rgba(0,230,118,0.07)")
        .attr("stroke","rgba(0,230,118,0.2)")
        .attr("stroke-dasharray","4,4");

      _g.append("line")
        .attr("x1",0).attr("y1", yScale(hi*1.5))
        .attr("x2",_w).attr("y2", yScale(hi*1.5))
        .attr("stroke","rgba(239,83,80,0.55)")
        .attr("stroke-dasharray","6,4")
        .attr("stroke-width","1.4");
    });

    // ── Drug lines
    drugs.forEach((dname, i) => {
      const col   = COLORS[i % COLORS.length];
      const concs = data.concentrations[dname];
      const lineData = times.map((t,j) => ({ t, c: concs[j] ?? 0 }));

      const lineFn = d3.line()
        .x(d => xScale(d.t)).y(d => yScale(d.c))
        .curve(d3.curveCatmullRom.alpha(0.5));

      const areaFn = d3.area()
        .x(d => xScale(d.t)).y0(_h).y1(d => yScale(d.c))
        .curve(d3.curveCatmullRom.alpha(0.5));

      _g.append("path").datum(lineData)
        .attr("fill", col).attr("opacity","0.08").attr("d", areaFn);

      const path = _g.append("path").datum(lineData)
        .attr("fill","none").attr("stroke", col)
        .attr("stroke-width","2.5").attr("d", lineFn);

      const totalLen = path.node().getTotalLength();
      path.attr("stroke-dasharray",`${totalLen} ${totalLen}`)
          .attr("stroke-dashoffset", totalLen)
          .transition().duration(900).ease(d3.easeQuadOut)
          .attr("stroke-dashoffset", 0);
    });

    // ── Receptor occupancy overlay
    const hasOcc = data.receptor_occupancy && data.receptor_occupancy.length > 1;
    if (hasOcc) {
      const occTimes = data.series_time_points || times;
      const yScaleR  = d3.scaleLinear().domain([0, 1]).range([_h, 0]);
      const occData  = occTimes.map((t, j) => ({ t, v: data.receptor_occupancy[j] ?? 0 }));

      const occLineFn = d3.line()
        .x(d => xScale(d.t)).y(d => yScaleR(d.v))
        .curve(d3.curveCatmullRom.alpha(0.5));

      _g.append("path").datum(occData)
        .attr("fill","none").attr("stroke","#ff7043")
        .attr("stroke-width","1.8").attr("stroke-dasharray","6,3")
        .attr("opacity","0.85").attr("d", occLineFn);

      _g.append("g")
        .attr("transform",`translate(${_w},0)`)
        .call(d3.axisRight(yScaleR).ticks(4).tickFormat(d => `${Math.round(d*100)}%`))
        .selectAll("text").attr("fill","#ff7043").attr("font-size","9");
    }

    // ── Crosshair + hover tooltip
    const bisect = d3.bisector(t => t).left;
    const crossX = _g.append("line").attr("class","crosshair-x")
        .attr("stroke","rgba(255,255,255,0.2)").attr("stroke-dasharray","4,3")
        .attr("y1",0).attr("y2",_h).attr("display","none");

    _svg.append("rect")
      .attr("width",_w).attr("height",_h)
      .attr("transform",`translate(${_margin.left},${_margin.top})`)
      .attr("fill","none").attr("pointer-events","all")
      .on("mousemove", (event) => {
        const [mx] = d3.pointer(event);
        const t  = xScale.invert(mx - _margin.left);
        const i  = bisect(times, t, 1);
        const t0 = times[Math.max(0, i-1)];
        crossX.attr("x1",mx-_margin.left).attr("x2",mx-_margin.left).attr("display",null);
        let html = `<b style="color:#e8eef7">t = ${t0.toFixed(1)} h</b><br/>`;
        drugs.forEach((dname,k) => {
          const c  = data.concentrations[dname][Math.max(0,i-1)] ?? 0;
          const tw = data.therapeutic_windows?.[dname] || [0,50];
          const status = c < tw[0] ? "sub" : c > tw[1]*1.5 ? "toxic" : c > tw[1] ? "supra" : "ok";
          const sc = status==="ok" ? "#00e676" : status==="toxic" ? "#ef5350" : "#ffab40";
          html += `<span style="color:${COLORS[k%COLORS.length]}">${dname}</span>: <b style="color:${sc}">${c.toFixed(3)}</b> mg/L<br/>`;
        });
        _tooltip.innerHTML = html;
        _tooltip.style.display = "block";
        _tooltip.style.left = `${Math.min(mx+12, cw-160)}px`;
        _tooltip.style.top  = `${_margin.top+8}px`;
      })
      .on("mouseleave", () => {
        crossX.attr("display","none");
        _tooltip.style.display = "none";
      });

    // ── Axes
    _g.append("g").attr("class","chart-axis")
        .attr("transform",`translate(0,${_h})`)
        .call(d3.axisBottom(xScale).ticks(8).tickFormat(d => `${d}h`))
        .selectAll("text").attr("fill","#7b8fb5").attr("font-size","10");

    _g.append("g").attr("class","chart-axis")
        .call(d3.axisLeft(yScale).ticks(5).tickFormat(d => `${d.toFixed(1)}`))
        .selectAll("text").attr("fill","#7b8fb5").attr("font-size","10");

    _g.append("text")
        .attr("transform",`translate(${_w/2},${_h+34})`)
        .attr("text-anchor","middle").attr("fill","#7b8fb5").attr("font-size","10")
        .attr("font-family","Inter,sans-serif").text("Time (hours)");
    _g.append("text")
        .attr("transform","rotate(-90)")
        .attr("y",-42).attr("x",-_h/2)
        .attr("text-anchor","middle").attr("fill","#7b8fb5").attr("font-size","10")
        .attr("font-family","Inter,sans-serif").text("Concentration (mg/L)");

    // ── Legend: each item shows a mini line sample matching the real line style
    const legendEl = document.getElementById("pk-legend");
    legendEl.innerHTML = "";

    const W = 24, H = 10; // swatch dimensions

    function lineSwatch(stroke, dash = "", width = "2.5") {
      return `<svg width="${W}" height="${H}" style="vertical-align:middle;margin-right:5px">
        <line x1="0" y1="${H/2}" x2="${W}" y2="${H/2}"
          stroke="${stroke}" stroke-width="${width}"
          ${dash ? `stroke-dasharray="${dash}"` : ""}/>
      </svg>`;
    }

    function bandSwatch(stroke) {
      return `<svg width="${W}" height="${H}" style="vertical-align:middle;margin-right:5px">
        <rect x="0" y="1" width="${W}" height="${H-2}" rx="2"
          fill="${stroke}" fill-opacity="0.18"
          stroke="${stroke}" stroke-opacity="0.5" stroke-width="1" stroke-dasharray="3,2"/>
      </svg>`;
    }

    drugs.forEach((d,i) => {
      const col = COLORS[i % COLORS.length];
      legendEl.insertAdjacentHTML("beforeend",
        `<div class="legend-item">${lineSwatch(col)}<span style="color:${col};font-weight:600">${d}</span></div>`);
    });

    legendEl.insertAdjacentHTML("beforeend",
      `<div class="legend-item">${bandSwatch("#00e676")}<span style="color:rgba(0,230,118,0.8)">Therapeutic range</span></div>`);

    legendEl.insertAdjacentHTML("beforeend",
      `<div class="legend-item">${lineSwatch("rgba(239,83,80,0.7)", "6,4", "1.5")}<span style="color:rgba(239,83,80,0.8)">Toxic threshold</span></div>`);

    if (hasOcc) {
      legendEl.insertAdjacentHTML("beforeend",
        `<div class="legend-item">${lineSwatch("#ff7043", "6,3", "1.8")}<span style="color:#ff7043">Receptor occupancy</span></div>`);
    }
  }

  return { init, render };
})();
