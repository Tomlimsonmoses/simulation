/* ================================================================
   longterm.js — Long-term simulation results renderer
   Renders output from POST /api/simulate/longterm
   ================================================================ */

const LongTerm = (() => {
  const COLORS = ["#00bcd4","#ffab40","#e040fb","#00e676","#ef5350","#ff7043","#b388ff","#80cbc4"];
  const MARGIN  = { top: 16, right: 24, bottom: 44, left: 56 };

  /* ── util ── */
  function _timeLabel(h) {
    if (h < 48)        return h.toFixed(0) + "h";
    const d = h / 24;
    if (d < 14)        return "Day " + d.toFixed(0);
    const w = d / 7;
    if (w < 9)         return "Wk " + w.toFixed(0);
    const mo = d / 30.4;
    if (mo < 18)       return mo.toFixed(0) + "mo";
    return (d / 365).toFixed(1) + "yr";
  }

  function _sparseTicks(time_h, n) {
    if (!time_h || time_h.length <= n) return (time_h || []).map((_, i) => i);
    const step = (time_h.length - 1) / (n - 1);
    return Array.from({ length: n }, (_, i) => Math.round(i * step));
  }

  /* ── D3 multi-line chart ── */
  function _drawLines(containerId, time_h, series, opts = {}) {
    const el = document.getElementById(containerId);
    if (!el || !time_h || time_h.length === 0) return;

    const cw = el.clientWidth  || 500;
    const ch = opts.h || Math.max(el.clientHeight || 220, 180);
    const w  = cw - MARGIN.left - MARGIN.right;
    const h  = ch - MARGIN.top  - MARGIN.bottom;

    d3.select(el).select("svg").remove();
    const svg = d3.select(el).append("svg").attr("width", cw).attr("height", ch);
    const g   = svg.append("g").attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    const allVals = series.flatMap(s => (s.values || []).filter(v => isFinite(v)));
    const yMin  = opts.yMin != null ? opts.yMin : (allVals.length ? Math.min(...allVals) : 0);
    const yMax  = allVals.length ? Math.max(...allVals) * 1.12 : 1;

    const xScale = d3.scaleLinear().domain([0, time_h[time_h.length - 1]]).range([0, w]);
    const yScale = d3.scaleLinear().domain([yMin, yMax || 1]).range([h, 0]).nice();

    // grid
    g.append("g").attr("class", "chart-grid")
      .call(d3.axisLeft(yScale).tickSize(-w).ticks(4).tickFormat(""))
      .selectAll("line").attr("stroke", "rgba(255,255,255,0.05)");
    g.select(".chart-grid .domain").remove();

    // lines
    series.forEach((s, i) => {
      const col = COLORS[i % COLORS.length];
      const lineData = time_h.map((t, j) => ({ t, v: (s.values || [])[j] }));
      const lineFn = d3.line()
        .defined(d => d.v != null && isFinite(d.v))
        .x(d => xScale(d.t))
        .y(d => yScale(Math.min(yMax, Math.max(yMin, d.v))))
        .curve(d3.curveCatmullRom.alpha(0.5));
      g.append("path").datum(lineData)
        .attr("fill", "none").attr("stroke", col)
        .attr("stroke-width", "1.8").attr("d", lineFn);
    });

    // x-axis sparse ticks
    const tickIdxs = _sparseTicks(time_h, 7);
    const tickVals = tickIdxs.map(i => time_h[i]);
    const tickFmt  = tickIdxs.map(i => _timeLabel(time_h[i]));
    g.append("g").attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(xScale).tickValues(tickVals).tickFormat((_, i) => tickFmt[i]))
      .selectAll("text").attr("fill", "#7b8fb5").attr("font-size", "9").attr("font-family", "Inter,sans-serif");
    g.append("g")
      .call(d3.axisLeft(yScale).ticks(4))
      .selectAll("text").attr("fill", "#7b8fb5").attr("font-size", "9").attr("font-family", "Inter,sans-serif");
    g.selectAll(".domain").attr("stroke", "rgba(255,255,255,0.1)");

    // legend when multiple series
    if (series.length > 1) {
      const maxPerRow = Math.floor(w / 80);
      series.forEach((s, i) => {
        if (i >= 8) return;
        const row = Math.floor(i / maxPerRow);
        const col_idx = i % maxPerRow;
        const lx = col_idx * 80 + 4;
        const ly = MARGIN.top + 6 + row * 14;
        const col = COLORS[i % COLORS.length];
        svg.append("line")
          .attr("x1", MARGIN.left + lx).attr("x2", MARGIN.left + lx + 12)
          .attr("y1", ly).attr("y2", ly)
          .attr("stroke", col).attr("stroke-width", 2);
        svg.append("text")
          .attr("x", MARGIN.left + lx + 16).attr("y", ly + 4)
          .attr("fill", col).attr("font-size", "9").attr("font-family", "Inter,sans-serif")
          .text(s.label.length > 10 ? s.label.slice(0, 9) + "…" : s.label);
      });
    }
  }

  /* ── Summary KPIs ── */
  function _renderSummary(data) {
    const s  = data.summary || {};
    const ev = (data.events || []).length;

    const fields = [
      { id: "lt-kpi-bone-loss", val: s.bone_loss_pct_per_year, unit: "%/yr" },
      { id: "lt-kpi-ldl",       val: s.ldl_change_pct,         unit: "%" },
      { id: "lt-kpi-b12",       val: s.b12_depletion_pct,      unit: "%" },
      { id: "lt-kpi-events",    val: ev,                       unit: "" },
    ];

    fields.forEach(f => {
      const el  = document.getElementById(f.id);
      if (!el) return;
      const vEl = el.querySelector(".lt-kpi-val");
      if (!vEl) return;
      if (f.val != null) {
        vEl.textContent = (typeof f.val === "number" ? f.val.toFixed(1) : f.val) + (f.unit ? " " + f.unit : "");
      } else {
        vEl.textContent = "—";
      }
    });

    // secondary detail bar
    const extras = [];
    if (s.cough_onset_h       != null) extras.push("Cough: Day " + Math.round(s.cough_onset_h / 24));
    if (s.myopathy_onset_h    != null) extras.push("Myopathy: Day " + Math.round(s.myopathy_onset_h / 24));
    if (s.infection_cleared_h != null) extras.push("Infxn cleared: " + s.infection_cleared_h.toFixed(0) + "h");
    if (s.peak_crp_mg_l       != null) extras.push("Peak CRP: " + s.peak_crp_mg_l.toFixed(1) + " mg/L");
    const detEl = document.getElementById("lt-kpi-detail");
    if (detEl) detEl.textContent = extras.join(" · ");
  }

  /* ── Plasma concentrations ── */
  function _renderPlasma(data) {
    const plasma = data.plasma_concentrations || {};
    const time_h = data.time_h || [];
    const series = Object.entries(plasma).map(([drug, vals]) => ({ label: drug, values: vals }));
    if (!series.length || !time_h.length) return;
    _drawLines("lt-plasma-chart", time_h, series, { h: 240 });
  }

  /* ── Immune markers ── */
  function _renderImmune(data) {
    const im     = data.immune_markers || {};
    const time_h = data.time_h || [];
    const SHOW   = [
      { key: "crp",               label: "CRP" },
      { key: "wbc",               label: "WBC" },
      { key: "il6",               label: "IL-6" },
      { key: "cortisol",          label: "Cortisol" },
      { key: "inflammation_index",label: "Inflam.Idx" },
    ];
    const series = SHOW.filter(s => im[s.key]?.length).map(s => ({ label: s.label, values: im[s.key] }));
    if (!series.length || !time_h.length) return;
    _drawLines("lt-immune-chart", time_h, series, { h: 240 });
  }

  /* ── Sparkline card builder ── */
  function _sparkCard(label, values, unit) {
    const card = document.createElement("div");
    card.className = "lt-spark-card";

    const ttl = document.createElement("div");
    ttl.className = "lt-spark-title";
    ttl.textContent = label + (unit ? ` (${unit})` : "");
    card.appendChild(ttl);

    const vEl   = document.createElement("div");
    vEl.className = "lt-spark-val";
    const last  = values[values.length - 1];
    const first = values[0];
    vEl.textContent = (last != null && isFinite(last)) ? last.toFixed(2) : "—";
    if (last != null && first != null && isFinite(last) && isFinite(first)) {
      const delta = last - first;
      vEl.classList.add(delta > 0.001 ? "up" : delta < -0.001 ? "down" : "flat");
    }
    card.appendChild(vEl);

    // mini SVG sparkline via D3
    const NS  = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "44");
    svg.setAttribute("viewBox", "0 0 120 44");
    svg.setAttribute("preserveAspectRatio", "none");
    card.appendChild(svg);

    const valid = values.filter(v => isFinite(v));
    if (valid.length > 1) {
      const pad  = 4;
      const xSc  = d3.scaleLinear().domain([0, values.length - 1]).range([pad, 120 - pad]);
      const yMin = Math.min(...valid);
      const yMax = Math.max(...valid);
      const ySc  = d3.scaleLinear().domain([yMin, yMax === yMin ? yMin + 1 : yMax]).range([44 - pad, pad]);
      const pts  = values.map((v, i) => [i, v]).filter(([, v]) => isFinite(v));
      const dPath = pts.map(([i, v], j) =>
        (j === 0 ? "M" : "L") + xSc(i).toFixed(1) + "," + ySc(v).toFixed(1)
      ).join(" ");
      const path = document.createElementNS(NS, "path");
      path.setAttribute("d", dPath);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", "#00bcd4");
      path.setAttribute("stroke-width", "1.6");
      path.setAttribute("vector-effect", "non-scaling-stroke");
      svg.appendChild(path);
    }

    return card;
  }

  /* ── Biomarkers grid ── */
  function _renderBiomarkers(data) {
    const cont = document.getElementById("lt-biomarkers-grid");
    if (!cont) return;
    cont.innerHTML = "";
    const bm = data.biomarkers || {};
    const PRIO  = ["BMD","ALT","AST","CREATININE","B12_LEVEL","CK","GFR","BRADYKININ","LEUKOCYTES","PLATELETS"];
    const UNITS = { BMD:"g/cm²", ALT:"U/L", AST:"U/L", CREATININE:"mg/dL", B12_LEVEL:"pg/mL", CK:"U/L", GFR:"mL/min" };
    const shown = new Set();
    PRIO.forEach(k => {
      if (bm[k]?.length) {
        cont.appendChild(_sparkCard(k.replace(/_/g, " "), bm[k], UNITS[k] || ""));
        shown.add(k);
      }
    });
    Object.entries(bm).forEach(([k, v]) => {
      if (!shown.has(k) && v?.length) {
        cont.appendChild(_sparkCard(k.replace(/_/g, " "), v, ""));
      }
    });
  }

  /* ── Adaptations grid ── */
  function _renderAdaptations(data) {
    const cont = document.getElementById("lt-adaptations-grid");
    if (!cont) return;
    cont.innerHTML = "";
    const ad = data.adaptations || {};
    const PRIO = ["LDL","HDL","GABA_A_R","SERT","BZD_TOLERANCE","CYP3A4","CYP2D6","DOPAMINE_R","BETA_AR","HERG_CHANNEL","ACE2_R","AT1_R"];
    const shown = new Set();
    let count = 0;
    PRIO.forEach(k => {
      if (ad[k]?.length && count < 14) {
        cont.appendChild(_sparkCard(k.replace(/_/g, " "), ad[k], ""));
        shown.add(k);
        count++;
      }
    });
    Object.entries(ad).forEach(([k, v]) => {
      if (count < 14 && !shown.has(k) && v?.length) {
        cont.appendChild(_sparkCard(k.replace(/_/g, " "), v, ""));
        count++;
      }
    });
  }

  /* ── Clinical events ── */
  function _renderEvents(events) {
    const cont = document.getElementById("lt-events-list");
    if (!cont) return;
    cont.innerHTML = "";
    if (!events || events.length === 0) {
      cont.innerHTML = '<div class="lt-no-events">No clinical events recorded</div>';
      return;
    }
    events.forEach(ev => {
      const item = document.createElement("div");
      const type = (ev.event_type || "info").toLowerCase().replace(/\s+/g, "_");
      item.className = "lt-event-item";
      const h = ev.time_h || 0;
      item.innerHTML =
        `<span class="lt-event-time">${_timeLabel(h)}</span>` +
        `<span class="lt-badge-${type}">${(ev.event_type || "INFO").toUpperCase()}</span>` +
        `<span class="lt-event-desc">${ev.description || ""}</span>`;
      cont.appendChild(item);
    });
  }

  /* ── Disease severity sparkline ── */
  function _renderSeverityTrend(data) {
    const cont = document.getElementById("lt-severity-trend");
    if (!cont || !data.disease_severity_series?.length || !data.time_h?.length) return;
    cont.innerHTML = "";
    _drawLines("lt-severity-trend", data.time_h,
      [{ label: "Disease Severity", values: data.disease_severity_series }],
      { h: 120, yMin: 0 });
  }

  /* ── Public API ── */
  function render(data) {
    if (!data) return;
    _renderSummary(data);
    _renderPlasma(data);
    _renderImmune(data);
    _renderBiomarkers(data);
    _renderAdaptations(data);
    _renderEvents(data.events || []);
    _renderSeverityTrend(data);
  }

  return { render };
})();
