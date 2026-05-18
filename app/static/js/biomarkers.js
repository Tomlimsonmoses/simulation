/* ================================================================
   biomarkers.js — alert cards + biomarker sparklines
   ================================================================ */

const Biomarkers = (() => {

  const SEV = {
    high:        { color: "#ef5350", bg: "rgba(239,83,80,0.10)",  border: "rgba(239,83,80,0.30)",  label: "HIGH"   },
    interaction: { color: "#ba68c8", bg: "rgba(186,104,200,0.10)", border: "rgba(186,104,200,0.30)", label: "INTERACTION" },
    medium:      { color: "#ffab40", bg: "rgba(255,171,64,0.08)",  border: "rgba(255,171,64,0.25)", label: "MODERATE" },
    low:         { color: "#00e676", bg: "rgba(0,230,118,0.08)",   border: "rgba(0,230,118,0.20)",  label: "LOW"    },
  };

  function render(data) {
    const container = document.getElementById("alerts-container");
    if (!container || !data) return;
    container.innerHTML = "";

    const warnings        = data.warnings        || [];
    const recommendations = data.recommendations || [];

    // ── Toxicity / interaction alerts ──────────────────────────────
    if (warnings.length > 0) {
      const toxCount   = warnings.filter(w => !w.startsWith("INTERACTION")).length;
      const ixnCount   = warnings.filter(w =>  w.startsWith("INTERACTION")).length;
      const chips = [
        toxCount ? `<span style="background:#ef535022;border:1px solid #ef5350;color:#ef5350;border-radius:4px;padding:1px 7px;font-size:10px;font-weight:700">${toxCount} Toxic</span>` : "",
        ixnCount ? `<span style="background:#ba68c822;border:1px solid #ba68c8;color:#ba68c8;border-radius:4px;padding:1px 7px;font-size:10px;font-weight:700">${ixnCount} Drug Interaction</span>` : "",
      ].filter(Boolean).join(" ");

      container.insertAdjacentHTML("beforeend",
        `<div class="alert-section-title" style="display:flex;align-items:center;gap:8px;">
           <span>⚠ Alerts</span>${chips}
         </div>`);

      warnings.forEach(w => {
        const isIxn  = w.startsWith("INTERACTION");
        const key    = isIxn ? "interaction" : "high";
        const s      = SEV[key];
        const raw    = w.replace(/^(INTERACTION|ALERT):\s*/, "");

        // Split "Name — detail" into title + body for better readability
        const dashIdx = raw.indexOf(" — ");
        const title   = dashIdx > -1 ? raw.slice(0, dashIdx) : raw;
        const detail  = dashIdx > -1 ? raw.slice(dashIdx + 3) : "";

        const kb = typeof _seKBLookup === "function" ? _seKBLookup(title) : null;
        container.insertAdjacentHTML("beforeend", `
          <div style="
            background:${s.bg};
            border:1px solid ${s.border};
            border-left:3px solid ${s.color};
            border-radius:6px;
            padding:9px 12px;
            margin-bottom:6px;
            display:flex;
            gap:10px;
            align-items:flex-start;">
            <span style="font-size:15px;line-height:1.3">${isIxn ? "🔀" : "🚨"}</span>
            <div style="flex:1;min-width:0">
              <div style="display:flex;align-items:center;gap:6px;margin-bottom:2px">
                <strong style="color:${s.color};font-size:12px">${title}</strong>
                <span style="background:${s.color};color:#000;border-radius:3px;padding:0 5px;font-size:9px;font-weight:800;letter-spacing:.04em">${s.label}</span>
              </div>
              ${detail ? `<div style="color:#8a9aae;font-size:11px;line-height:1.4;margin-bottom:${kb ? 6 : 0}px">${detail}</div>` : ""}
              ${kb ? `
                <div style="background:rgba(255,171,64,0.06);border:1px solid rgba(255,171,64,0.15);border-radius:4px;padding:5px 8px;margin-bottom:5px">
                  <div style="color:#ffab40;font-size:9px;font-weight:700;letter-spacing:.04em;margin-bottom:2px">⚡ CAUSE</div>
                  <div style="color:#8a9aae;font-size:10.5px;line-height:1.45">${kb.cause}</div>
                </div>
                <div style="background:rgba(0,230,118,0.05);border:1px solid rgba(0,230,118,0.14);border-radius:4px;padding:5px 8px">
                  <div style="color:#00e676;font-size:9px;font-weight:700;letter-spacing:.04em;margin-bottom:2px">✓ HOW TO REDUCE</div>
                  <div style="color:#8a9aae;font-size:10.5px;line-height:1.45">${kb.reduce}</div>
                </div>` : ""}
            </div>
          </div>`);
      });
    }

    // ── Clinical notes ─────────────────────────────────────────────
    if (recommendations.length > 0) {
      container.insertAdjacentHTML("beforeend",
        `<div class="alert-section-title">💊 Clinical Notes</div>`);

      recommendations.slice(0, 5).forEach(r => {
        const s = SEV.medium;
        container.insertAdjacentHTML("beforeend", `
          <div style="
            background:${s.bg};
            border:1px solid ${s.border};
            border-left:3px solid ${s.color};
            border-radius:6px;
            padding:8px 12px;
            margin-bottom:5px;
            display:flex;
            gap:10px;
            align-items:flex-start;">
            <span style="font-size:14px;line-height:1.4">📋</span>
            <span style="color:#c8d8e8;font-size:12px;line-height:1.45">${r}</span>
          </div>`);
      });
    }

    // ── All-clear state ────────────────────────────────────────────
    if (warnings.length === 0 && recommendations.length === 0) {
      const bm = data.biomarkers || {};
      const hasBm = Object.keys(bm).some(k => bm[k] && bm[k].length > 1);
      container.insertAdjacentHTML("beforeend", `
        <div style="
          background:rgba(0,230,118,0.07);
          border:1px solid rgba(0,230,118,0.20);
          border-left:3px solid #00e676;
          border-radius:6px;
          padding:10px 14px;
          display:flex;gap:10px;align-items:center;margin-bottom:8px;">
          <span style="font-size:18px">✅</span>
          <div>
            <strong style="color:#00e676;font-size:12px">No Alerts</strong>
            <div style="color:#7b8fb5;font-size:11px;margin-top:1px">
              ${hasBm ? "Biomarkers are being tracked for early changes." : "Simulation looks clinically safe."}
            </div>
          </div>
        </div>`);
    }

    // ── Biomarker sparklines ────────────────────────────────────────
    const bm     = data.biomarkers || {};
    const bmKeys = Object.keys(bm).filter(k => bm[k] && bm[k].length > 1).slice(0, 6);
    if (bmKeys.length === 0) return;

    container.insertAdjacentHTML("beforeend",
      `<div class="alert-section-title">📈 Biomarkers</div>
       <div class="sparkline-grid" id="sparkline-grid"></div>`);
    const grid = document.getElementById("sparkline-grid");

    bmKeys.forEach(key => {
      const vals  = bm[key];
      const last  = vals[vals.length - 1];
      const first = vals[0];
      const trend = last - first;
      const trendArrow = Math.abs(trend) < 0.5 ? "→" : trend > 0 ? "↑" : "↓";
      const trendColor = Math.abs(trend) < 0.5 ? "#7b8fb5" : trend > 0 ? "#ffab40" : "#00e676";
      const label = key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
      const div   = document.createElement("div");
      div.className = "sparkline-card";
      div.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:flex-start">
          <div class="sparkline-label">${label}</div>
          <span style="color:${trendColor};font-size:13px;font-weight:700;line-height:1">${trendArrow}</span>
        </div>
        <div class="sparkline-val">${_fmt(last, key)}</div>
        <svg class="spark" width="100%" height="32"></svg>`;
      grid.appendChild(div);
      _drawSparkline(div.querySelector(".spark"), vals);
    });
  }

  function _drawSparkline(svgEl, vals) {
    if (!svgEl || vals.length < 2) return;
    const w = svgEl.clientWidth || 120, h = 32;
    const x = d3.scaleLinear().domain([0, vals.length - 1]).range([0, w]);
    const y = d3.scaleLinear().domain([d3.min(vals) * 0.95, d3.max(vals) * 1.05]).range([h - 2, 2]);
    const line = d3.line().x((_, i) => x(i)).y(v => y(v)).curve(d3.curveCatmullRom.alpha(0.5));
    const trend = vals[vals.length - 1] - vals[0];
    const col = Math.abs(trend) < 0.5 ? "#7b8fb5" : trend > 0 ? "#ffab40" : "#00e676";
    const svg = d3.select(svgEl);
    // Area fill under sparkline
    const area = d3.area().x((_, i) => x(i)).y0(h).y1(v => y(v)).curve(d3.curveCatmullRom.alpha(0.5));
    svg.append("path").datum(vals).attr("fill", col).attr("opacity", "0.08").attr("d", area);
    svg.append("path").datum(vals).attr("fill", "none").attr("stroke", col).attr("stroke-width", "1.5").attr("d", line);
  }

  function _fmt(v, key) {
    if (v === null || v === undefined) return "—";
    const UNITS = {
      systolic_bp: "mmHg", diastolic_bp: "mmHg", heart_rate: "bpm",
      fasting_glucose: "mg/dL", temperature: "°C", creatinine: "mg/dL",
      gfr: "mL/min", wbc: "×10³/μL", crp: "mg/L", hba1c: "%",
    };
    const unit = UNITS[key] || "";
    return `${parseFloat(v).toFixed(1)}${unit ? " " + unit : ""}`;
  }

  return { render };
})();
