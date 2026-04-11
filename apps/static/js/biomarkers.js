/* ================================================================
   biomarkers.js — alert cards + biomarker sparklines
   ================================================================ */

const Biomarkers = (() => {

  function render(data) {
    const container = document.getElementById("alerts-container");
    if (!container || !data) return;
    container.innerHTML = "";

    // ── Interaction warnings + side-effect alerts
    const warnings = data.warnings || [];
    const recommendations = data.recommendations || [];

    if (warnings.length > 0) {
      container.insertAdjacentHTML("beforeend",`<div class="alert-section-title">⚠ Alerts</div>`);
      warnings.forEach(w => {
        const isInteraction = w.startsWith("INTERACTION");
        const icon = isInteraction ? "🔀" : "🚨";
        const cls  = isInteraction ? "interaction" : "high";
        const text = w.replace(/^(INTERACTION|ALERT):\s*/,"");
        container.insertAdjacentHTML("beforeend",`
          <div class="alert-item ${cls}">
            <span class="ai-icon">${icon}</span>
            <span class="ai-text">${text}</span>
          </div>`);
      });
    }

    if (recommendations.length > 0) {
      container.insertAdjacentHTML("beforeend",`<div class="alert-section-title">💊 Clinical Notes</div>`);
      recommendations.slice(0,5).forEach(r => {
        container.insertAdjacentHTML("beforeend",`
          <div class="alert-item medium">
            <span class="ai-icon">📋</span>
            <span class="ai-text">${r}</span>
          </div>`);
      });
    }

    if (warnings.length === 0 && recommendations.length === 0) {
      container.insertAdjacentHTML("beforeend",`
        <div class="alert-item low">
          <span class="ai-icon">✅</span>
          <span class="ai-text"><strong>No interactions or alerts.</strong> Simulation looks clinically safe.</span>
        </div>`);
    }

    // ── Biomarker sparklines
    const bm = data.biomarkers || {};
    const bmKeys = Object.keys(bm).filter(k => bm[k] && bm[k].length > 1).slice(0, 6);
    if (bmKeys.length === 0) return;

    container.insertAdjacentHTML("beforeend",`<div class="alert-section-title">📈 Biomarkers</div><div class="sparkline-grid" id="sparkline-grid"></div>`);
    const grid = document.getElementById("sparkline-grid");

    bmKeys.forEach(key => {
      const vals = bm[key];
      const last = vals[vals.length - 1];
      const label = key.replace(/_/g," ").replace(/\b\w/g,c=>c.toUpperCase());
      const div = document.createElement("div");
      div.className = "sparkline-card";
      div.innerHTML = `<div class="sparkline-label">${label}</div><div class="sparkline-val">${_fmt(last, key)}</div><svg class="spark" width="100%" height="32"></svg>`;
      grid.appendChild(div);
      _drawSparkline(div.querySelector(".spark"), vals);
    });
  }

  function _drawSparkline(svgEl, vals) {
    if (!svgEl || vals.length < 2) return;
    const w = svgEl.clientWidth || 120, h = 32;
    const x = d3.scaleLinear().domain([0, vals.length-1]).range([0, w]);
    const y = d3.scaleLinear().domain([d3.min(vals)*0.95, d3.max(vals)*1.05]).range([h-2, 2]);
    const line = d3.line().x((_,i)=>x(i)).y(v=>y(v)).curve(d3.curveCatmullRom.alpha(0.5));
    // trend color
    const trend = vals[vals.length-1] - vals[0];
    const col = trend > 0 ? "#ffab40" : "#00e676";
    const svg = d3.select(svgEl);
    svg.append("path").datum(vals).attr("fill","none").attr("stroke",col).attr("stroke-width","1.5").attr("d",line);
  }

  function _fmt(v, key) {
    if (v === null || v === undefined) return "—";
    const UNITS = {
      systolic_bp:"mmHg",diastolic_bp:"mmHg",heart_rate:"bpm",
      fasting_glucose:"mg/dL",temperature:"°C",creatinine:"mg/dL",
      gfr:"mL/min",wbc:"×10³/μL",crp:"mg/L",hba1c:"%"
    };
    const unit = UNITS[key] || "";
    return `${parseFloat(v).toFixed(1)}${unit ? " "+unit : ""}`;
  }

  return { render };
})();
