/* ================================================================
   app.js — Main controller: sidebar events → API → dispatch to vizs
   ================================================================ */

(async () => {
  // ───────────────────────────────────────── state
  let _state = {
    drug1: null, drug2: null, dose1: null, dose2: null,
    disease: null, severity: 0.55,
    age: 65, weight: 70, sex: 1,
    kidney: 0.80, liver: 0.90, hr: 72,
    duration: 72, useDrug2: false,
    lastResult: null,
    ltMode: false,
    ltDuration: 30,
    popMode: false,
    popN: 50,
    popDuration: 72,
    cypPhenotype: "normal",
  };
  let _drugsData = [], _diseasesData = [];
  let _debounceTimer = null;
  const DEBOUNCE_MS = 320;

  // ───────────────────────────────────────── boot
  async function boot() {
    _wireSliders();
    CellView.init();
    PKChart.init();
    OrganViz.setScrubber(
      document.getElementById("organ-time-scrubber"),
      document.getElementById("scrubber-label")
    );
    await Promise.all([_fetchDrugs(), _fetchDiseases()]);
    _populateDiseases();
    _populateDrugs();
    _wireExtraEvents();
    // Auto-run with defaults
    await _runSim();
  }

  // ───────────────────────────────────────── data fetches
  async function _fetchDrugs() {
    const r = await fetch("/api/drugs");
    _drugsData = await r.json();
  }
  async function _fetchDiseases() {
    const r = await fetch("/api/diseases");
    _diseasesData = await r.json();
  }

  // ───────────────────────────────────────── populate selects
  function _populateDiseases() {
    const sel = document.getElementById("disease");
    sel.innerHTML = "";
    _diseasesData.forEach(d => {
      const opt = document.createElement("option");
      opt.value = d.name; opt.textContent = d.name;
      sel.appendChild(opt);
    });
    _state.disease = _diseasesData[0]?.name || "Hypertension";
    sel.value = _state.disease;
    sel.addEventListener("change", () => {
      _state.disease = sel.value;
      const dd = _diseasesData.find(d => d.name === sel.value);
      if (dd) {
        document.getElementById("severity").value = Math.round(dd.default_severity * 100);
        document.getElementById("lbl-severity").textContent = Math.round(dd.default_severity*100) + "%";
        _state.severity = dd.default_severity;
      }
      _filterDrugsByDisease();
      _triggerLiveOrNone();
    });
  }

  function _populateDrugs() {
    _fillDrugSelect("drug1");
    _fillDrugSelect("drug2");
    const first = _drugsData[0]?.name;
    if (first) {
      _state.drug1 = first;
      document.getElementById("drug1").value = first;
      _state.dose1 = _drugsData[0]?.default_dose || 10;
      document.getElementById("dose1").value = _state.dose1;
      _updateDrugInfo("drug1", first);
      _fetchGenomicVariants(first);
    }
    document.getElementById("drug1").addEventListener("change", e => {
      _state.drug1 = e.target.value;
      const dd = _drugsData.find(d => d.name === e.target.value);
      if (dd) { _state.dose1 = dd.default_dose; document.getElementById("dose1").value = dd.default_dose; document.getElementById("lbl-dose1").textContent = dd.default_dose; }
      _updateDrugInfo("drug1", e.target.value);
      _fetchGenomicVariants(e.target.value);
      _triggerLiveOrNone();
    });
    document.getElementById("drug2").addEventListener("change", e => {
      _state.drug2 = e.target.value;
      const dd = _drugsData.find(d => d.name === e.target.value);
      if (dd) { _state.dose2 = dd.default_dose; document.getElementById("dose2").value = dd.default_dose; document.getElementById("lbl-dose2").textContent = dd.default_dose; }
      _updateDrugInfo("drug2", e.target.value);
      _triggerLiveOrNone();
    });
    document.getElementById("dose1").addEventListener("input", e => {
      _state.dose1 = parseFloat(e.target.value) || 10;
      document.getElementById("lbl-dose1").textContent = _state.dose1;
      _triggerLiveOrNone();
    });
    document.getElementById("dose2").addEventListener("input", e => {
      _state.dose2 = parseFloat(e.target.value) || 10;
      document.getElementById("lbl-dose2").textContent = _state.dose2;
      _triggerLiveOrNone();
    });

    // Filter drugs based on default disease
    _filterDrugsByDisease();
  }

  function _fillDrugSelect(id) {
    const sel = document.getElementById(id);
    sel.innerHTML = "";
    _drugsData.forEach(d => {
      const opt = document.createElement("option");
      opt.value = d.name; opt.textContent = `${d.name} (${d.drug_class})`;
      sel.appendChild(opt);
    });
  }

  function _filterDrugsByDisease() {
    const dis = _diseasesData.find(d => d.name === _state.disease);
    if (!dis || !dis.indicated_drugs || dis.indicated_drugs.length === 0) return;
    // Sort drug selects: indicated drugs first
    ["drug1","drug2"].forEach(id => {
      const sel = document.getElementById(id);
      const indicated = dis.indicated_drugs;
      const allOpts = Array.from(sel.options);
      allOpts.sort((a,b) => {
        const ai = indicated.indexOf(a.value), bi = indicated.indexOf(b.value);
        if (ai >= 0 && bi < 0) return -1;
        if (bi >= 0 && ai < 0) return 1;
        return a.text.localeCompare(b.text);
      });
      sel.innerHTML = "";
      allOpts.forEach(o => sel.appendChild(o));
    });
    // Auto-select first indicated drug
    if (dis.indicated_drugs.length > 0) {
      const sel = document.getElementById("drug1");
      sel.value = dis.indicated_drugs[0];
      _state.drug1 = dis.indicated_drugs[0];
      const dd = _drugsData.find(d => d.name === _state.drug1);
      if (dd) { _state.dose1 = dd.default_dose; document.getElementById("dose1").value = dd.default_dose; document.getElementById("lbl-dose1").textContent = dd.default_dose; }
      _updateDrugInfo("drug1", _state.drug1);
    }
  }

  function _updateDrugInfo(target, drugName) {
    const dd = _drugsData.find(d => d.name === drugName);
    const el = document.getElementById(`${target}-info`);
    if (!dd || !el) return;
    el.innerHTML = `<strong>${dd.drug_class}</strong> · ${dd.administration_route} · ${dd.dosing_frequency}<br/>
      t½: ${dd.half_life}h · F: ${Math.round(dd.bioavailability*100)}% · PB: ${Math.round(dd.protein_binding*100)}%<br/>
      TW: ${dd.therapeutic_window[0]}–${dd.therapeutic_window[1]} mg/L`;
  }

  // ─── Genomic CYP variant fetch & CYP phenotype hint ──────────────────────
  function _fetchGenomicVariants(drugName) {
    if (!drugName) return;
    fetch(`/api/genomics/${encodeURIComponent(drugName)}`)
      .then(r => r.json())
      .then(data => {
        const hintEl = document.getElementById("cyp-genomic-hint");
        if (!hintEl) return;
        if (!data.variants || data.variants.length === 0) {
          hintEl.innerHTML = "";
          hintEl.style.display = "none";
          return;
        }
        // Build concise hint
        const gene = data.cyp_gene || "UNKNOWN";
        const topV = data.variants.slice(0, 3);
        const sigCount = topV.filter(v => v.significance === "Significant").length;
        const effects = [...new Set(topV.map(v => v.effect).filter(Boolean))];
        let hint = `<strong>${gene}</strong> substrate · ${data.count} variant${data.count !== 1 ? "s" : ""}`;
        if (effects.length) hint += `<br><span style="color:#ffab40;font-size:9px">${effects.slice(0,2).join(" · ")}</span>`;
        if (sigCount > 0)   hint += `<span style="color:#ef5350;font-size:9px"> · ${sigCount} significant</span>`;
        hintEl.innerHTML = hint;
        hintEl.style.display = "";
        // Auto-suggest CYP phenotype if not already customised
        if (_state.cypPhenotype === "normal") {
          const hasPoorEffect = topV.some(v => v.effect && v.effect.includes("decreased_clearance"));
          const hasUltraEffect = topV.some(v => v.effect && v.effect.includes("increased_clearance"));
          const sel = document.getElementById("cyp-phenotype");
          if (hasPoorEffect && sel) {
            // Suggest but don't auto-set — just highlight
            hint += `<br><span style="color:#ef9a9a;font-size:9px">↓ CYP: consider Poor Metabolizer</span>`;
            hintEl.innerHTML = hint;
          } else if (hasUltraEffect && sel) {
            hint += `<br><span style="color:#a5d6a7;font-size:9px">↑ CYP: consider Ultra-Rapid Metabolizer</span>`;
            hintEl.innerHTML = hint;
          }
        }
      })
      .catch(() => {});
  }

  // ───────────────────────────────────────── slider wiring
  function _wireSliders() {
    const bindings = [
      { id:"age",      label:"lbl-age",      key:"age",     fmt: v => v,                  parse: v=>parseInt(v) },
      { id:"weight",   label:"lbl-weight",   key:"weight",  fmt: v => v + " kg",          parse: v=>parseInt(v) },
      { id:"kidney",   label:"lbl-kidney",   key:"kidney",  fmt: v => v + "%",            parse: v=>parseInt(v)/100 },
      { id:"liver",    label:"lbl-liver",    key:"liver",   fmt: v => v + "%",            parse: v=>parseInt(v)/100 },
      { id:"heart-rate",label:"lbl-hr",      key:"hr",      fmt: v => v + " bpm",         parse: v=>parseInt(v) },
      { id:"severity", label:"lbl-severity", key:"severity",fmt: v => v + "%",            parse: v=>parseInt(v)/100 },
    ];
    bindings.forEach(b => {
      const el = document.getElementById(b.id);
      const lbl = document.getElementById(b.label);
      if (!el) return;
      el.addEventListener("input", () => {
        lbl && (lbl.textContent = b.fmt(el.value));
        _state[b.key] = b.parse(el.value);
        _triggerLiveOrNone();
      });
    });

    // Sex toggle
    document.getElementById("sex-m")?.addEventListener("click", () => {
      _state.sex = 1;
      document.getElementById("sex-m").classList.add("active");
      document.getElementById("sex-f").classList.remove("active");
      _triggerLiveOrNone();
    });
    document.getElementById("sex-f")?.addEventListener("click", () => {
      _state.sex = 0;
      document.getElementById("sex-f").classList.add("active");
      document.getElementById("sex-m").classList.remove("active");
      _triggerLiveOrNone();
    });
  }

  function _wireExtraEvents() {
    // Drug2 toggle
    document.getElementById("drug2-toggle")?.addEventListener("change", e => {
      _state.useDrug2 = e.target.checked;
      const panel = document.getElementById("drug2-panel");
      panel.classList.toggle("open", _state.useDrug2);
      if (_state.useDrug2) {
        const sel = document.getElementById("drug2");
        _state.drug2 = sel.value;
        _updateDrugInfo("drug2", _state.drug2);
      } else {
        _state.drug2 = null;
      }
      _triggerLiveOrNone();
    });

    // CYP phenotype
    document.getElementById("cyp-phenotype")?.addEventListener("change", e => {
      _state.cypPhenotype = e.target.value;
      _triggerLiveOrNone();
    });

    // Duration buttons
    document.querySelectorAll(".dur-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".dur-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        _state.duration = parseInt(btn.dataset.h);
        _triggerLiveOrNone();
      });
    });

    // Run button — dispatches to correct sim based on mode
    document.getElementById("run-btn")?.addEventListener("click", () => {
      if (_state.popMode) _runPopSim();
      else if (_state.ltMode) _runLongTermSim();
      else _runSim();
    });

    // Mode switcher
    document.querySelectorAll(".mode-btn").forEach(btn => {
      btn.addEventListener("click", () => _switchMode(btn.dataset.mode));
    });

    // LT duration buttons
    document.querySelectorAll(".lt-dur-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".lt-dur-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        _state.ltDuration = parseInt(btn.dataset.d);
      });
    });

    // Population cohort size buttons
    document.querySelectorAll(".pop-n-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".pop-n-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        _state.popN = parseInt(btn.dataset.n);
      });
    });

    // Population duration buttons
    document.querySelectorAll(".pop-d-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".pop-d-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        _state.popDuration = parseInt(btn.dataset.h);
      });
    });
  }

  // ───────────────────────────────────────── live mode debounce
  function _triggerLiveOrNone() {
    if (!document.getElementById("live-mode")?.checked) return;
    clearTimeout(_debounceTimer);
    _debounceTimer = setTimeout(() => _runSim(), DEBOUNCE_MS);
  }

  // ───────────────────────────────────────── run simulation
  async function _runSim() {
    _setStatus("running","Running…");
    _showLoading(true);

    const payload = {
      drug:   _state.drug1   || "Lisinopril",
      dose:   _state.dose1   || 10,
      drug2:  _state.useDrug2 ? (_state.drug2 || null) : null,
      dose2:  _state.useDrug2 ? (_state.dose2 || null) : null,
      disease: _state.disease || "Hypertension",
      severity: _state.severity || 0.55,
      age:    _state.age     || 65,
      weight: _state.weight  || 70,
      sex:    _state.sex     || 1,
      kidney_function: _state.kidney || 0.80,
      liver_function:  _state.liver  || 0.90,
      heart_rate:      _state.hr     || 72,
      duration_hours:  _state.duration || 72,
      cyp_phenotype:   _state.cypPhenotype || "normal",
    };

    try {
      const resp = await fetch("/api/simulate", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      // Attach doses for cell API
      data._doses = { [payload.drug]: payload.dose };
      if (payload.drug2) data._doses[payload.drug2] = payload.dose2;
      _state.lastResult = data;
      _dispatchResults(data, payload);
      _setStatus("done","Simulation complete");
    } catch(e) {
      console.error("Simulation error:", e);
      _setStatus("error","Error — check console");
    } finally {
      _showLoading(false);
    }
  }

  // ───────────────────────────────────────── dispatch to vizs
  function _dispatchResults(data, payload) {
    // Subtitle
    document.getElementById("sim-subtitle").textContent =
      `${data.drugs?.join(" + ")} · ${data.disease_name} · ${_state.duration}h simulation`;

    // Organ body
    OrganViz.update(data.organ_health_final, data.concentrations);
    OrganViz.setHeartRate(data.heart_rate || 72);
    // Animate organ health series if available
    if (data.organ_health_series && data.organ_health_series.length > 1) {
      OrganViz.animateSeries(data.organ_health_series, data.series_time_points);
    }

    // PK Chart (with receptor occupancy overlay)
    PKChart.render({
      time_points:         data.time_points,
      concentrations:      data.concentrations,
      therapeutic_windows: data.pk_metrics ? Object.fromEntries(
        Object.entries(data.pk_metrics).map(([d,m]) => [d, m.therapeutic_window])
      ) : {},
      receptor_occupancy:  data.receptor_occupancy,
      series_time_points:  data.series_time_points,
    });

    // Radar
    RadarChart.render(data.organ_health_final, RadarChart.getBaseline());

    // Timeline
    Timeline.render(data.side_effects, _state.duration);

    // Alerts + Biomarkers
    Biomarkers.render(data);

    // KPIs (primary drug)
    const firstDrug = data.drugs?.[0];
    if (firstDrug && data.pk_metrics?.[firstDrug]) {
      const m = data.pk_metrics[firstDrug];
      document.querySelector("#kpi-cmax .kpi-val").textContent  = m.cmax?.toFixed(2)     ?? "—";
      document.querySelector("#kpi-tmax .kpi-val").textContent  = m.tmax?.toFixed(1)     ?? "—";
      document.querySelector("#kpi-auc .kpi-val").textContent   = m.auc?.toFixed(0)      ?? "—";
      document.querySelector("#kpi-halflife .kpi-val").textContent = m.half_life?.toFixed(1) ?? "—";
    }

    // Health ring (main)
    _drawMainHealthRing(data.overall_health);
    document.getElementById("health-pct").textContent = Math.round((data.overall_health||0.85)*100) + "%";

    // Therapeutic status
    const tsEl  = document.getElementById("therapeutic-status");
    const tsCard = document.getElementById("therapeutic-status-card");
    const ts = data.therapeutic_status || "therapeutic";
    tsEl.textContent = ts.replace("-"," ").toUpperCase();
    tsEl.className = `status-value ${ts}`;

    // Update open cell drawer if applicable
    if (data.organ_health_final) {
      const activeOrgan = document.getElementById("cell-organ-name")?.textContent?.toLowerCase().replace(/ /g,"_");
      if (activeOrgan) {
        const h = data.organ_health_final[activeOrgan] || 0.85;
        CellView.update(activeOrgan, h, data, data.drugs);
      }
    }

    // Drug Effect Dynamics row
    _renderPDChart(data);

    // Engine / ML / GNN status badges
    _renderEngineBadges(data);
  }

  function _renderEngineBadges(data) {
    const row = document.getElementById("engine-badge-row");
    if (!row) return;
    row.innerHTML = "";
    const _badge = (label, active, activeColor) => {
      const span = document.createElement("span");
      span.style.cssText = `padding:2px 7px;border-radius:10px;font-size:10px;font-weight:500;` +
        `background:${active ? activeColor + "22" : "rgba(255,255,255,0.05)"};` +
        `color:${active ? activeColor : "#555c6e"};border:1px solid ${active ? activeColor + "55" : "rgba(255,255,255,0.08)"}`;
      span.textContent = label;
      row.appendChild(span);
    };
    _badge(data.engine === "mechanistic" ? "PBPK" : "Analytical", data.engine === "mechanistic", "#00bcd4");
    _badge("ML",  data.ml_active  === true, "#ffab40");
    _badge("GNN", data.gnn_active === true, "#e040fb");
    if (data.cyp_phenotype && data.cyp_phenotype !== "normal") {
      const labels = { poor: "PM", intermediate: "IM", ultra_rapid: "URM" };
      _badge("CYP:" + (labels[data.cyp_phenotype] || data.cyp_phenotype), true, "#ff7043");
    }
  }

  // ───────────────────────────────────────── PD chart (receptor occ + PD response)
  function _renderPDChart(data) {
    const container = document.getElementById("pd-chart-container");
    const row = document.getElementById("pd-row");
    if (!container) return;

    const hasOcc = data.receptor_occupancy && data.receptor_occupancy.length > 1;
    const hasPD  = data.pd_response && data.pd_response.length > 1;
    if (!hasOcc && !hasPD) { if (row) row.style.display = "none"; return; }
    if (row) row.style.display = "";

    const times = data.series_time_points || data.time_points;
    const cw = container.clientWidth || 700;
    const ch = 200;
    const margin = { top: 10, right: 50, bottom: 36, left: 48 };
    const w = cw - margin.left - margin.right;
    const h = ch - margin.top  - margin.bottom;

    d3.select(container).select("svg").remove();

    const svg = d3.select(container).append("svg").attr("width",cw).attr("height",ch);
    const g   = svg.append("g").attr("transform",`translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([0, times[times.length-1]]).range([0,w]);
    const yScale = d3.scaleLinear().domain([0,1]).range([h,0]);

    // Grid
    g.append("g").call(d3.axisLeft(yScale).tickSize(-w).ticks(4).tickFormat(""))
      .selectAll("line").attr("stroke","rgba(255,255,255,0.05)");
    g.select(".domain").remove();

    // Receptor occupancy line
    if (hasOcc) {
      const lineData = times.map((t,j) => ({ t, v: data.receptor_occupancy[j] ?? 0 }));
      const lineFn = d3.line().x(d=>xScale(d.t)).y(d=>yScale(d.v)).curve(d3.curveCatmullRom.alpha(0.5));
      g.append("path").datum(lineData).attr("fill","none")
        .attr("stroke","#ff7043").attr("stroke-width","2")
        .attr("d", lineFn);
    }

    // PD response line
    if (hasPD) {
      const lineData = times.map((t,j) => ({ t, v: data.pd_response[j] ?? 0 }));
      const lineFn = d3.line().x(d=>xScale(d.t)).y(d=>yScale(d.v)).curve(d3.curveCatmullRom.alpha(0.5));
      g.append("path").datum(lineData).attr("fill","none")
        .attr("stroke","#00e676").attr("stroke-width","2").attr("stroke-dasharray","6,3")
        .attr("d", lineFn);
    }

    // Axes
    g.append("g").attr("transform",`translate(0,${h})`)
      .call(d3.axisBottom(xScale).ticks(8).tickFormat(d=>`${d}h`))
      .selectAll("text").attr("fill","#7b8fb5").attr("font-size","10");
    g.append("g")
      .call(d3.axisLeft(yScale).ticks(4).tickFormat(d=>`${Math.round(d*100)}%`))
      .selectAll("text").attr("fill","#7b8fb5").attr("font-size","10");

    // Axis label
    svg.append("text").attr("x",margin.left + w/2).attr("y",ch-2)
      .attr("text-anchor","middle").attr("fill","#7b8fb5").attr("font-size","10")
      .attr("font-family","Inter,sans-serif").text("Time (hours)");

    // Legend
    const legendY = margin.top + 4;
    const legendX = margin.left + w - 180;
    if (hasOcc) {
      svg.append("line").attr("x1",legendX).attr("x2",legendX+18).attr("y1",legendY+6).attr("y2",legendY+6)
        .attr("stroke","#ff7043").attr("stroke-width",2);
      svg.append("text").attr("x",legendX+22).attr("y",legendY+10).attr("fill","#ff7043")
        .attr("font-size","10").attr("font-family","Inter,sans-serif").text("Receptor occupancy");
    }
    if (hasPD) {
      svg.append("line").attr("x1",legendX).attr("x2",legendX+18).attr("y1",legendY+22).attr("y2",legendY+22)
        .attr("stroke","#00e676").attr("stroke-width",2).attr("stroke-dasharray","6,3");
      svg.append("text").attr("x",legendX+22).attr("y",legendY+26).attr("fill","#00e676")
        .attr("font-size","10").attr("font-family","Inter,sans-serif").text("PD response");
    }
  }

  // ─────────────────────────────────────────── mode switching
  function _switchMode(mode) {
    _state.ltMode  = (mode === "long");
    _state.popMode = (mode === "pop");
    const isShort = (mode === "short");
    // toggle mode buttons
    document.querySelectorAll(".mode-btn").forEach(b => b.classList.toggle("active", b.dataset.mode === mode));
    // sidebar duration sections
    const stDur = document.getElementById("st-duration-section");
    const ltDur = document.getElementById("lt-duration-section");
    if (stDur) stDur.style.display = isShort ? "" : "none";
    if (ltDur) ltDur.style.display = _state.ltMode ? "" : "none";
    // short-term rows
    [".row1",".row2",".row3"].forEach(cls => {
      const el = document.querySelector(cls);
      if (el) el.style.display = isShort ? "" : "none";
    });
    const pdRow  = document.getElementById("pd-row");
    if (pdRow) pdRow.style.display = "none";
    // panels
    const ltPanel  = document.getElementById("lt-panel");
    const popPanel = document.getElementById("pop-panel");
    if (ltPanel)  ltPanel.style.display  = _state.ltMode  ? "" : "none";
    if (popPanel) popPanel.style.display = _state.popMode ? "" : "none";
    // run button text
    const runBtn = document.getElementById("run-btn");
    if (runBtn) {
      const labels = { short: " Run Simulation", long: " Run Long-Term", pop: " Run Population" };
      runBtn.childNodes[runBtn.childNodes.length - 1].textContent = labels[mode] || " Run Simulation";
    }
  }

  // ─────────────────────────────────────────── long-term simulation
  async function _runLongTermSim() {
    _setStatus("running", "Running long-term…");
    _showLoading(true);
    const payload = {
      drug:            _state.drug1   || "Lisinopril",
      dose:            _state.dose1   || 10,
      drug2:           _state.useDrug2 ? (_state.drug2 || null) : null,
      dose2:           _state.useDrug2 ? (_state.dose2 || null) : null,
      disease:         _state.disease || "Hypertension",
      severity:        _state.severity || 0.55,
      age:             _state.age     || 65,
      weight:          _state.weight  || 70,
      sex:             _state.sex     || 1,
      kidney_function: _state.kidney  || 0.80,
      liver_function:  _state.liver   || 0.90,
      duration_days:   _state.ltDuration || 30,
      cyp_phenotype:   _state.cypPhenotype || "normal",
    };
    try {
      const resp = await fetch("/api/simulate/longterm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      LongTerm.render(data);
      document.getElementById("sim-subtitle").textContent =
        `${payload.drug} · ${payload.disease} · ${_state.ltDuration}-day long-term simulation`;
      _setStatus("done", "Long-term complete");
    } catch (e) {
      console.error("Long-term simulation error:", e);
      _setStatus("error", "Error — check console");
    } finally {
      _showLoading(false);
    }
  }

  // ─────────────────────────────────────────── population simulation
  async function _runPopSim() {
    _setStatus("running", "Running population cohort…");
    _showLoading(true);
    const payload = {
      drug:            _state.drug1   || "Lisinopril",
      dose:            _state.dose1   || 10,
      disease:         _state.disease || "Hypertension",
      severity:        _state.severity || 0.55,
      age:             _state.age     || 65,
      weight:          _state.weight  || 70,
      sex:             _state.sex     || 1,
      kidney_function: _state.kidney  || 0.80,
      liver_function:  _state.liver   || 0.90,
      n_patients:      _state.popN    || 50,
      duration_hours:  _state.popDuration || 72,
    };
    try {
      const resp = await fetch("/api/population", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
      const data = await resp.json();
      _renderPopResults(data, payload);
      document.getElementById("sim-subtitle").textContent =
        `${payload.drug} · ${payload.disease} · ${payload.n_patients}-patient cohort`;
      _setStatus("done", "Population complete");
    } catch (e) {
      console.error("Population simulation error:", e);
      _setStatus("error", e.message || "Error — check console");
    } finally {
      _showLoading(false);
    }
  }

  function _renderPopResults(data, payload) {
    // KPIs
    const setKpi = (id, val, decimals = 1) => {
      const el = document.querySelector(`#${id} .lt-kpi-val`);
      if (el) el.textContent = val != null ? (typeof val === "number" ? val.toFixed(decimals) : val) : "—";
    };
    setKpi("pop-kpi-n",      data.n_succeeded, 0);
    setKpi("pop-kpi-within", (data.frac_within_window * 100), 1);
    document.querySelector("#pop-kpi-within .lt-kpi-val").textContent =
      data.frac_within_window != null ? (data.frac_within_window * 100).toFixed(1) + "%" : "—";
    setKpi("pop-kpi-cmax",   data.cmax_p50);
    setKpi("pop-kpi-auc",    data.auc_p50, 0);

    // Population envelope chart
    const container = document.getElementById("pop-pk-chart");
    if (!container || !data.time_points) return;
    d3.select(container).select("svg").remove();

    const TW = data.therapeutic_window || [0, 100];
    const margin = { top: 14, right: 24, bottom: 44, left: 56 };
    const cw = container.clientWidth || 700;
    const ch = container.clientHeight || 300;
    const w  = cw - margin.left - margin.right;
    const h  = ch - margin.top  - margin.bottom;

    const svg = d3.select(container).append("svg").attr("width", cw).attr("height", ch);
    const g   = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const times = data.time_points;
    const allVals = [...(data.p05||[]), ...(data.p95||[])].filter(isFinite);
    const yMax = allVals.length ? Math.max(...allVals) * 1.15 : 10;

    const xSc = d3.scaleLinear().domain([0, times[times.length - 1]]).range([0, w]);
    const ySc = d3.scaleLinear().domain([0, yMax]).range([h, 0]).nice();

    // Grid
    g.append("g").call(d3.axisLeft(ySc).tickSize(-w).ticks(5).tickFormat(""))
      .selectAll("line").attr("stroke", "rgba(255,255,255,0.05)");
    g.selectAll(".domain").attr("stroke", "rgba(255,255,255,0.1)");

    // Therapeutic window bands
    g.append("rect")
      .attr("x", 0).attr("y", ySc(TW[1]))
      .attr("width", w).attr("height", Math.max(0, ySc(TW[0]) - ySc(TW[1])))
      .attr("fill", "rgba(0,230,118,0.08)");
    [TW[0], TW[1]].forEach(v => {
      g.append("line")
        .attr("x1", 0).attr("x2", w)
        .attr("y1", ySc(v)).attr("y2", ySc(v))
        .attr("stroke", "rgba(0,230,118,0.35)").attr("stroke-dasharray", "4,3");
    });

    // Shaded bands
    const area = (lo, hi) => d3.area().x((_, i) => xSc(times[i]))
      .y0(d => ySc(d)).y1((_, i) => ySc(hi[i]))
      .defined((d, i) => isFinite(d) && isFinite(hi[i]));

    if (data.p05 && data.p95)
      g.append("path").datum(data.p05)
        .attr("fill", "rgba(0,188,212,0.12)").attr("d", area(data.p05, data.p95));
    if (data.p25 && data.p75)
      g.append("path").datum(data.p25)
        .attr("fill", "rgba(0,188,212,0.20)").attr("d", area(data.p25, data.p75));

    // Median line
    if (data.p50) {
      const lineFn = d3.line().x((_, i) => xSc(times[i])).y(d => ySc(d))
        .defined(d => isFinite(d)).curve(d3.curveCatmullRom.alpha(0.5));
      g.append("path").datum(data.p50)
        .attr("fill", "none").attr("stroke", "#00bcd4")
        .attr("stroke-width", 2).attr("d", lineFn);
    }

    // Axes
    g.append("g").attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(xSc).ticks(8).tickFormat(d => `${d}h`))
      .selectAll("text").attr("fill", "#7b8fb5").attr("font-size", "10");
    g.append("g")
      .call(d3.axisLeft(ySc).ticks(5))
      .selectAll("text").attr("fill", "#7b8fb5").attr("font-size", "10");

    // Y-axis label
    svg.append("text").attr("transform", "rotate(-90)")
      .attr("x", -(margin.top + h / 2)).attr("y", 14)
      .attr("text-anchor", "middle").attr("fill", "#7b8fb5").attr("font-size", "10")
      .attr("font-family", "Inter,sans-serif").text("Conc. (mg/L)");

    // Legend
    const lx = margin.left + 8, ly = margin.top + 8;
    [["rgba(0,188,212,0.12)", "p5–p95"], ["rgba(0,188,212,0.20)", "p25–p75"]].forEach(([c, lbl], i) => {
      svg.append("rect").attr("x", lx + i * 70).attr("y", ly).attr("width", 12).attr("height", 8).attr("fill", c);
      svg.append("text").attr("x", lx + i * 70 + 16).attr("y", ly + 7)
        .attr("fill", "#7b8fb5").attr("font-size", "9").attr("font-family", "Inter,sans-serif").text(lbl);
    });
    svg.append("line").attr("x1", lx + 144).attr("x2", lx + 158).attr("y1", ly + 4).attr("y2", ly + 4)
      .attr("stroke", "#00bcd4").attr("stroke-width", 2);
    svg.append("text").attr("x", lx + 162).attr("y", ly + 7)
      .attr("fill", "#00bcd4").attr("font-size", "9").attr("font-family", "Inter,sans-serif").text("Median");
  }

  function _drawMainHealthRing(health) {
    const svg  = d3.select("#health-ring-svg");
    svg.selectAll("*").remove();
    const W = 160, H = 160, cx = W/2, cy = H/2, outerR = 68, sw = 12;
    const arc  = d3.arc().innerRadius(outerR-sw).outerRadius(outerR).startAngle(0);
    const col  = d3.scaleLinear().domain([0,0.5,1]).range(["#ef5350","#ffab40","#00e676"])(health);
    svg.append("path").datum({endAngle:2*Math.PI}).attr("d",arc).attr("fill","rgba(255,255,255,0.05)").attr("transform",`translate(${cx},${cy})`);
    svg.append("path").datum({endAngle:0}).attr("d",arc).attr("fill",col).attr("transform",`translate(${cx},${cy})`)
      .transition().duration(900)
      .attrTween("d",function(d){ const i=d3.interpolate(d.endAngle,2*Math.PI*health); return t=>{ d.endAngle=i(t); return arc(d); }; });
  }

  // ───────────────────────────────────────── UI helpers
  function _setStatus(cls, text) {
    const pill = document.getElementById("status-pill");
    const dot  = pill?.querySelector(".status-dot");
    if (!pill) return;
    pill.childNodes[pill.childNodes.length-1].textContent = " " + text;
    if (dot) dot.className = `status-dot ${cls}`;
  }

  function _showLoading(show) {
    const el = document.getElementById("loading-overlay");
    el?.classList.toggle("active", show);
  }

  // ───────────────────────────────────────── organ click → cell drawer
  function _onOrganClick(organId, organHealth) {
    const drugNames = _state.lastResult?.drugs || [_state.drug1];
    CellView.open(organId, organHealth, _state.lastResult, drugNames);
  }

  // ───────────────────────────────────────── init organ viz
  OrganViz.init(_onOrganClick);

  // ───────────────────────────────────────── start
  boot();
})();
