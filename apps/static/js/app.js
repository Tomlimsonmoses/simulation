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
  };
  let _drugsData = [], _diseasesData = [];
  let _debounceTimer = null;
  const DEBOUNCE_MS = 320;

  // ───────────────────────────────────────── boot
  async function boot() {
    _wireSliders();
    CellView.init();
    PKChart.init();
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
    }
    document.getElementById("drug1").addEventListener("change", e => {
      _state.drug1 = e.target.value;
      const dd = _drugsData.find(d => d.name === e.target.value);
      if (dd) { _state.dose1 = dd.default_dose; document.getElementById("dose1").value = dd.default_dose; document.getElementById("lbl-dose1").textContent = dd.default_dose; }
      _updateDrugInfo("drug1", e.target.value);
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

    // Duration buttons
    document.querySelectorAll(".dur-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".dur-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        _state.duration = parseInt(btn.dataset.h);
        _triggerLiveOrNone();
      });
    });

    // Run button
    document.getElementById("run-btn")?.addEventListener("click", () => _runSim());
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

    // PK Chart
    PKChart.render({
      time_points:       data.time_points,
      concentrations:    data.concentrations,
      therapeutic_windows: data.pk_metrics ? Object.fromEntries(
        Object.entries(data.pk_metrics).map(([d,m]) => [d, m.therapeutic_window])
      ) : {},
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
