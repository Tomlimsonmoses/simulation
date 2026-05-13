/* ================================================================
   app.js — Main controller: sidebar events → API → dispatch to vizs
   ================================================================ */

(async () => {
  // ───────────────────────────────────────── state
  let _state = {
    drug1: null, drug2: null, drug3: null,
    dose1: null, dose2: null, dose3: null,
    disease: null, severity: 0.55,
    age: 65, weight: 70, sex: 1,
    kidney: 0.80, liver: 0.90, hr: 72,
    duration: 72, useDrug2: false, useDrug3: false,
    lastResult: null,
    ltMode: false, ltDuration: 30,
    popMode: false, popN: 50, popDuration: 72,
    cypPhenotype: "normal",
    route1: "oral",
    timeScale: 1,
    hlaBAllele: "none",
    dpydDeficient: false,
    g6pdDeficient: false,
    tpmtLow: false,
    showAllDrugs: false,
  };
  let _drugsData = [], _diseasesData = [];
  let _debounceTimer = null;
  const DEBOUNCE_MS = 320;

  // ───────────────────────────────────────── boot
  async function boot() {
    _wireSliders();
    CellView.init();
    PKChart.init();
    HumanBody3D.init('human-3d-container', _onOrganClick);
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
    _fillDrugSelect("drug3");
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
    const indicated = (dis && dis.indicated_drugs) ? dis.indicated_drugs : [];
    const drugsToShow = (_state.showAllDrugs || indicated.length === 0)
      ? _drugsData
      : _drugsData.filter(d => indicated.includes(d.name));

    // Repopulate all three drug selects
    ["drug1", "drug2", "drug3"].forEach(id => {
      const sel = document.getElementById(id);
      if (!sel) return;
      const curVal = sel.value;
      sel.innerHTML = "";
      drugsToShow.forEach(d => {
        const opt = document.createElement("option");
        opt.value = d.name;
        opt.textContent = `${d.name} (${d.drug_class})`;
        sel.appendChild(opt);
      });
      if (drugsToShow.find(d => d.name === curVal)) sel.value = curVal;
    });

    _renderDrugChips(indicated);
    _renderDiseaseInfoStrip(dis, indicated);

    // Auto-select first indicated drug for drug1
    const pick = indicated.length > 0 ? indicated[0] : drugsToShow[0]?.name;
    if (pick) {
      const sel = document.getElementById("drug1");
      if (sel) sel.value = pick;
      _state.drug1 = pick;
      const dd = _drugsData.find(d => d.name === pick);
      if (dd) {
        _state.dose1 = dd.default_dose;
        document.getElementById("dose1").value = dd.default_dose;
        document.getElementById("lbl-dose1").textContent = dd.default_dose;
      }
      _updateDrugInfo("drug1", pick);
      _fetchGenomicVariants(pick);
    }
  }

  function _renderDrugChips(indicated) {
    const container = document.getElementById("indicated-drug-chips");
    if (!container) return;
    container.innerHTML = "";
    if (!indicated || indicated.length === 0) return;
    indicated.forEach(name => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "drug-chip" + (name === _state.drug1 ? " selected" : "");
      chip.textContent = name;
      chip.title = name;
      chip.addEventListener("click", () => {
        _state.drug1 = name;
        const sel = document.getElementById("drug1");
        if (sel) sel.value = name;
        const dd = _drugsData.find(d => d.name === name);
        if (dd) {
          _state.dose1 = dd.default_dose;
          document.getElementById("dose1").value = dd.default_dose;
          document.getElementById("lbl-dose1").textContent = dd.default_dose;
        }
        _updateDrugInfo("drug1", name);
        _fetchGenomicVariants(name);
        container.querySelectorAll(".drug-chip").forEach(c => c.classList.remove("selected"));
        chip.classList.add("selected");
        _triggerLiveOrNone();
      });
      container.appendChild(chip);
    });
  }

  function _renderDiseaseInfoStrip(dis, indicated) {
    const strip = document.getElementById("disease-info-strip");
    if (!strip) return;
    if (!dis) { strip.style.display = "none"; return; }
    const organTags = (dis.affected_organs || []).map(o =>
      `<span class="organ-tag">${o.replace(/_/g, " ")}</span>`
    ).join(" ");
    const drugNames = indicated.slice(0, 4).map(n => `<strong style="color:var(--accent)">${n}</strong>`).join(", ");
    strip.innerHTML = `
      <div style="font-size:9.5px;color:var(--text-muted);margin-bottom:3px">Affected: ${organTags || "—"}</div>
      ${indicated.length ? `<div style="font-size:9.5px;color:var(--text-muted)">First-line: ${drugNames}${indicated.length > 4 ? " +" + (indicated.length - 4) + " more" : ""}</div>` : ""}
    `;
    strip.style.display = "";
  }

  function _updateDrugInfo(target, drugName) {
    const dd = _drugsData.find(d => d.name === drugName);
    const el = document.getElementById(`${target}-info`);
    if (!dd || !el) return;
    el.innerHTML = `<strong>${dd.drug_class}</strong> · ${dd.administration_route} · ${dd.dosing_frequency}<br/>
      t½: ${dd.half_life}h · F: ${Math.round(dd.bioavailability*100)}% · PB: ${Math.round(dd.protein_binding*100)}%<br/>
      TW: ${dd.therapeutic_window[0]}–${dd.therapeutic_window[1]} mg/L`;
    if (target === "drug1") {
      const routeSel = document.getElementById("route1");
      if (routeSel) {
        const routeMap = { oral: "oral", iv: "iv", intravenous: "iv", subcutaneous: "subcutaneous",
          im: "im", intramuscular: "im", sublingual: "sublingual", transdermal: "transdermal",
          inhalation: "inhalation" };
        const normalized = routeMap[dd.administration_route?.toLowerCase()] || "oral";
        routeSel.value = normalized;
        _state.route1 = normalized;
        _updateRouteInfo(normalized, dd.bioavailability, dd.default_dose);
      }
      _calcRecommendedDose(drugName);
    }
  }

  function _calcRecommendedDose(drugName) {
    const dd = _drugsData.find(d => d.name === drugName);
    const badge = document.getElementById("rec-dose1");
    if (!dd || !badge) return null;
    // Allometric weight scaling (Anderson & Holford)
    const wtFactor = Math.pow(Math.max(1, _state.weight) / 70, 0.75);
    // Renal adjustment: applies if low protein binding OR short half-life
    const renalDep = dd.protein_binding < 0.5 || dd.half_life < 6;
    const renalFactor = renalDep ? Math.max(0.25, _state.kidney) : 1.0;
    // Hepatic adjustment: applies if low oral bioavailability (heavy first-pass)
    const hepaticDep = dd.bioavailability < 0.4;
    const liverFactor = hepaticDep ? Math.max(0.5, Math.sqrt(Math.max(0.01, _state.liver))) : 1.0;
    const rec = Math.round(dd.default_dose * wtFactor * renalFactor * liverFactor * 10) / 10;
    badge.textContent = `Rec: ${rec} mg`;
    badge.dataset.dose = rec;
    return rec;
  }

  function _updateRouteInfo(routeValue, bioavailability, baseDose) {
    const el = document.getElementById("route1-info");
    if (!el) return;
    const ba = Math.round((bioavailability || 0.7) * 100);
    let msg = "";
    if (routeValue === "iv") {
      const equiv = Math.round((baseDose || 10) * (bioavailability || 0.7) * 10) / 10;
      msg = `100% bioavailability — oral-to-IV equivalent ≈ ${equiv} mg`;
    } else if (routeValue === "sublingual") {
      msg = `Bypasses first-pass metabolism — faster onset than oral`;
    } else if (routeValue === "oral") {
      msg = `Oral bioavailability: ${ba}% — may vary with food intake`;
    } else if (routeValue === "transdermal") {
      msg = `Slow sustained absorption — steady-state reached over days`;
    } else if (routeValue === "inhalation") {
      msg = `Direct pulmonary absorption — fast onset, lower systemic exposure`;
    } else {
      msg = `Bioavailability: ${ba}%`;
    }
    el.textContent = msg;
    el.style.display = msg ? "" : "none";
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
        if (b.id === "age") HumanBody3D.updateAge(_state.age);
        if (["weight", "kidney", "liver"].includes(b.id)) _calcRecommendedDose(_state.drug1);
        _triggerLiveOrNone();
      });
    });

    // Sex toggle
    document.getElementById("sex-m")?.addEventListener("click", () => {
      _state.sex = 1;
      document.getElementById("sex-m").classList.add("active");
      document.getElementById("sex-f").classList.remove("active");
      HumanBody3D.updateGender('male');
      _triggerLiveOrNone();
    });
    document.getElementById("sex-f")?.addEventListener("click", () => {
      _state.sex = 0;
      document.getElementById("sex-f").classList.add("active");
      document.getElementById("sex-m").classList.remove("active");
      HumanBody3D.updateGender('female');
      _triggerLiveOrNone();
    });
  }

  function _wireExtraEvents() {
    // Drug2 toggle
    document.getElementById("drug2-toggle")?.addEventListener("change", e => {
      _state.useDrug2 = e.target.checked;
      const panel = document.getElementById("drug2-panel");
      panel.classList.toggle("open", _state.useDrug2);
      const addBtn = document.getElementById("add-drug3-btn");
      if (addBtn) addBtn.style.display = _state.useDrug2 ? "" : "none";
      if (_state.useDrug2) {
        const sel = document.getElementById("drug2");
        _state.drug2 = sel.value;
        _updateDrugInfo("drug2", _state.drug2);
      } else {
        _state.drug2 = null;
        _state.useDrug3 = false;
        _state.drug3 = null;
        const d3row = document.getElementById("combo-drug3-row");
        if (d3row) d3row.style.display = "none";
        if (addBtn) addBtn.style.display = "none";
      }
      _checkInteractions();
      _triggerLiveOrNone();
    });

    // Add drug3 button
    document.getElementById("add-drug3-btn")?.addEventListener("click", () => {
      _state.useDrug3 = true;
      const row = document.getElementById("combo-drug3-row");
      if (row) row.style.display = "";
      document.getElementById("add-drug3-btn").style.display = "none";
      const sel = document.getElementById("drug3");
      if (sel && sel.value) { _state.drug3 = sel.value; }
      _checkInteractions();
    });

    // Drug3 changes
    document.getElementById("drug3")?.addEventListener("change", e => {
      _state.drug3 = e.target.value;
      _checkInteractions();
      _triggerLiveOrNone();
    });
    document.getElementById("dose3")?.addEventListener("input", e => {
      _state.dose3 = parseFloat(e.target.value) || 10;
      _triggerLiveOrNone();
    });

    // Show all drugs link
    document.getElementById("show-all-drugs-link")?.addEventListener("click", e => {
      e.preventDefault();
      _state.showAllDrugs = !_state.showAllDrugs;
      e.target.textContent = _state.showAllDrugs ? "Show indicated" : "Show all";
      _filterDrugsByDisease();
    });

    // Route selector
    document.getElementById("route1")?.addEventListener("change", e => {
      _state.route1 = e.target.value;
      const dd = _drugsData.find(d => d.name === _state.drug1);
      _updateRouteInfo(e.target.value, dd?.bioavailability, dd?.default_dose);
    });

    // Recommended dose badge click
    document.getElementById("rec-dose1")?.addEventListener("click", () => {
      const badge = document.getElementById("rec-dose1");
      const dose = parseFloat(badge.dataset.dose);
      if (!isNaN(dose)) {
        document.getElementById("dose1").value = dose;
        document.getElementById("lbl-dose1").textContent = dose;
        _state.dose1 = dose;
        _triggerLiveOrNone();
      }
    });

    // CYP phenotype
    document.getElementById("cyp-phenotype")?.addEventListener("change", e => {
      _state.cypPhenotype = e.target.value;
      _triggerLiveOrNone();
    });

    // HLA-B allele
    document.getElementById("hla-b-allele")?.addEventListener("change", e => {
      _state.hlaBAllele = e.target.value;
    });

    // Gene deficiency checkboxes
    document.getElementById("dpyd-deficient")?.addEventListener("change", e => { _state.dpydDeficient = e.target.checked; });
    document.getElementById("g6pd-deficient")?.addEventListener("change", e => { _state.g6pdDeficient = e.target.checked; });
    document.getElementById("tpmt-low")?.addEventListener("change", e => { _state.tpmtLow = e.target.checked; });

    // Duration buttons (short-term only)
    document.querySelectorAll("#st-duration-section .dur-btn[data-h]").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("#st-duration-section .dur-btn[data-h]").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        _state.duration = parseInt(btn.dataset.h);
        _triggerLiveOrNone();
      });
    });

    // Time-scale buttons
    document.querySelectorAll(".ts-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".ts-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        _state.timeScale = parseInt(btn.dataset.scale) || 1;
        const lbl = document.getElementById("lbl-timescale");
        const hint = document.getElementById("timescale-hint");
        if (lbl) lbl.textContent = _state.timeScale + "×";
        if (hint) hint.textContent = `Each animation frame = ${_state.timeScale} hour${_state.timeScale > 1 ? "s" : ""}`;
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

  // ─── Drug interaction checker ─────────────────────────────────────────────
  function _checkInteractions() {
    const panel = document.getElementById("interaction-warning-panel");
    if (!panel) return;
    const selectedDrugs = [_state.drug1, _state.useDrug2 ? _state.drug2 : null, _state.useDrug3 ? _state.drug3 : null]
      .filter(Boolean);
    if (selectedDrugs.length < 2) { panel.style.display = "none"; return; }

    const warnings = [];
    // Check for same-class interactions
    for (let i = 0; i < selectedDrugs.length; i++) {
      for (let j = i + 1; j < selectedDrugs.length; j++) {
        const a = _drugsData.find(d => d.name === selectedDrugs[i]);
        const b = _drugsData.find(d => d.name === selectedDrugs[j]);
        if (!a || !b) continue;
        if (a.drug_class === b.drug_class) {
          warnings.push(`<div class="interaction-warn-item">⚠ ${a.name} + ${b.name}: same class (${a.drug_class}) — additive toxicity risk</div>`);
        }
        // QT-prolonging drug combo check
        const qtDrugs = ["Ciprofloxacin", "Sertraline", "Lorazepam"];
        if (qtDrugs.includes(a.name) && qtDrugs.includes(b.name)) {
          warnings.push(`<div class="interaction-warn-item">⚠ ${a.name} + ${b.name}: additive QT prolongation risk</div>`);
        }
        // CNS depressants
        const cnsDrugs = ["Lorazepam", "Sertraline"];
        if (cnsDrugs.includes(a.name) && cnsDrugs.includes(b.name)) {
          warnings.push(`<div class="interaction-warn-item">⚠ ${a.name} + ${b.name}: additive CNS depression</div>`);
        }
        // Renal/nephrotoxic combos
        const renalRisk = ["Ibuprofen", "Naproxen", "Ciprofloxacin"];
        if (renalRisk.includes(a.name) && renalRisk.includes(b.name)) {
          warnings.push(`<div class="interaction-warn-item">⚠ ${a.name} + ${b.name}: increased nephrotoxicity risk</div>`);
        }
      }
    }
    if (warnings.length === 0) { panel.style.display = "none"; return; }
    panel.innerHTML = `<div class="interaction-warn"><div class="interaction-warn-title">⚠ Drug Interactions Detected</div>${warnings.join("")}</div>`;
    panel.style.display = "";
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
    if (data.organ_health_final) {
      // Disease wave first, then health colour update
      HumanBody3D.playDiseaseEffect(data.organ_health_final);
      Object.entries(data.organ_health_final).forEach(([organId, health]) => {
        HumanBody3D.updateOrganHealth(organId, health);
      });
      // Medication absorption wave ~300 ms after health colours settle
      const organIds = Object.keys(data.organ_health_final);
      setTimeout(() => HumanBody3D.playMedicationEffect(organIds), 350);
    }
    // Animate organ health series if available
    if (data.organ_health_series && data.organ_health_series.length > 1) {
      HumanBody3D.animateSeries(data.organ_health_series, data.series_time_points, _state.timeScale);
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

    // Patient vitality banner
    _renderVitality(data);

    // Simulation timing
    _renderTiming(data);

    // Simulation insights (side effects analysis + key notes)
    _renderSimInsights(data);
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
    const genSec = document.getElementById("st-genetics-section");
    if (stDur) stDur.style.display = isShort ? "" : "none";
    if (ltDur) ltDur.style.display = _state.ltMode ? "" : "none";
    if (genSec) genSec.style.display = isShort ? "" : "none";
    // short-term rows
    [".row1",".row2",".row3"].forEach(cls => {
      const el = document.querySelector(cls);
      if (el) el.style.display = isShort ? "" : "none";
    });
    const pdRow  = document.getElementById("pd-row");
    if (pdRow) pdRow.style.display = "none";
    // hide insights when not in short mode
    const insRow = document.getElementById("insights-row");
    if (insRow && !isShort) insRow.style.display = "none";
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

  // ───────────────────────────────────────── organ click → cell detail drawer
  function _onOrganClick(organId, organHealth) {
    const drugNames = _state.lastResult?.drugs || [_state.drug1];
    CellView.open(organId, organHealth, _state.lastResult, drugNames);
  }

  // ───────────────────────────────────────── patient vitality banner
  function _renderVitality(data) {
    const banner   = document.getElementById("vitality-banner");
    const iconEl   = document.getElementById("vitality-icon");
    const outcomeEl = document.getElementById("vitality-outcome");
    const detailEl = document.getElementById("vitality-detail");
    const survEl   = document.getElementById("vitality-survival");
    const todEl    = document.getElementById("vitality-tod");
    const todWrap  = document.getElementById("vitality-tod-wrap");
    const pillsEl  = document.getElementById("vitality-organ-pills");
    if (!banner) return;

    const outcome = data.patient_outcome || "alive";
    const surv    = data.survival_probability ?? 0.99;
    const tod     = data.time_of_death;
    const fatal   = data.fatal_organs   || [];
    const crit    = data.critical_organs || [];
    const events  = data.organ_failure_events || [];

    banner.classList.remove("vitality-hidden","vitality-alive","vitality-critical","vitality-deceased");

    if (outcome === "deceased") {
      banner.classList.add("vitality-deceased");
      iconEl.innerHTML = `<svg width="32" height="32" viewBox="0 0 32 32" fill="none">
        <circle cx="16" cy="16" r="14" stroke="#ff1744" stroke-width="2"/>
        <path d="M8 16 L16 16" stroke="#ff1744" stroke-width="2.5" stroke-linecap="round"/>
        <text x="16" y="10" text-anchor="middle" font-size="9" fill="#ff1744" font-family="Inter,sans-serif">FLATLINE</text>
        <line x1="6" y1="16" x2="26" y2="16" stroke="#ff1744" stroke-width="1.5"/>
      </svg>`;
      outcomeEl.textContent = "PATIENT DECEASED";
      detailEl.textContent  = data.cause_of_death || "Multi-organ failure";
    } else if (outcome === "critical") {
      banner.classList.add("vitality-critical");
      iconEl.innerHTML = `<svg width="32" height="32" viewBox="0 0 32 32" fill="none">
        <circle cx="16" cy="16" r="14" stroke="#ff6d00" stroke-width="2"/>
        <path d="M4 16 Q8 8 10 16 Q12 24 14 16 Q16 8 18 16 Q20 24 22 16 Q24 8 28 16"
              stroke="#ff6d00" stroke-width="2" fill="none" stroke-linecap="round"/>
      </svg>`;
      outcomeEl.textContent = "CRITICAL CONDITION";
      detailEl.textContent  = data.cause_of_death || "Multiple organs in critical state";
    } else {
      banner.classList.add("vitality-alive");
      iconEl.innerHTML = `<svg width="32" height="32" viewBox="0 0 32 32" fill="none">
        <circle cx="16" cy="16" r="14" stroke="#00e676" stroke-width="2"/>
        <path d="M4 16 Q8 10 10 16 Q12 22 14 14 Q16 6 18 16 Q20 26 22 16 Q24 12 28 16"
              stroke="#00e676" stroke-width="2" fill="none" stroke-linecap="round"/>
      </svg>`;
      outcomeEl.textContent = "PATIENT ALIVE";
      detailEl.textContent  = crit.length
        ? `${crit.length} organ(s) under watch: ${crit.map(o=>o.replace("_"," ")).join(", ")}`
        : "All organs within survivable range";
    }

    survEl.textContent  = Math.round(surv * 100) + "%";

    if (tod != null) {
      todWrap.style.display = "";
      todEl.textContent = tod.toFixed(1) + " h";
    } else {
      todWrap.style.display = "none";
    }

    // Organ pills
    pillsEl.innerHTML = "";
    const allPills = [
      ...fatal.map(o => ({ organ: o, cls: "pill-fatal",    label: o.replace(/_/g," ") + " ✕" })),
      ...crit.map(o  => ({ organ: o, cls: "pill-critical", label: o.replace(/_/g," ") + " ⚠" })),
    ];
    allPills.forEach(({ organ, cls, label }) => {
      const span = document.createElement("span");
      span.className = `vitality-pill ${cls}`;
      span.textContent = label;
      pillsEl.appendChild(span);
    });

    // Failure events under body (if any)
    if (events.length) {
      const evSpan = document.createElement("span");
      evSpan.className = "vitality-pill pill-event";
      evSpan.textContent = `${events.length} failure event${events.length>1?"s":""}`;
      pillsEl.appendChild(evSpan);
    }
  }

  function _renderTiming(data) {
    const el = document.getElementById("sim-timing");
    if (!el) return;
    if (data.simulation_ms != null) {
      el.textContent = `Computed in ${data.simulation_ms.toFixed(0)} ms`;
      el.style.display = "";
    } else {
      el.style.display = "none";
    }
  }

  // ─────────────────────────────────────────── Side Effect Management Table
  const _SE_MANAGEMENT = {
    "Liver Toxicity":          "Monitor LFTs; consider N-acetylcysteine for severe cases; avoid alcohol",
    "Nausea":                  "Take with food; antiemetics if persistent; usually resolves within 1–2 weeks",
    "Allergic Reaction":       "STOP drug immediately; administer antihistamine; epinephrine if anaphylaxis",
    "GI Bleeding":             "STOP NSAID immediately; add PPI cover; endoscopy if haematemesis/melena",
    "Kidney Damage":           "Monitor creatinine & eGFR; reduce dose or discontinue; ensure hydration",
    "GI Irritation":           "Take with food; antacids or PPI; consider dose reduction",
    "Fluid Retention":         "Restrict sodium; consider low-dose diuretic; monitor daily weight",
    "Angioedema":              "EMERGENCY — stop drug immediately; epinephrine 0.3 mg IM; airway assessment",
    "Dry Cough":               "Switch to ARB (e.g. losartan) if cough persists beyond 2–4 weeks",
    "Hyperkalemia":            "Check serum K⁺; avoid K⁺ supplements & K-sparing diuretics; may reduce dose",
    "Hypotension":             "Lie supine; monitor BP; take first dose at night; ensure adequate fluid intake",
    "Bradycardia":             "Check HR and ECG; reduce dose; withhold if HR < 50 bpm",
    "Fatigue":                 "Usually resolves; dose in evening; review after 4–6 weeks",
    "Bronchospasm":            "STOP beta-blocker immediately; bronchodilator; avoid in asthma/COPD",
    "Cold Extremities":        "Reassure; compression stockings; switch to vasodilating beta-blocker if severe",
    "Myopathy":                "Check CK; stop statin if CK > 10× ULN; ensure adequate hydration",
    "Liver Enzyme Elev.":      "Monitor LFTs at 3 and 12 months; consider dose reduction if > 3× ULN",
    "Rhabdomyolysis":          "EMERGENCY — IV fluids; stop statin; creatinine monitoring; ICU if AKI",
    "Peripheral Oedema":       "Elevate legs; reduce salt; may add low-dose diuretic; usually benign",
    "Flushing":                "Usually self-limiting; take at bedtime; avoid hot drinks after dose",
    "Reflex Tachycardia":      "Monitor HR; add low-dose beta-blocker if symptomatic",
    "Lactic Acidosis":         "EMERGENCY — stop metformin; IV bicarbonate; consider dialysis if renal failure",
    "GI Distress":             "Take with food; start at lowest dose and titrate; probiotics may help",
    "Vitamin B12 Defic.":      "Annual B12 monitoring; supplement 500–1000 mcg/day if deficient",
    "Hypoglycaemia":           "15 g fast-acting carbs; recheck in 15 min; adjust dose/frequency",
    "Weight Gain":             "Diet & exercise review; consider switch to weight-neutral agent",
    "C.diff Infection":        "Isolate; oral vancomycin or fidaxomicin; stop causative antibiotic",
    "Diarrhoea":               "Oral rehydration; probiotics; stop if severe (> 6/day or with blood)",
    "Tendinopathy":            "STOP fluoroquinolone; rest; physiotherapy; avoid corticosteroids",
    "CNS Effects":             "Avoid driving; reduce dose if seizure risk; monitor neurological status",
    "QT Prolongation":         "Check ECG QTc; avoid co-administration with other QT-prolonging drugs",
    "Serotonin Syndrome":      "STOP serotonergic drug immediately; benzodiazepines for agitation; cyproheptadine",
    "Insomnia":                "Dose in the morning; sleep hygiene; melatonin if persistent",
    "Sedation":                "Avoid driving/alcohol; dose in evening; assess falls risk",
    "Respiratory Depr.":       "Monitor SpO₂; have reversal agent available; reduce dose or discontinue",
    "Dependence":              "Do NOT stop abruptly; taper over 4–8 weeks; consider CBT support",
    "Palpitations":            "Check ECG and TSH; dose adjustment likely needed; check for over-replacement",
    "Chest Pain":              "Urgent cardiology review; ECG; consider dose reduction or switch",
    "Hyperglycaemia":          "Monitor blood glucose; dietary review; may require insulin adjustment",
    "Immunosuppression":       "Prophylactic antibiotics if high-risk; avoid live vaccines; infection monitoring",
    "GI Ulceration":           "Add PPI co-prescription; always take with food; check for occult blood",
    "Osteoporosis":            "Calcium & vitamin D supplementation; DEXA scan; bisphosphonate if indicated",
    "Amoxicillin Reaction":    "STOP immediately; check cross-reactivity with other penicillins; MedicAlert",
  };

  // ─────────────────────────────────────────── Simulation Insights
  function _renderSimInsights(data) {
    const row = document.getElementById("insights-row");
    if (!row) return;
    row.style.display = "flex";
    _renderSEAnalysis(data);
    _renderKeyNotes(data);
  }

  function _renderSEAnalysis(data) {
    const container = document.getElementById("se-analysis-container");
    const hint = document.getElementById("se-analysis-hint");
    if (!container) return;

    const sideEffects = data.side_effects || [];
    if (hint) hint.textContent = sideEffects.length > 0 ? `${sideEffects.length} effect(s) detected` : "None detected";

    if (sideEffects.length === 0) {
      container.innerHTML = `<div style="padding:24px;text-align:center;color:var(--text-muted);font-size:11px">
        <div style="font-size:22px;margin-bottom:8px">✓</div>
        No adverse effects detected in this simulation period.<br/>
        See Key Notes for a summary of this simulation.
      </div>`;
      return;
    }

    // Sort: high → medium → low
    const sevOrder = { high: 0, medium: 1, low: 2 };
    const sorted = [...sideEffects].sort((a, b) =>
      (sevOrder[a.severity] ?? 2) - (sevOrder[b.severity] ?? 2)
    );

    // Deduplicate by name
    const seen = new Set();
    const unique = sorted.filter(e => { if (seen.has(e.name)) return false; seen.add(e.name); return true; });

    container.innerHTML = "";
    unique.forEach(e => {
      const sev = e.severity || "low";
      const onsetH = e.onset_time != null ? e.onset_time.toFixed(1) : "?";
      const durationH = e.duration != null ? e.duration.toFixed(0) : "?";
      const probPct = e.probability != null ? Math.round(e.probability * 100) : "?";
      const management = _SE_MANAGEMENT[e.name]
        || (sev === "high"   ? "URGENT — seek immediate medical review; do not delay"
          : sev === "medium" ? "Monitor closely; review with prescriber at next appointment"
          :                    "Typically self-limiting; manage symptomatically");

      const card = document.createElement("div");
      card.className = `se-card se-${sev}`;
      card.innerHTML = `
        <div class="se-card-header">
          <span class="se-name">${e.name}</span>
          <span class="se-sev-badge se-sev-${sev}">${sev}</span>
          <span class="se-onset-badge">Onset ~${onsetH}h</span>
        </div>
        <div class="se-meta">
          <span>Duration ~${durationH}h</span>
          <span class="se-prob">Likelihood: ${probPct}%</span>
        </div>
        ${e.description ? `<div class="se-description">${e.description}</div>` : ""}
        <div class="se-management"><span class="se-management-label">Management:</span>${management}</div>
      `;
      container.appendChild(card);
    });
  }

  function _renderKeyNotes(data) {
    const container = document.getElementById("sim-notes-container");
    if (!container) return;

    const firstDrug = data.drugs?.[0] || _state.drug1;
    const m = (firstDrug && data.pk_metrics?.[firstDrug]) ? data.pk_metrics[firstDrug] : null;
    const ts = data.therapeutic_status || "therapeutic";
    const health = data.overall_health ?? 0.85;

    const notes = [];

    // 1. Therapeutic status
    if (ts === "therapeutic") {
      notes.push({ cls: "note-ok", icon: "✓", title: "Therapeutic Range", text: `Plasma concentration is within the therapeutic window — effective dosing confirmed.` });
    } else if (ts === "sub-therapeutic") {
      notes.push({ cls: "note-warn", icon: "↓", title: "Sub-Therapeutic", text: `Concentration below minimum effective level. Consider increasing dose or frequency.` });
    } else if (ts === "toxic") {
      notes.push({ cls: "note-alert", icon: "⚠", title: "Toxic Exposure", text: `Concentration exceeds therapeutic window. Dose reduction or drug holiday recommended.` });
    } else {
      notes.push({ cls: "note-info", icon: "◎", title: "Status: " + ts.replace("-", " ").toUpperCase(), text: `Review simulation parameters and clinical context.` });
    }

    // 2. Peak concentration
    if (m) {
      notes.push({ cls: "note-info", icon: "⬆", title: "Peak Concentration",
        text: `C<sub>max</sub> = ${m.cmax?.toFixed(2) ?? "—"} mg/L at T<sub>max</sub> = ${m.tmax?.toFixed(1) ?? "—"}h · AUC = ${m.auc?.toFixed(0) ?? "—"} mg·h/L` });
    }

    // 3. Half-life and steady-state
    if (m && m.half_life) {
      const ss = (m.half_life * 4).toFixed(0);
      notes.push({ cls: "note-info", icon: "⏱", title: "Pharmacokinetics",
        text: `t<sub>½</sub> = ${m.half_life.toFixed(1)}h — steady-state reached in ~${ss}h (4×t<sub>½</sub>). ${m.mechanistic ? "Mechanistic PBPK model." : "Analytical PK model."}` });
    }

    // 4. Organ health summary
    const organFinal = data.organ_health_final || {};
    const weakOrgans = Object.entries(organFinal).filter(([, v]) => v < 0.7).map(([k]) => k.replace(/_/g, " "));
    if (weakOrgans.length === 0) {
      notes.push({ cls: "note-ok", icon: "🫀", title: "Organ Health", text: `All organs within healthy range (overall: ${Math.round(health * 100)}%).` });
    } else {
      notes.push({ cls: "note-warn", icon: "🫀", title: "Organ Health", text: `${weakOrgans.length} organ(s) under stress: ${weakOrgans.slice(0, 3).join(", ")}${weakOrgans.length > 3 ? " +" + (weakOrgans.length - 3) + " more" : ""}. (Overall: ${Math.round(health * 100)}%)` });
    }

    // 5. CYP / Metabolism note
    const cypLabels = { normal: "Normal metabolizer — standard dosing applies",
      poor: "Poor metabolizer — expect higher exposure; consider dose reduction",
      intermediate: "Intermediate metabolizer — slightly elevated exposure",
      ultra_rapid: "Ultra-rapid metabolizer — may need higher dose for efficacy" };
    notes.push({ cls: "note-info", icon: "⚗", title: "CYP Metabolism",
      text: cypLabels[_state.cypPhenotype] || "Standard metabolism assumed" });

    // 6. Drug combination note
    if (_state.useDrug2 && _state.drug2) {
      const drug2m = data.pk_metrics?.[_state.drug2];
      const drugList = [firstDrug, _state.drug2, _state.useDrug3 && _state.drug3 ? _state.drug3 : null].filter(Boolean);
      notes.push({ cls: "note-info", icon: "⊕", title: "Combination Therapy",
        text: `${drugList.join(" + ")} — monitor for additive effects. CYP interactions may alter plasma levels of co-administered drugs.` });
    }

    // 7. Genetics / allele flags
    const geneFlags = [];
    if (_state.hlaBAllele !== "none") geneFlags.push(`${_state.hlaBAllele} allele active — screen for hypersensitivity before administration`);
    if (_state.dpydDeficient) geneFlags.push("DPYD deficiency — avoid fluoropyrimidine-based chemotherapy");
    if (_state.g6pdDeficient) geneFlags.push("G6PD deficiency — avoid oxidant drugs (primaquine, dapsone, high-dose aspirin)");
    if (_state.tpmtLow) geneFlags.push("Low TPMT activity — thiopurine dose reduction required (azathioprine, mercaptopurine)");
    if (geneFlags.length > 0) {
      notes.push({ cls: "note-gene", icon: "🧬", title: "Genetic Flags", text: geneFlags.join(" · ") });
    }

    container.innerHTML = "";
    notes.forEach(n => {
      const card = document.createElement("div");
      card.className = `note-card ${n.cls}`;
      card.innerHTML = `<div class="note-icon">${n.icon}</div><div class="note-body"><div class="note-title">${n.title}</div><div class="note-text">${n.text}</div></div>`;
      container.appendChild(card);
    });
  }

  // ───────────────────────────────────────── init organ viz
  // HumanBody3D is already initialized in boot() function

  // ───────────────────────────────────────── start
  boot();
})();
