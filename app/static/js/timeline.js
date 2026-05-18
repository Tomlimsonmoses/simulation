/* ================================================================
   timeline.js — D3.js Gantt-style side effects timeline
   ================================================================ */

/* Shared knowledge base (also used by biomarkers.js) */
const SE_KNOWLEDGE = {
  "hepatotoxicity": {
    cause: "Drug metabolism generates reactive intermediates that damage hepatocytes or disrupt bile flow.",
    reduce: "Monitor ALT/AST every 4–8 weeks. Avoid alcohol. Reduce dose if LFTs rise >3× ULN. Ensure adequate hydration."
  },
  "liver toxicity": {
    cause: "Drug or metabolites cause direct hepatocellular stress at supratherapeutic concentrations.",
    reduce: "Monitor LFTs regularly. Avoid alcohol and other hepatotoxins. Dose reduction or switch drug if persistent."
  },
  "liver enzyme": {
    cause: "Drug-induced transaminase elevation through hepatocellular stress or cholestasis.",
    reduce: "Monitor LFTs every 4–8 weeks. Usually resolves with dose reduction or discontinuation."
  },
  "nephrotoxicity": {
    cause: "Drug or metabolites reduce renal perfusion or directly damage proximal tubular cells.",
    reduce: "Maintain hydration (≥2 L/day). Dose-adjust for eGFR. Avoid concurrent NSAIDs or aminoglycosides. Monitor creatinine."
  },
  "kidney damage": {
    cause: "Prostaglandin inhibition reduces renal blood flow, risking acute tubular ischaemia.",
    reduce: "Maintain hydration. Monitor eGFR. Avoid co-prescribing with other nephrotoxins. Reduce dose in renal impairment."
  },
  "neurotoxicity": {
    cause: "CNS penetration of drug or metabolites disrupts neuronal membrane function or neurotransmitter balance.",
    reduce: "Lower dose. Consider less CNS-penetrant alternative. Monitor for confusion, tremor or seizures."
  },
  "cardiotoxicity": {
    cause: "Drug affects cardiac ion channels (hERG, Na⁺, Ca²⁺) or mitochondrial function, increasing arrhythmia risk.",
    reduce: "Baseline and periodic ECG. Correct electrolytes (K⁺, Mg²⁺). Avoid other QT-prolonging agents. Cardiology review."
  },
  "gi mucosal injury": {
    cause: "COX inhibition reduces gastric protective prostaglandins, weakening the mucosal barrier.",
    reduce: "Take with food. Add proton pump inhibitor (e.g. omeprazole 20 mg/day). Limit NSAID duration."
  },
  "gi bleeding": {
    cause: "NSAID-mediated gastropathy leads to mucosal erosion, ulceration and haemorrhage.",
    reduce: "Add PPI prophylaxis. Take with food. Test and treat H. pylori. Consider switching to a COX-2 selective agent."
  },
  "gi irritation": {
    cause: "Direct mucosal irritation from the acidic properties of the drug.",
    reduce: "Take with food or milk. Use enteric-coated formulation if available. Add antacid."
  },
  "gi distress": {
    cause: "Drug alters gut motility or disrupts microbiome, causing nausea, cramping and diarrhoea.",
    reduce: "Take with meals. Titrate dose slowly upward. Consider probiotics (Lactobacillus) to protect microbiome."
  },
  "gi ulceration": {
    cause: "Steroid-induced reduction in prostaglandin synthesis weakens gastric mucosa and impairs healing.",
    reduce: "Co-prescribe PPI. Take with food. Use lowest effective corticosteroid dose for shortest duration."
  },
  "myopathy": {
    cause: "Statin-induced CoQ10 depletion and mitochondrial dysfunction impair muscle energy metabolism.",
    reduce: "Check serum CK. Consider CoQ10 supplementation (100–200 mg/day). Switch to lower-dose or hydrophilic statin."
  },
  "myotoxicity": {
    cause: "Drug accumulates in muscle tissue causing mitochondrial uncoupling and myofibril damage.",
    reduce: "Monitor CK. Ensure adequate hydration. Reduce dose or pause drug if CK rises >5× ULN."
  },
  "rhabdomyolysis": {
    cause: "Massive muscle fibre breakdown releases myoglobin into circulation, risking acute kidney injury.",
    reduce: "Stop offending drug immediately. Aggressive IV hydration (target urine output >200 mL/hr). Monitor CK, creatinine and urine output."
  },
  "muscle toxicity": {
    cause: "Drug-induced sarcolemmal damage causes leakage of muscle enzymes and contractile dysfunction.",
    reduce: "Check CK levels. Reduce dose. Instruct patient to report new muscle pain, weakness or dark urine promptly."
  },
  "pulmonary toxicity": {
    cause: "Drug or metabolites damage alveolar epithelial cells or trigger immune-mediated interstitial inflammation.",
    reduce: "Monitor for new dyspnoea or cough. High-resolution CT if symptoms develop. Dose reduction or drug cessation may be required."
  },
  "nausea": {
    cause: "Drug stimulates the chemoreceptor trigger zone or directly irritates the gastric mucosa.",
    reduce: "Take with food. Use antiemetics (ondansetron, metoclopramide) if persistent. Start at low dose and titrate up."
  },
  "headache": {
    cause: "Drug alters cerebral blood flow or triggers prostaglandin-mediated vascular changes.",
    reduce: "Ensure adequate hydration. Simple analgesia (paracetamol). Reduce dose if severe or persisting beyond 2 weeks."
  },
  "allergic reaction": {
    cause: "IgE-mediated hypersensitivity or direct mast cell activation by the drug molecule or excipient.",
    reduce: "Discontinue drug immediately. Antihistamine for mild reactions. Epinephrine + emergency care for anaphylaxis. Document allergy."
  },
  "bradycardia": {
    cause: "Beta-1 adrenoceptor blockade reduces sinoatrial node firing rate and AV nodal conduction.",
    reduce: "Monitor resting HR. Reduce dose if HR <50 bpm. Avoid concurrent bradycardic agents (digoxin, diltiazem)."
  },
  "reflex tachycardia": {
    cause: "Vasodilatory effect triggers sympathetic reflex to maintain perfusion pressure.",
    reduce: "Combine with a beta-blocker if clinically appropriate. Monitor resting HR. Ensure adequate hydration."
  },
  "palpitations": {
    cause: "Supraphysiological drug or hormone levels stimulate cardiac adrenergic receptors.",
    reduce: "Recheck serum levels and adjust dose. Avoid caffeine. Obtain ECG if palpitations are sustained."
  },
  "chest pain": {
    cause: "Increased cardiac demand from drug-induced tachycardia in patients with ischaemic disease.",
    reduce: "Reduce dose. Obtain ECG and cardiac enzymes if episode occurs. Formally assess coronary artery disease risk."
  },
  "angioedema": {
    cause: "ACE inhibitor-induced bradykinin accumulation causes submucosal oedema of face, lips or tongue.",
    reduce: "Discontinue ACE inhibitor immediately. Switch to ARB (e.g. losartan) which does not cause bradykinin accumulation."
  },
  "dry cough": {
    cause: "ACE inhibitor-induced bradykinin and substance P accumulation irritates airway sensory nerves.",
    reduce: "Switch to an ARB (e.g. losartan) which provides equivalent BP control without bradykinin-mediated cough."
  },
  "hyperkalemia": {
    cause: "ACE inhibitor reduces aldosterone secretion, decreasing renal potassium excretion.",
    reduce: "Monitor serum potassium weekly initially. Reduce dietary potassium intake. Adjust dose. Review K-sparing co-medications."
  },
  "hypotension": {
    cause: "Vasodilatory or diuretic mechanism causes BP to fall, especially with first dose or dehydration.",
    reduce: "Start at lowest dose. Take first dose at bedtime. Rise slowly from lying to standing. Ensure adequate hydration."
  },
  "sedation": {
    cause: "GABAergic enhancement causes CNS depression, reducing arousal and psychomotor performance.",
    reduce: "Use lowest effective dose. Avoid alcohol and other CNS depressants. Do not drive or operate machinery while sedated."
  },
  "respiratory depr": {
    cause: "GABAergic and opioid receptor activation suppresses brainstem respiratory centres.",
    reduce: "Administer at lowest effective dose. Have flumazenil reversal agent available. Avoid concurrent opioids or alcohol."
  },
  "dependence": {
    cause: "Chronic GABA enhancement leads to receptor downregulation, neuroadaptation and withdrawal syndrome.",
    reduce: "Limit benzodiazepine use to <2–4 weeks. Taper dose slowly (≤10% per week). Explore CBT and non-pharmacological alternatives."
  },
  "serotonin syndrome": {
    cause: "Excess serotonin accumulation, especially when combined with MAOIs, tramadol or other serotonergic drugs.",
    reduce: "Avoid co-prescribing with MAOIs (washout ≥14 days). Stop drug if fever, clonus or agitation develops. Seek emergency care."
  },
  "insomnia": {
    cause: "Drug stimulates CNS adrenergic or thyroid hormone pathways, disrupting sleep architecture.",
    reduce: "Take morning doses. Reduce dose if possible. Melatonin or sleep hygiene measures. Avoid caffeine after midday."
  },
  "fatigue": {
    cause: "Beta-blockade reduces cardiac output and blunts adrenergic compensatory responses to exertion.",
    reduce: "Reduce dose if possible. Regular moderate exercise to maintain fitness. Exclude anaemia or hypothyroidism."
  },
  "bronchospasm": {
    cause: "Beta-2 adrenoceptor blockade in airway smooth muscle causes bronchoconstriction.",
    reduce: "Use cardioselective beta-1 blocker (e.g. bisoprolol) instead. Avoid in asthma/COPD. Have rescue inhaler available."
  },
  "cold extremities": {
    cause: "Peripheral vasoconstriction from beta-blockade reduces blood flow to hands and feet.",
    reduce: "Keep extremities warm. Consider switching to a vasodilatory antihypertensive if symptoms are severe."
  },
  "peripheral oedema": {
    cause: "Vasodilatory effect increases capillary hydrostatic pressure, shifting fluid into interstitium.",
    reduce: "Elevate legs when resting. Compression stockings. Consider switching to ACE inhibitor or ARB if oedema is troublesome."
  },
  "flushing": {
    cause: "Direct peripheral vasodilation increases cutaneous blood flow, causing warmth and redness.",
    reduce: "Usually self-limiting. Take aspirin 325 mg 30 min before niacin doses. Reduce dose initially and titrate up slowly."
  },
  "hyperglycaemia": {
    cause: "Corticosteroids antagonise insulin signalling and increase hepatic glucose output.",
    reduce: "Monitor fasting glucose daily in diabetic patients. Adjust diabetes medication dose. Use lowest effective steroid dose."
  },
  "immunosuppression": {
    cause: "Corticosteroids inhibit T-lymphocyte proliferation and pro-inflammatory cytokine production.",
    reduce: "Use shortest course possible. Prophylactic co-trimoxazole for prolonged courses. Avoid live vaccines. Infection vigilance."
  },
  "osteoporosis": {
    cause: "Long-term corticosteroid use inhibits osteoblasts and increases renal calcium excretion.",
    reduce: "Calcium (1000 mg/day) + Vitamin D3 (800 IU/day). DEXA scan for courses >3 months. Consider bisphosphonate."
  },
  "lactic acidosis": {
    cause: "Mitochondrial complex I inhibition impairs lactate clearance; risk is greatly elevated in renal impairment.",
    reduce: "Hold metformin if eGFR <30 mL/min or patient is acutely unwell, dehydrated or having contrast imaging."
  },
  "vitamin b12": {
    cause: "Metformin inhibits ileal cubilin receptor, blocking B12–intrinsic factor complex absorption.",
    reduce: "Annual serum B12 monitoring. Supplement orally (1000 µg/day cyanocobalamin) or IM injection if deficient."
  },
  "hypoglycaemia": {
    cause: "Sulphonylurea stimulates pancreatic insulin secretion independent of blood glucose level.",
    reduce: "Reduce dose if meals are skipped. Carry glucose tablets (15–20 g fast-acting carb). Consider switching to GLP-1/SGLT-2."
  },
  "weight gain": {
    cause: "Insulin secretion and anabolic storage effects promote adipose tissue accumulation.",
    reduce: "Dietary counselling and regular exercise. Consider switching to weight-neutral/favourable agent (SGLT-2 or GLP-1 agonist)."
  },
  "diarrhoea": {
    cause: "Antibiotic-induced dysbiosis alters gut motility and impairs water absorption in the large bowel.",
    reduce: "Take probiotic (Lactobacillus rhamnosus) during antibiotic course. Maintain hydration. Reduce dose if severe."
  },
  "c.diff": {
    cause: "Dysbiosis from broad-spectrum antibiotic use allows Clostridioides difficile spore germination and toxin production.",
    reduce: "Consider probiotics prophylactically. Discontinue antibiotic if possible. Treat C. diff with oral vancomycin or fidaxomicin."
  },
  "tendinopathy": {
    cause: "Fluoroquinolone-mediated inhibition of tenocyte collagen synthesis and mitochondrial function in tendons.",
    reduce: "Avoid fluoroquinolones in elderly patients or those on corticosteroids. Stop drug immediately if tendon pain develops."
  },
  "cns effects": {
    cause: "Drug crosses the blood-brain barrier and alters neuronal excitability or neurotransmitter systems.",
    reduce: "Reduce dose. Avoid activities requiring full alertness. Monitor for seizure threshold changes in susceptible patients."
  },
  "qt prolongation": {
    cause: "Drug blocks hERG (IKr) potassium channels, delaying ventricular repolarisation and prolonging QTc interval.",
    reduce: "Baseline and periodic ECG. Avoid other QT-prolonging drugs. Correct electrolytes (K⁺ >4.0, Mg²⁺ >0.8 mmol/L)."
  },
  "fluid retention": {
    cause: "Prostaglandin inhibition reduces renal sodium excretion, promoting water retention and oedema.",
    reduce: "Monitor weight and blood pressure. Reduce NSAID dose or switch to paracetamol. Add a loop diuretic if oedema is significant."
  },
};

function _seKBLookup(name) {
  const n = (name || "").toLowerCase();
  for (const key of Object.keys(SE_KNOWLEDGE)) {
    if (n.includes(key)) return SE_KNOWLEDGE[key];
  }
  return null;
}

/* ================================================================ */

const Timeline = (() => {
  const SEV_COLOR = { high: "#ef5350", medium: "#ffab40", low: "#00e676" };
  const SEV_RANK  = { high: 0, medium: 1, low: 2 };

  function _fmtTime(h) {
    if (h == null) return "";
    if (h < 48)   return `${Math.round(h)}h`;
    if (h < 336)  return `Day ${Math.round(h / 24)}`;
    if (h < 2160) return `Wk ${Math.round(h / 168)}`;
    return `Mo ${Math.round(h / 720)}`;
  }

  function _fmtDur(h) {
    if (h == null) return "";
    if (h < 48)   return `~${Math.round(h)}h`;
    if (h < 336)  return `~${Math.round(h / 24)}d`;
    if (h < 2160) return `~${Math.round(h / 168)}wk`;
    return `~${Math.round(h / 720)}mo`;
  }

  function render(sideEffects, durationHours) {
    const container = document.getElementById("timeline-container");
    if (!container) return;

    container.innerHTML = "";

    if (!sideEffects || sideEffects.length === 0) {
      container.innerHTML = `
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:140px;gap:10px;">
          <div style="width:46px;height:46px;border-radius:50%;background:rgba(0,230,118,0.08);border:1.5px solid rgba(0,230,118,0.25);display:flex;align-items:center;justify-content:center;font-size:22px">✓</div>
          <div style="color:#3d4e6b;font-size:12px;text-align:center;line-height:1.6">
            <strong style="color:#00e676;display:block;margin-bottom:2px;font-size:13px">No Adverse Effects</strong>
            No side effects detected in this simulation period
          </div>
        </div>`;
      return;
    }

    // Deduplicate, sort high → medium → low
    const seen = new Map();
    sideEffects.forEach(e => { if (!seen.has(e.name)) seen.set(e.name, e); });
    const effects = Array.from(seen.values())
      .sort((a, b) => (SEV_RANK[a.severity] ?? 2) - (SEV_RANK[b.severity] ?? 2));

    const highCount = effects.filter(e => e.severity === "high").length;
    const medCount  = effects.filter(e => e.severity === "medium").length;

    // Summary strip
    container.insertAdjacentHTML("beforeend", `
      <div style="display:flex;align-items:center;gap:8px;padding:6px 4px 4px;flex-wrap:wrap;">
        <span style="font-size:11px;color:#7b8fb5;text-transform:uppercase;letter-spacing:.05em">
          ${effects.length} effect${effects.length !== 1 ? "s" : ""} detected
        </span>
        ${highCount ? `<span style="background:#ef535022;border:1px solid #ef5350;color:#ef5350;border-radius:4px;padding:1px 7px;font-size:10px;font-weight:700">${highCount} High</span>` : ""}
        ${medCount  ? `<span style="background:#ffab4022;border:1px solid #ffab40;color:#ffab40;border-radius:4px;padding:1px 7px;font-size:10px;font-weight:700">${medCount} Medium</span>` : ""}
        <span style="font-size:10px;color:#4a5a70;margin-left:auto">Hover bars for details</span>
      </div>`);

    // Gantt chart (cap rows at 8)
    const chartEffects = effects.slice(0, 8);
    const margin = { top: 8, right: 20, bottom: 28, left: 150 };
    const cw  = container.clientWidth || 520;
    const rowH = 24;
    const ch  = chartEffects.length * rowH + margin.top + margin.bottom;
    const W   = cw - margin.left - margin.right;

    const svg = d3.select(container).append("svg").attr("width", cw).attr("height", ch);
    const g   = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    const xScale = d3.scaleLinear().domain([0, durationHours || 72]).range([0, W]);

    // Grid
    g.append("g").attr("class","chart-grid")
      .call(d3.axisBottom(xScale).tickSize(chartEffects.length * rowH).ticks(6).tickFormat(""))
      .selectAll("line").attr("stroke","rgba(255,255,255,0.04)");
    g.select(".chart-grid .domain").remove();

    // Shared tooltip
    let tip = document.getElementById("_se-tooltip");
    if (!tip) {
      tip = document.createElement("div");
      tip.id = "_se-tooltip";
      tip.style.cssText = "position:fixed;pointer-events:none;z-index:9999;background:#0e1c2a;border:1px solid rgba(255,255,255,0.12);border-radius:8px;padding:10px 14px;font-size:12px;color:#c8d8e8;max-width:320px;display:none;line-height:1.6;box-shadow:0 4px 20px rgba(0,0,0,0.5)";
      document.body.appendChild(tip);
    }

    chartEffects.forEach((e, i) => {
      const y   = i * rowH + 2;
      const col = SEV_COLOR[e.severity] || "#7b8fb5";
      const dur = durationHours || 72;
      const x0  = xScale(Math.min(e.onset_time ?? 0, dur));
      const x1  = xScale(Math.min((e.onset_time ?? 0) + (e.duration || 0), dur));
      const bw  = Math.max(x1 - x0, 4);

      g.append("rect").attr("x",0).attr("y",y).attr("width",W).attr("height",rowH-3)
        .attr("fill", i%2===0 ? "rgba(255,255,255,0.015)" : "transparent");

      g.append("rect").attr("x",x0).attr("y",y+2).attr("width",0).attr("height",rowH-7)
        .attr("rx","3").attr("fill",col).attr("opacity","0.82")
        .transition().delay(i*50).duration(550).attr("width",bw);

      g.append("line").attr("x1",x0).attr("x2",x0).attr("y1",y+2).attr("y2",y+rowH-5)
        .attr("stroke",col).attr("stroke-width","1.5").attr("opacity","0.6");

      if (bw > 40) {
        g.append("text").attr("x",x0+bw/2).attr("y",y+rowH/2+1)
          .attr("text-anchor","middle").attr("alignment-baseline","middle")
          .attr("font-size","8").attr("fill","rgba(0,0,0,0.85)").attr("font-weight","800")
          .attr("font-family","Inter,sans-serif").text(e.severity.toUpperCase());
      }

      // Hover zone — tooltip includes cause + reduce from knowledge base
      g.append("rect").attr("x",0).attr("y",y).attr("width",W).attr("height",rowH-3)
        .attr("fill","transparent").style("cursor","pointer")
        .on("mousemove", evt => {
          const kb = _seKBLookup(e.name);
          const onsetStr = e.onset_time != null ? `Onset: ${_fmtTime(e.onset_time)}` : "";
          const durStr   = e.duration   != null ? `Duration: ${_fmtDur(e.duration)}`  : "";
          tip.innerHTML = `
            <strong style="color:${col};font-size:13px">${e.name}</strong><br>
            <span style="color:#7b8fb5;font-size:10px;text-transform:uppercase;letter-spacing:.05em">${e.severity} severity</span>
            ${onsetStr || durStr ? `<br><span style="color:#9aaabf;font-size:11px">${onsetStr}${onsetStr && durStr ? " · " : ""}${durStr}</span>` : ""}
            ${kb ? `
              <div style="margin-top:7px;border-top:1px solid rgba(255,255,255,0.07);padding-top:7px">
                <div style="margin-bottom:4px"><span style="color:#ffab40;font-size:10px;font-weight:700">⚡ CAUSE</span><br><span style="color:#8a9aae;font-size:11px">${kb.cause}</span></div>
                <div><span style="color:#00e676;font-size:10px;font-weight:700">✓ HOW TO REDUCE</span><br><span style="color:#8a9aae;font-size:11px">${kb.reduce}</span></div>
              </div>` : (e.description ? `<div style="color:#8a9aae;font-size:11px;margin-top:5px;border-top:1px solid rgba(255,255,255,0.07);padding-top:5px">${e.description}</div>` : "")}`;
          tip.style.display = "block";
          tip.style.left = (evt.clientX+14)+"px";
          tip.style.top  = (evt.clientY-10)+"px";
        })
        .on("mouseleave", () => { tip.style.display = "none"; });

      const label = e.name.length > 18 ? e.name.slice(0,17)+"…" : e.name;
      g.append("text").attr("x",-6).attr("y",y+rowH/2+1)
        .attr("text-anchor","end").attr("alignment-baseline","middle")
        .attr("font-size","10.5").attr("fill","#9aaabf").attr("font-family","Inter,sans-serif")
        .text(label);
    });

    // X axis with smart time labels
    g.append("g").attr("class","chart-axis")
      .attr("transform",`translate(0,${chartEffects.length*rowH})`)
      .call(d3.axisBottom(xScale).ticks(6).tickFormat(d => _fmtTime(d)))
      .selectAll("text").attr("fill","#7b8fb5").attr("font-size","10");
    g.select(".chart-axis .domain").attr("stroke","rgba(255,255,255,0.1)");

    // Legend
    const legG = svg.append("g").attr("transform",`translate(${margin.left},${ch-10})`);
    [{s:"high",l:"High"},{s:"medium",l:"Medium"},{s:"low",l:"Low"}].forEach((item,i) => {
      legG.append("rect").attr("x",i*72).attr("y",0).attr("width",10).attr("height",10).attr("rx","2").attr("fill",SEV_COLOR[item.s]);
      legG.append("text").attr("x",i*72+14).attr("y",9).attr("font-size","9").attr("fill","#7b8fb5").attr("font-family","Inter,sans-serif").text(item.l);
    });

    // Event cards with cause + recommendation
    const listDiv = document.createElement("div");
    listDiv.style.cssText = "margin-top:10px;display:flex;flex-direction:column;gap:6px;max-height:220px;overflow-y:auto;padding:0 2px 4px";

    effects.forEach(e => {
      const col   = SEV_COLOR[e.severity] || "#7b8fb5";
      const bgMap = { high: "rgba(239,83,80,0.07)", medium: "rgba(255,171,64,0.07)", low: "rgba(0,230,118,0.06)" };
      const bdMap = { high: "rgba(239,83,80,0.28)", medium: "rgba(255,171,64,0.22)", low: "rgba(0,230,118,0.18)" };
      const icon  = e.severity === "high" ? "🚨" : e.severity === "medium" ? "⚠️" : "ℹ️";
      const onsetStr = e.onset_time != null ? _fmtTime(e.onset_time) : null;
      const durStr   = e.duration   != null ? _fmtDur(e.duration)    : null;
      const timing   = [onsetStr ? `Onset ${onsetStr}` : null, durStr ? `${durStr} duration` : null].filter(Boolean).join(" · ");
      const kb = _seKBLookup(e.name);

      const card = document.createElement("div");
      card.style.cssText = `background:${bgMap[e.severity]||"transparent"};border:1px solid ${bdMap[e.severity]||"rgba(255,255,255,0.08)"};border-left:3px solid ${col};border-radius:6px;padding:8px 11px;display:flex;gap:8px;align-items:flex-start`;
      card.innerHTML = `
        <span style="font-size:14px;line-height:1.4;margin-top:2px">${icon}</span>
        <div style="flex:1;min-width:0">
          <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:3px">
            <strong style="color:${col};font-size:11.5px">${e.name}</strong>
            <span style="background:${col};color:#000;border-radius:3px;padding:0 5px;font-size:9px;font-weight:800;letter-spacing:.04em">${e.severity.toUpperCase()}</span>
            ${timing ? `<span style="color:#6a7f9e;font-size:10px;margin-left:auto">${timing}</span>` : ""}
          </div>
          ${e.description ? `<div style="color:#7b8fb5;font-size:10.5px;line-height:1.45;margin-bottom:${kb ? 6 : 0}px">${e.description}</div>` : ""}
          ${kb ? `
            <div style="background:rgba(255,171,64,0.06);border:1px solid rgba(255,171,64,0.15);border-radius:4px;padding:5px 8px;margin-bottom:5px">
              <div style="color:#ffab40;font-size:9.5px;font-weight:700;letter-spacing:.04em;margin-bottom:2px">⚡ CAUSE</div>
              <div style="color:#8a9aae;font-size:10.5px;line-height:1.45">${kb.cause}</div>
            </div>
            <div style="background:rgba(0,230,118,0.05);border:1px solid rgba(0,230,118,0.15);border-radius:4px;padding:5px 8px">
              <div style="color:#00e676;font-size:9.5px;font-weight:700;letter-spacing:.04em;margin-bottom:2px">✓ HOW TO REDUCE</div>
              <div style="color:#8a9aae;font-size:10.5px;line-height:1.45">${kb.reduce}</div>
            </div>` : ""}
        </div>`;
      listDiv.appendChild(card);
    });

    container.appendChild(listDiv);
  }

  return { render };
})();
