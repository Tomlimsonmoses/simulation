"""
Realistic Multi-Scale Simulator
=================================
Integrates the biological cascade, organ damage, and long-term adaptation engines
into a unified simulation that spans hours → years.

Architecture
------------
Input:  drug(s), patient parameters, disease state, simulation duration
Output: ClinicalTimeline — full time-series of physiological variables and events

Time-scale handling
-------------------
Duration   | Resolution  | Dominant processes
-----------|-------------|---------------------------------------------------
0 – 72 h   | 30 min      | PK absorption/distribution, acute immune response
72h – 30d  | 2 h         | Inflammatory resolution, early side-effect onset
30d – 6 mo | 12 h        | Metabolic adaptation, tolerance, enzyme induction
6 mo – 5 y | 24 h        | Structural remodeling (bone, LDL), neuroplasticity

All four phases are concatenated into a single seamless timeline.
State is carried forward between phases.

Clinical events are detected by threshold-crossing on key variables and
annotated with severity (INFO / WARNING / CRITICAL) and organ system.
"""

from __future__ import annotations

import logging
import sys
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure src/ is importable when module is used from app/
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.normpath(os.path.join(_HERE, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from simulation.biological_cascade import (
    BiologicalCascadeEngine, CascadeInput, CascadeResult,
    DISEASE_INITIAL_STATES, build_cascade_input_for_drug,
)
from simulation.organ_damage_model import (
    OrganDamageEngine, DamageState, BIOMARKER_BASELINE, BM_NAMES, interpret_biomarker,
)
from simulation.long_term_adaptation import (
    LongTermAdaptationEngine, ADP_NAMES, ADAPTATION_BASELINE,
    compute_clearance_multiplier, compute_antidepressant_response,
    compute_withdrawal_risk,
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ClinicalEvent:
    """A discrete clinical event detected during simulation."""
    time_h: float
    severity: str        # 'info' | 'warning' | 'critical'
    organ_system: str    # e.g. 'liver', 'kidney', 'cardiovascular'
    title: str
    description: str
    biomarker_name: Optional[str] = None
    biomarker_value: Optional[float] = None
    biomarker_unit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_h": round(self.time_h, 1),
            "time_label": _format_time_label(self.time_h),
            "severity": self.severity,
            "organ_system": self.organ_system,
            "title": self.title,
            "description": self.description,
            "biomarker_name": self.biomarker_name,
            "biomarker_value": round(self.biomarker_value, 2) if self.biomarker_value is not None else None,
            "biomarker_unit": self.biomarker_unit,
        }


@dataclass
class LongTermSimulationResult:
    """Complete output of the realistic multi-scale simulation."""
    # Time series
    time_h: np.ndarray
    time_labels: List[str]

    # PK
    plasma_conc: Dict[str, np.ndarray]  # drug → concentration (mg/L)

    # Immune / inflammatory
    crp_series: np.ndarray          # mg/L
    il6_series: np.ndarray          # pg/mL
    tnfa_series: np.ndarray         # pg/mL
    wbc_series: np.ndarray          # ×10⁹/L
    cortisol_series: np.ndarray     # nmol/L
    glucose_series: np.ndarray      # mmol/L
    tsh_series: np.ndarray          # mIU/L
    platelet_series: np.ndarray     # ×10⁹/L
    pathogen_series: np.ndarray     # log₁₀ CFU equivalent
    inflammation_index: np.ndarray  # 0–1

    # Organ biomarkers
    biomarkers: Dict[str, np.ndarray]   # name → array (clinical units)

    # Adaptation
    adaptations: Dict[str, np.ndarray]  # name → array (normalised)

    # Derived metrics
    antidepressant_response: Optional[np.ndarray]   # 0–1 (for SSRIs)
    clearance_multiplier: Dict[str, np.ndarray]      # drug → time-varying CL ratio
    withdrawal_risk: Dict[str, np.ndarray]           # drug → 0–1 risk

    # Disease progression
    disease_severity_series: np.ndarray

    # Clinical events
    events: List[ClinicalEvent]

    # Summary
    peak_crp: float
    peak_wbc: float
    infection_cleared_at_h: Optional[float]
    cough_onset_h: Optional[float]
    myopathy_onset_h: Optional[float]
    bone_loss_pct_per_year: float
    ldl_change_pct: float
    b12_depletion_pct: float

    def to_api_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serialisable dict for API response."""
        def _round_arr(arr, dec=3):
            if arr is None:
                return None
            return [round(float(v), dec) for v in arr]

        def _thin(arr, max_pts: int = 300):
            """Thin array to ≤max_pts points for API response."""
            if arr is None or len(arr) == 0:
                return arr
            step = max(1, len(arr) // max_pts)
            return arr[::step]

        t_thin = self.time_h[::max(1, len(self.time_h) // 300)]
        thin_factor = max(1, len(self.time_h) // 300)

        return {
            "time_h": _round_arr(self.time_h[::thin_factor], 1),
            "time_labels": self.time_labels[::thin_factor],

            # PK
            "plasma_concentrations": {
                drug: _round_arr(conc[::thin_factor])
                for drug, conc in self.plasma_conc.items()
            },

            # Immune / inflammation
            "immune_markers": {
                "crp":       _round_arr(self.crp_series[::thin_factor], 2),
                "il6":       _round_arr(self.il6_series[::thin_factor], 2),
                "tnf_alpha": _round_arr(self.tnfa_series[::thin_factor], 2),
                "wbc":       _round_arr(self.wbc_series[::thin_factor], 2),
                "cortisol":  _round_arr(self.cortisol_series[::thin_factor], 1),
                "glucose":   _round_arr(self.glucose_series[::thin_factor], 2),
                "tsh":       _round_arr(self.tsh_series[::thin_factor], 3),
                "platelets": _round_arr(self.platelet_series[::thin_factor], 1),
                "pathogen":  _round_arr(self.pathogen_series[::thin_factor], 3),
                "inflammation_index": _round_arr(self.inflammation_index[::thin_factor], 3),
            },

            # Organ biomarkers
            "biomarkers": {
                k: _round_arr(v[::thin_factor])
                for k, v in self.biomarkers.items()
            },

            # Adaptation state
            "adaptations": {
                k: _round_arr(v[::thin_factor], 4)
                for k, v in self.adaptations.items()
            },

            # Derived
            "antidepressant_response": _round_arr(
                self.antidepressant_response[::thin_factor] if self.antidepressant_response is not None else None,
                3
            ),
            "clearance_multipliers": {
                drug: _round_arr(arr[::thin_factor], 3)
                for drug, arr in self.clearance_multiplier.items()
            },
            "withdrawal_risks": {
                drug: _round_arr(arr[::thin_factor], 3)
                for drug, arr in self.withdrawal_risk.items()
            },

            "disease_severity_series": _round_arr(self.disease_severity_series[::thin_factor], 3),

            # Events timeline
            "events": [e.to_dict() for e in self.events],

            # Summary
            "summary": {
                "peak_crp_mg_l":        round(self.peak_crp, 1),
                "peak_wbc_10e9_l":      round(self.peak_wbc, 2),
                "infection_cleared_h":  round(self.infection_cleared_at_h, 1) if self.infection_cleared_at_h else None,
                "cough_onset_h":        round(self.cough_onset_h, 1) if self.cough_onset_h else None,
                "myopathy_onset_h":     round(self.myopathy_onset_h, 1) if self.myopathy_onset_h else None,
                "bone_loss_pct_per_year": round(self.bone_loss_pct_per_year, 2),
                "ldl_change_pct":        round(self.ldl_change_pct, 1),
                "b12_depletion_pct":     round(self.b12_depletion_pct, 1),
            },
        }


# ---------------------------------------------------------------------------
# Realistic Simulator
# ---------------------------------------------------------------------------

class RealisticSimulator:
    """
    Multi-scale simulator combining PK, immune cascade, organ damage,
    and long-term adaptation into a unified time series.
    """

    def __init__(self) -> None:
        self._cascade_engine   = BiologicalCascadeEngine()
        self._damage_engine    = OrganDamageEngine()
        self._adaptation_engine = LongTermAdaptationEngine()

    def simulate(
        self,
        drug_names: List[str],
        doses_mg: Dict[str, float],
        disease_name: str,
        disease_severity: float,
        patient_age: float = 65.0,
        patient_weight_kg: float = 70.0,
        patient_sex: int = 1,           # 1 = male, 0 = female
        kidney_function: float = 1.0,
        liver_function: float = 1.0,
        dosing_frequency: str = "daily",
        duration_days: float = 30.0,
        cyp_clearance_multipliers: Optional[Dict[str, float]] = None,
    ) -> LongTermSimulationResult:
        """
        Run the complete multi-scale realistic simulation.

        Parameters
        ----------
        drug_names        : list of drug names to simulate simultaneously
        doses_mg          : {drug: dose_mg}
        disease_name      : disease for initial conditions
        disease_severity  : 0–1 scaling of disease initial conditions
        duration_days     : total simulation length in days (can be 0.5 to 1825 = 5 years)

        Returns
        -------
        LongTermSimulationResult with full time-series and event log
        """
        duration_h = duration_days * 24.0

        # Build multi-scale time grid
        time_h = self._build_time_grid(duration_h)

        # ------------------------------------------------------------------
        # 1. PK: compute plasma concentration time series using the
        #    mechanistic engine (imported lazily to avoid circular imports)
        # ------------------------------------------------------------------
        plasma_conc = self._compute_pk(
            drug_names, doses_mg, time_h, dosing_frequency,
            kidney_function, liver_function, patient_weight_kg, patient_age,
            disease_name, disease_severity,
            cyp_clearance_multipliers=cyp_clearance_multipliers,
        )

        # ------------------------------------------------------------------
        # 2. Immune cascade: runs the full biological ODE system
        # ------------------------------------------------------------------
        # Compute aggregate drug effect on the cascade
        cascade_inputs = [
            build_cascade_input_for_drug(
                d,
                float(np.mean(plasma_conc.get(d, np.zeros(1)))),
                _get_therapeutic_window_hi(d),
            )
            for d in drug_names
        ]
        combined_cascade = _merge_cascade_inputs(cascade_inputs)

        cascade_result: CascadeResult = self._cascade_engine.simulate(
            duration_hours   = duration_h,
            disease_name     = disease_name,
            disease_severity = disease_severity,
            cascade_input    = combined_cascade,
            patient_age      = patient_age,
            time_resolution_h= max(0.5, duration_h / 4000.0),
        )

        # Align cascade to our time grid
        crp     = np.interp(time_h, cascade_result.time_h, cascade_result.crp_series)
        il6     = np.interp(time_h, cascade_result.time_h, cascade_result.il6_series)
        tnfa    = np.interp(time_h, cascade_result.time_h, cascade_result.tnfa_series)
        wbc     = np.interp(time_h, cascade_result.time_h, cascade_result.wbc_series)
        cortisol= np.interp(time_h, cascade_result.time_h, cascade_result.cortisol_series)
        glucose = np.interp(time_h, cascade_result.time_h, cascade_result.glucose_series)
        tsh_arr = np.interp(time_h, cascade_result.time_h, cascade_result.tsh_series)
        platelets=np.interp(time_h, cascade_result.time_h, cascade_result.platelets_series)
        pathogen= np.interp(time_h, cascade_result.time_h, cascade_result.pathogen_series)
        inflam  = np.interp(time_h, cascade_result.time_h, cascade_result.inflammation_index_series)

        # ------------------------------------------------------------------
        # 3. Organ damage & biomarkers
        # ------------------------------------------------------------------
        try:
            biomarkers = self._damage_engine.simulate(
                drug_names         = drug_names,
                plasma_conc_series = plasma_conc,
                time_h             = time_h,
                disease_name       = disease_name,
                disease_severity   = disease_severity,
                patient_weight_kg  = patient_weight_kg,
                patient_age        = patient_age,
                kidney_function    = kidney_function,
                liver_function     = liver_function,
            )
        except Exception as exc:
            logger.warning("OrganDamageEngine failed: %s", exc)
            biomarkers = {n: np.full(len(time_h), BIOMARKER_BASELINE[n][0]) for n in BM_NAMES}

        # ------------------------------------------------------------------
        # 4. Long-term adaptation
        # ------------------------------------------------------------------
        adaptations = self._adaptation_engine.simulate(
            duration_hours     = duration_h,
            drug_names         = drug_names,
            plasma_conc_series = plasma_conc,
            time_h             = time_h,
            disease_name       = disease_name,
            disease_severity   = disease_severity,
            patient_age        = patient_age,
            time_resolution_h  = max(1.0, duration_h / 2000.0),
        )

        # ------------------------------------------------------------------
        # 5. Derived metrics
        # ------------------------------------------------------------------
        cl_mult = {d: compute_clearance_multiplier(adaptations, d) for d in drug_names}
        wd_risk  = {d: compute_withdrawal_risk(adaptations, d)    for d in drug_names}

        has_ssri = any(d in ("Sertraline",) for d in drug_names)
        antidep  = compute_antidepressant_response(adaptations) if has_ssri else None

        # Disease severity trajectory (from cascade inflammation index)
        disease_sev_series = disease_severity * (1.0 - inflam * 0.4) + inflam * 0.2
        disease_sev_series = np.clip(disease_sev_series, 0.0, 1.0)

        # ------------------------------------------------------------------
        # 6. Clinical event detection
        # ------------------------------------------------------------------
        events = self._detect_events(
            time_h, drug_names, disease_name,
            plasma_conc, crp, il6, wbc, biomarkers, adaptations, pathogen
        )

        # ------------------------------------------------------------------
        # 7. Summary statistics
        # ------------------------------------------------------------------
        infection_cleared_h = self._find_infection_clearance(pathogen, time_h)
        cough_onset_h       = self._find_cough_onset(biomarkers, drug_names, time_h)
        myopathy_onset_h    = self._find_myopathy_onset(biomarkers, drug_names, time_h)

        bmd_init  = float(biomarkers["BMD"][0])
        bmd_final = float(biomarkers["BMD"][-1])
        bone_loss_pct_per_y = 0.0
        if bmd_init > 0 and duration_days > 7:
            bone_loss_pct_per_y = (bmd_init - bmd_final) / bmd_init * 100.0 * (365.0 / duration_days)

        ldl_init  = float(adaptations["LDL"][0])
        ldl_final = float(adaptations["LDL"][-1])
        ldl_change_pct = (ldl_final - ldl_init) / max(ldl_init, 0.01) * 100.0

        b12_init  = float(biomarkers["B12_LEVEL"][0])
        b12_final = float(biomarkers["B12_LEVEL"][-1])
        b12_depletion_pct = (b12_init - b12_final) / max(b12_init, 0.01) * 100.0

        time_labels = [_format_time_label(t) for t in time_h]

        return LongTermSimulationResult(
            time_h              = time_h,
            time_labels         = time_labels,
            plasma_conc         = plasma_conc,
            crp_series          = crp,
            il6_series          = il6,
            tnfa_series         = tnfa,
            wbc_series          = wbc,
            cortisol_series     = cortisol,
            glucose_series      = glucose,
            tsh_series          = tsh_arr,
            platelet_series     = platelets,
            pathogen_series     = pathogen,
            inflammation_index  = inflam,
            biomarkers          = biomarkers,
            adaptations         = adaptations,
            antidepressant_response = antidep,
            clearance_multiplier= cl_mult,
            withdrawal_risk     = wd_risk,
            disease_severity_series = disease_sev_series,
            events              = events,
            peak_crp            = float(np.max(crp)),
            peak_wbc            = float(np.max(wbc)),
            infection_cleared_at_h = infection_cleared_h,
            cough_onset_h       = cough_onset_h,
            myopathy_onset_h    = myopathy_onset_h,
            bone_loss_pct_per_year = bone_loss_pct_per_y,
            ldl_change_pct      = ldl_change_pct,
            b12_depletion_pct   = b12_depletion_pct,
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_time_grid(self, duration_h: float) -> np.ndarray:
        """Build a non-uniform time grid: finer resolution early, coarser later."""
        segments = []
        if duration_h <= 72.0:
            segments.append(np.arange(0.0, min(duration_h, 72.0) + 0.5, 0.5))
        elif duration_h <= 720.0:       # up to 30 days
            segments.append(np.arange(0.0, 72.0, 0.5))
            segments.append(np.arange(72.0, min(duration_h, 720.0) + 2.0, 2.0))
        elif duration_h <= 4380.0:      # up to 6 months
            segments.append(np.arange(0.0, 72.0, 0.5))
            segments.append(np.arange(72.0, 720.0, 2.0))
            segments.append(np.arange(720.0, min(duration_h, 4380.0) + 12.0, 12.0))
        else:                           # > 6 months up to 5 years
            segments.append(np.arange(0.0, 72.0, 0.5))
            segments.append(np.arange(72.0, 720.0, 2.0))
            segments.append(np.arange(720.0, 4380.0, 12.0))
            segments.append(np.arange(4380.0, duration_h + 24.0, 24.0))

        combined = np.unique(np.concatenate(segments))
        return combined[combined <= duration_h + 1e-6]

    def _compute_pk(
        self,
        drug_names: List[str],
        doses_mg: Dict[str, float],
        time_h: np.ndarray,
        dosing_frequency: str,
        kidney_function: float,
        liver_function: float,
        weight_kg: float,
        age: float,
        disease_name: str,
        disease_severity: float,
        cyp_clearance_multipliers: Optional[Dict[str, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute plasma concentrations using the existing MechanisticPKPDEngine.
        Falls back to analytical 1-compartment model if engine unavailable.
        """
        try:
            from simulation.mechanistic_engine import MechanisticPKPDEngine
            engine = MechanisticPKPDEngine()
            results: Dict[str, np.ndarray] = {}

            # Import drug/patient dataclasses from server
            from app.server import (
                BiologicalParameters, DrugProperties, DiseaseState, _simulator
            )

            bio = BiologicalParameters(
                age=age, weight=weight_kg,
                kidney_function=kidney_function, liver_function=liver_function,
            )
            for d in drug_names:
                dp_base = _simulator.drug_database.get(d)
                if dp_base is None:
                    results[d] = np.zeros(len(time_h))
                    continue
                import copy
                dp = copy.deepcopy(dp_base)
                dp.dose_mg = doses_mg.get(d, dp.dose_mg)
                dp.dosing_frequency = dosing_frequency

                # Apply CYP phenotype clearance adjustment if provided
                if cyp_clearance_multipliers and d in cyp_clearance_multipliers:
                    dp.clearance = float(np.clip(
                        dp.clearance * cyp_clearance_multipliers[d], 0.01, 1000.0
                    ))

                ds_base = _simulator.disease_database.get(disease_name)
                ds = copy.deepcopy(ds_base) if ds_base else DiseaseState(name=disease_name, severity=disease_severity)
                ds.severity = disease_severity

                eng_result = engine.simulate_drug(dp, bio, [ds], time_h, apply_ml_correction=True)
                results[d] = np.clip(eng_result.plasma_conc, 0.0, None)
            return results

        except Exception as exc:
            logger.warning("MechanisticPKPDEngine failed in RealisticSimulator: %s — using analytical PK", exc)
            return self._analytical_pk(drug_names, doses_mg, time_h, dosing_frequency,
                                       kidney_function, liver_function, weight_kg)

    def _analytical_pk(
        self,
        drug_names: List[str],
        doses_mg: Dict[str, float],
        time_h: np.ndarray,
        dosing_frequency: str,
        kidney_function: float,
        liver_function: float,
        weight_kg: float,
    ) -> Dict[str, np.ndarray]:
        """Fallback 1-compartment analytical PK model."""
        interval_map = {"daily": 24.0, "bid": 12.0, "tid": 8.0, "qid": 6.0}
        interval = interval_map.get(dosing_frequency, 24.0)

        # Build fallback PK DB from data file (via DataLoader)
        try:
            import sys as _sys
            import os as _os
            _here = _os.path.dirname(_os.path.abspath(__file__))
            _project = _os.path.dirname(_os.path.dirname(_here))
            if _project not in _sys.path:
                _sys.path.insert(0, _project)
                _sys.path.insert(0, _os.path.join(_project, "src"))
            from data.data_loader import get_data_loader as _gdl
            _FALLBACK_DB: Dict[str, Dict] = {
                r["drug_name"]: {
                    "hl": float(r["half_life"]),
                    "ba": float(r["bioavailability"]),
                    "vd": float(r["volume_distribution"]),
                }
                for r in _gdl().load_drugs()
            }
        except Exception:
            _FALLBACK_DB = {
                "Atorvastatin": {"hl": 14.0, "ba": 0.14, "vd": 381.0},
                "Metformin":    {"hl":  6.0, "ba": 0.55, "vd": 126.0},
                "Sertraline":   {"hl": 26.0, "ba": 0.44, "vd": 273.0},
                "Lisinopril":   {"hl": 12.0, "ba": 0.25, "vd": 120.0},
                "Prednisone":   {"hl":  3.5, "ba": 0.70, "vd":  35.0},
                "Lorazepam":    {"hl": 14.0, "ba": 0.85, "vd": 121.0},
            }
        _DEFAULT = {"hl": 8.0, "ba": 0.70, "vd": 50.0}

        results: Dict[str, np.ndarray] = {}
        for d in drug_names:
            params = _FALLBACK_DB.get(d, _DEFAULT)
            hl = params["hl"]
            ba = params["ba"]
            vd = params["vd"] * (weight_kg / 70.0)
            ke = 0.693 / max(hl, 0.1) * kidney_function * liver_function

            dose = doses_mg.get(d, 100.0)
            dose_times = np.arange(0.0, time_h[-1] + 1e-6, interval)
            conc = np.zeros(len(time_h))
            for t_dose in dose_times:
                dt = time_h - t_dose
                mask = dt >= 0.0
                if np.any(mask):
                    conc[mask] += (dose * ba / max(vd, 1e-6)) * np.exp(-ke * dt[mask])
            results[d] = np.clip(conc, 0.0, None)
        return results

    def _detect_events(
        self,
        time_h: np.ndarray,
        drug_names: List[str],
        disease_name: str,
        plasma_conc: Dict[str, np.ndarray],
        crp: np.ndarray,
        il6: np.ndarray,
        wbc: np.ndarray,
        biomarkers: Dict[str, np.ndarray],
        adaptations: Dict[str, np.ndarray],
        pathogen: np.ndarray,
    ) -> List[ClinicalEvent]:
        events: List[ClinicalEvent] = []

        def _first_crossing(arr: np.ndarray, threshold: float, above: bool = True) -> Optional[float]:
            """Return time of first threshold crossing."""
            mask = (arr > threshold) if above else (arr < threshold)
            idx = np.argmax(mask)
            return float(time_h[idx]) if mask[idx] else None

        # ── CRP elevation ────────────────────────────────────────────────────
        t = _first_crossing(crp, 10.0)
        if t is not None:
            events.append(ClinicalEvent(
                time_h=t, severity="info", organ_system="immune",
                title="CRP elevated (>10 mg/L)",
                description=f"C-reactive protein rose above 10 mg/L at {_format_time_label(t)}, indicating active inflammation. "
                            f"Peak: {float(np.max(crp)):.1f} mg/L",
                biomarker_name="CRP", biomarker_value=float(crp[np.searchsorted(time_h, t)]), biomarker_unit="mg/L",
            ))
        t = _first_crossing(crp, 100.0)
        if t is not None:
            events.append(ClinicalEvent(
                time_h=t, severity="warning", organ_system="immune",
                title="Severe inflammation — CRP >100 mg/L",
                description=f"CRP exceeded 100 mg/L at {_format_time_label(t)}. "
                            "Indicates severe infection, tissue damage, or acute flare. "
                            "Urgent clinical assessment recommended.",
                biomarker_name="CRP", biomarker_value=float(crp[np.searchsorted(time_h, t)]), biomarker_unit="mg/L",
            ))

        # ── WBC changes ──────────────────────────────────────────────────────
        t = _first_crossing(wbc, 11.0)
        if t is not None:
            events.append(ClinicalEvent(
                time_h=t, severity="info", organ_system="immune",
                title="Leukocytosis (WBC >11 ×10⁹/L)",
                description=f"White cell count rose above 11 ×10⁹/L at {_format_time_label(t)}, "
                            "reflecting active immune mobilisation against infection or inflammation.",
                biomarker_name="WBC", biomarker_value=float(wbc[np.searchsorted(time_h, t)]), biomarker_unit="×10⁹/L",
            ))
        t = _first_crossing(wbc, 4.0, above=False)
        if t is not None:
            events.append(ClinicalEvent(
                time_h=t, severity="warning", organ_system="immune",
                title="Leukopenia (WBC <4 ×10⁹/L)",
                description=f"White cell count fell below 4 ×10⁹/L at {_format_time_label(t)}. "
                            "May indicate myelosuppression, severe viral infection, or drug effect.",
                biomarker_name="WBC", biomarker_value=float(wbc[np.searchsorted(time_h, t)]), biomarker_unit="×10⁹/L",
            ))

        # ── Pathogen clearance ───────────────────────────────────────────────
        if disease_name == "Bacterial Infection":
            t_clear = _first_crossing(pathogen, 0.5, above=False)
            if t_clear is not None:
                events.append(ClinicalEvent(
                    time_h=t_clear, severity="info", organ_system="immune",
                    title="Infection resolving",
                    description=f"Pathogen load fell below detectable threshold at {_format_time_label(t_clear)}. "
                                "CRP and WBC should normalise over the following days.",
                    biomarker_name="Pathogen", biomarker_value=float(pathogen[np.searchsorted(time_h, t_clear)]),
                    biomarker_unit="log₁₀ CFU",
                ))

        # ── Liver enzyme elevation ───────────────────────────────────────────
        alt = biomarkers.get("ALT", np.zeros(len(time_h)))
        t = _first_crossing(alt, 56.0 * 3)  # 3× ULN
        if t is not None:
            events.append(ClinicalEvent(
                time_h=t, severity="warning", organ_system="liver",
                title="ALT >3× ULN — Hepatotoxicity signal",
                description=f"Alanine aminotransferase exceeded 3× upper limit of normal ({56*3} U/L) "
                            f"at {_format_time_label(t)}. "
                            "This indicates hepatocellular damage. Drug-induced liver injury should be considered.",
                biomarker_name="ALT", biomarker_value=float(alt[np.searchsorted(time_h, t)]), biomarker_unit="U/L",
            ))
        t = _first_crossing(alt, 56.0 * 10)  # 10× ULN
        if t is not None:
            events.append(ClinicalEvent(
                time_h=t, severity="critical", organ_system="liver",
                title="ALT >10× ULN — Severe hepatotoxicity",
                description=f"ALT exceeded 10× ULN at {_format_time_label(t)}. "
                            "Consistent with severe hepatocellular injury or acute liver failure. "
                            "Immediate drug cessation and specialist review required.",
                biomarker_name="ALT", biomarker_value=float(alt[np.searchsorted(time_h, t)]), biomarker_unit="U/L",
            ))

        # ── Renal function ───────────────────────────────────────────────────
        gfr_arr = biomarkers.get("GFR", np.full(len(time_h), 95.0))
        creat   = biomarkers.get("CREATININE", np.full(len(time_h), 85.0))
        t = _first_crossing(gfr_arr, 60.0, above=False)
        if t is not None:
            events.append(ClinicalEvent(
                time_h=t, severity="warning", organ_system="kidney",
                title="GFR <60 — Moderate kidney function impairment",
                description=f"Estimated GFR fell below 60 mL/min/1.73m² at {_format_time_label(t)}. "
                            "Renally cleared drugs may accumulate. "
                            "Consider dose adjustment for metformin, amoxicillin, ciprofloxacin.",
                biomarker_name="GFR", biomarker_value=float(gfr_arr[np.searchsorted(time_h, t)]), biomarker_unit="mL/min",
            ))
        t = _first_crossing(creat, 115 * 2)
        if t is not None:
            events.append(ClinicalEvent(
                time_h=t, severity="warning", organ_system="kidney",
                title="Creatinine doubled — Acute kidney injury signal",
                description=f"Serum creatinine exceeded 2× ULN at {_format_time_label(t)}, "
                            "consistent with acute kidney injury (AKI). "
                            "NSAID or other nephrotoxic drug effect should be assessed.",
                biomarker_name="CREATININE",
                biomarker_value=float(creat[np.searchsorted(time_h, t)]), biomarker_unit="µmol/L",
            ))

        # ── Statin myopathy ──────────────────────────────────────────────────
        ck_arr = biomarkers.get("CK", np.zeros(len(time_h)))
        t = _first_crossing(ck_arr, 174 * 5)   # 5× ULN
        if t is not None:
            events.append(ClinicalEvent(
                time_h=t, severity="warning", organ_system="muscle",
                title="CK >5× ULN — Statin myopathy",
                description=f"Creatine kinase exceeded 5× ULN ({174*5} U/L) at {_format_time_label(t)}. "
                            "Indicative of statin-induced myopathy due to mitochondrial CoQ10 depletion. "
                            "Myalgia is expected. Statin dose reduction or cessation recommended.",
                biomarker_name="CK", biomarker_value=float(ck_arr[np.searchsorted(time_h, t)]), biomarker_unit="U/L",
            ))
        t = _first_crossing(ck_arr, 174 * 40)   # 40× ULN (rhabdomyolysis)
        if t is not None:
            events.append(ClinicalEvent(
                time_h=t, severity="critical", organ_system="muscle",
                title="Rhabdomyolysis risk — CK >40× ULN",
                description=f"CK exceeded 40× ULN at {_format_time_label(t)}. "
                            "Myoglobinuria and acute kidney injury are imminent risks. "
                            "Emergency assessment and IV hydration required.",
                biomarker_name="CK", biomarker_value=float(ck_arr[np.searchsorted(time_h, t)]), biomarker_unit="U/L",
            ))

        # ── Bone density ─────────────────────────────────────────────────────
        bmd_arr = biomarkers.get("BMD", np.full(len(time_h), 1.0))
        bmd_baseline = float(bmd_arr[0])
        t = _first_crossing(bmd_arr, bmd_baseline * 0.90, above=False)
        if t is not None and "Prednisone" in drug_names:
            events.append(ClinicalEvent(
                time_h=t, severity="warning", organ_system="bone",
                title="Bone density reduced >10% — Osteoporosis risk",
                description=f"BMD fell more than 10% from baseline at {_format_time_label(t)} "
                            "due to corticosteroid-induced osteoblast suppression and increased bone resorption. "
                            "Bisphosphonate prophylaxis and calcium/vitamin D supplementation should be considered.",
                biomarker_name="BMD", biomarker_value=float(bmd_arr[np.searchsorted(time_h, t)]), biomarker_unit="g/cm²",
            ))

        # ── ACE inhibitor cough ───────────────────────────────────────────────
        bk_arr = biomarkers.get("BRADYKININ", np.zeros(len(time_h)))
        if "Lisinopril" in drug_names:
            t = _first_crossing(bk_arr, 0.13)   # ~2.6× baseline; cough at ≥3× is typical
            if t is not None:
                events.append(ClinicalEvent(
                    time_h=t, severity="info", organ_system="respiratory",
                    title="ACE inhibitor cough onset",
                    description=f"Bradykinin accumulation crossed the cough-sensitisation threshold at "
                                f"{_format_time_label(t)}. "
                                "The characteristic dry, tickling ACE inhibitor cough may develop. "
                                "Occurs in ~15-20% of patients (up to 40% in Asian populations). "
                                "Consider switching to an ARB if cough is intolerable.",
                    biomarker_name="BRADYKININ", biomarker_value=float(bk_arr[np.searchsorted(time_h, t)]),
                    biomarker_unit="normalised",
                ))

        # ── BZD tolerance / dependence ────────────────────────────────────────
        if "Lorazepam" in drug_names:
            bzd_tol = adaptations.get("BZD_TOLERANCE", np.zeros(len(time_h)))
            bzd_dep = adaptations.get("BZD_DEPEND", np.zeros(len(time_h)))
            t = _first_crossing(bzd_tol, 0.30)
            if t is not None:
                events.append(ClinicalEvent(
                    time_h=t, severity="warning", organ_system="central_nervous_system",
                    title="Benzodiazepine tolerance developing",
                    description=f"GABA-A receptor downregulation has produced significant benzodiazepine tolerance "
                                f"at {_format_time_label(t)}. "
                                "Anxiolytic efficacy is reduced. Dose escalation increases dependence risk. "
                                "Prescriber review and taper plan recommended.",
                    biomarker_name="BZD_TOLERANCE", biomarker_value=float(bzd_tol[np.searchsorted(time_h, t)]),
                ))
            t = _first_crossing(bzd_dep, 0.40)
            if t is not None:
                events.append(ClinicalEvent(
                    time_h=t, severity="critical", organ_system="central_nervous_system",
                    title="Benzodiazepine physical dependence established",
                    description=f"Physical dependence on lorazepam is established at {_format_time_label(t)}. "
                                "Abrupt cessation carries risk of severe withdrawal: rebound anxiety, "
                                "tremor, sweating, and seizures. Slow taper (>4-8 weeks) is required.",
                    biomarker_name="BZD_DEPEND", biomarker_value=float(bzd_dep[np.searchsorted(time_h, t)]),
                ))

        # ── SSRI antidepressant onset lag ─────────────────────────────────────
        if "Sertraline" in drug_names:
            ht1a = adaptations.get("HT1A_AUTO", np.ones(len(time_h)))
            t = _first_crossing(ht1a, 0.80, above=False)
            if t is not None:
                events.append(ClinicalEvent(
                    time_h=t, severity="info", organ_system="central_nervous_system",
                    title="SSRI: 5-HT1A autoreceptor desensitisation — therapeutic onset",
                    description=f"5-HT1A serotonergic autoreceptors have desensitised sufficiently at "
                                f"{_format_time_label(t)} to allow full serotonergic neurotransmission. "
                                "This marks the onset of antidepressant therapeutic benefit (typically 2-6 weeks). "
                                "Early side effects (nausea, insomnia) should be diminishing.",
                ))
            hippo = adaptations.get("HIPPO_NEURO", np.ones(len(time_h)) * 0.5)
            t = _first_crossing(hippo, 0.80)
            if t is not None:
                events.append(ClinicalEvent(
                    time_h=t, severity="info", organ_system="central_nervous_system",
                    title="SSRI: Hippocampal neuroplasticity — full antidepressant efficacy",
                    description=f"BDNF-driven hippocampal neurogenesis index has reached 80% at "
                                f"{_format_time_label(t)}. "
                                "This represents the full antidepressant efficacy of sertraline therapy, "
                                "typically achieved at 4-12 weeks. Mood stabilisation should be evident.",
                ))

        # ── Metformin B12 depletion ───────────────────────────────────────────
        if "Metformin" in drug_names:
            b12_arr = biomarkers.get("B12_LEVEL", np.full(len(time_h), 350.0))
            t = _first_crossing(b12_arr, 150.0, above=False)
            if t is not None:
                events.append(ClinicalEvent(
                    time_h=t, severity="warning", organ_system="haematological",
                    title="Vitamin B12 deficiency (Metformin-induced)",
                    description=f"Vitamin B12 fell below lower limit of normal (150 pmol/L) at "
                                f"{_format_time_label(t)}. "
                                "Metformin inhibits intrinsic factor-B12 complex absorption at the ileum. "
                                "Annual monitoring and supplementation (1000 µg/day oral) recommended.",
                    biomarker_name="B12_LEVEL", biomarker_value=float(b12_arr[np.searchsorted(time_h, t)]),
                    biomarker_unit="pmol/L",
                ))

        # ── LDL reduction milestone (statins) ─────────────────────────────────
        if "Atorvastatin" in drug_names:
            ldl = adaptations.get("LDL", np.full(len(time_h), 3.5))
            ldl_init = float(ldl[0])
            t = _first_crossing(ldl, ldl_init * 0.70, above=False)
            if t is not None:
                events.append(ClinicalEvent(
                    time_h=t, severity="info", organ_system="cardiovascular",
                    title="LDL-cholesterol reduced >30%",
                    description=f"LDL-cholesterol has fallen >30% from baseline at {_format_time_label(t)} "
                                "through statin-mediated upregulation of hepatic LDL receptors. "
                                "Cardiovascular risk reduction is establishing. "
                                f"Current LDL: {float(ldl[np.searchsorted(time_h, t)]):.2f} mmol/L",
                    biomarker_name="LDL", biomarker_value=float(ldl[np.searchsorted(time_h, t)]), biomarker_unit="mmol/L",
                ))

        # ── Hyperglycaemia (corticosteroids) ──────────────────────────────────
        if "Prednisone" in drug_names:
            gluc_arr = np.interp(time_h, time_h, biomarkers.get("GLUCOSE",
                        np.full(len(time_h), 5.0)))  # reuse cascade glucose
        else:
            gluc_arr = None

        # Sort events by time
        events.sort(key=lambda e: e.time_h)
        return events

    def _find_infection_clearance(self, pathogen: np.ndarray, time_h: np.ndarray) -> Optional[float]:
        mask = pathogen < 0.5
        idx = np.argmax(mask)
        return float(time_h[idx]) if mask[idx] else None

    def _find_cough_onset(
        self, biomarkers: Dict[str, np.ndarray], drug_names: List[str], time_h: np.ndarray
    ) -> Optional[float]:
        _ace_inhibitors = {"Lisinopril"}   # extend as more ACE-I are added to DB
        if not _ace_inhibitors.intersection(drug_names):
            return None
        bk = biomarkers.get("BRADYKININ", np.zeros(len(time_h)))
        mask = bk > 0.13   # ~2.6× baseline (0.05)
        idx = np.argmax(mask)
        return float(time_h[idx]) if mask[idx] else None

    def _find_myopathy_onset(
        self, biomarkers: Dict[str, np.ndarray], drug_names: List[str], time_h: np.ndarray
    ) -> Optional[float]:
        if "Atorvastatin" not in drug_names:
            return None
        ck = biomarkers.get("CK", np.zeros(len(time_h)))
        mask = ck > 174 * 5
        idx = np.argmax(mask)
        return float(time_h[idx]) if mask[idx] else None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _format_time_label(t_h: float) -> str:
    """Convert hours to human-readable label."""
    if t_h < 1.0:
        return f"{t_h*60:.0f} min"
    if t_h < 24.0:
        return f"{t_h:.1f} h"
    if t_h < 24.0 * 7:
        d = t_h / 24.0
        return f"Day {d:.1f}"
    if t_h < 24.0 * 30:
        w = t_h / (24.0 * 7)
        return f"Week {w:.1f}"
    if t_h < 24.0 * 365:
        m = t_h / (24.0 * 30.44)
        return f"Month {m:.1f}"
    y = t_h / (24.0 * 365.25)
    return f"Year {y:.2f}"


def _get_therapeutic_window_hi(drug_name: str) -> float:
    """Get upper therapeutic window from server drug database (with fallback)."""
    try:
        from app.server import _simulator
        dp = _simulator.drug_database.get(drug_name)
        return dp.therapeutic_window[1] if dp else 50.0
    except Exception:
        return 50.0


def _merge_cascade_inputs(inputs: List[CascadeInput]) -> CascadeInput:
    """Merge multiple CascadeInput objects (sum clipped to valid ranges)."""
    if not inputs:
        return CascadeInput()

    fields = {
        "antimicrobial_effect": 0.0,
        "anti_inflammatory_effect": 0.0,
        "immunosuppressive_effect": 0.0,
        "cortisol_supplement": 0.0,
        "thyroid_supplement": 0.0,
        "insulin_sensitizer": 0.0,
        "insulin_secretagogue": 0.0,
        "beta_blocker_intensity": 0.0,
        "ace_inhibition": 0.0,
        "serotonin_reuptake_block": 0.0,
        "il6_modulation": 0.0,
        "tnfa_modulation": 0.0,
    }
    for inp in inputs:
        for k in fields:
            fields[k] += getattr(inp, k, 0.0)

    # Clip 0–1 fields
    _clip_01 = {
        "antimicrobial_effect", "anti_inflammatory_effect", "immunosuppressive_effect",
        "cortisol_supplement", "thyroid_supplement", "insulin_sensitizer",
        "insulin_secretagogue", "beta_blocker_intensity", "ace_inhibition",
        "serotonin_reuptake_block",
    }
    for k in _clip_01:
        fields[k] = float(np.clip(fields[k], 0.0, 1.0))

    return CascadeInput(**fields)


# ---------------------------------------------------------------------------
# Singleton for the app server
# ---------------------------------------------------------------------------
_realistic_simulator: Optional[RealisticSimulator] = None


def get_realistic_simulator() -> RealisticSimulator:
    global _realistic_simulator
    if _realistic_simulator is None:
        _realistic_simulator = RealisticSimulator()
    return _realistic_simulator
