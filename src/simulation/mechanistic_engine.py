"""
Mechanistic PK/PD Engine
========================

Hybrid engine replacing the simple monoexponential approximation with genuinely
grounded pharmacokinetic and pharmacodynamic science.

Architecture (sequential pipeline):
  1. PBPK-grade PK layer  →  plasma C(t) + organ-resolved concentrations
       • Proper 1-compartment oral absorption via first-order gut input
       • Multiple dosing by superposition (valid for linear PK)
       • Tissue:plasma partitioning via Poulin-Theil approximation (logP-based)
       • Disease / organ-function PK modifications

  2. Receptor binding layer  →  free-drug receptor occupancy R(t)
       • Equilibrium Hill model: R = Cfree / (EC50 + Cfree)
       • EC50 derived from drug's therapeutic window

  3. Signal transduction ODE  →  net PD response E(t)
       • dR/dt  = k_on * occ*(1-R) - k_off*R   (response build-up)
       • dA/dt  = k_adapt*(R-A)                  (adaptation / tolerance)
       • net_E  = R - k_tol*A  (attenuates with prolonged exposure)

  4. Disease-modified PD sensitivity  →  final effect with disease amplification
  5. Dynamic disease progression  →  severity trajectory over time

No external API calls. No torch required for basic mechanistic simulation.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EngineResult:
    """Full simulation output from MechanisticPKPDEngine.simulate_drug()."""

    time: np.ndarray                        # h
    plasma_conc: np.ndarray                 # mg/L  (= µg/mL)
    organ_conc: Dict[str, np.ndarray]       # mg/L per organ
    receptor_occupancy: np.ndarray          # 0–1  (fraction of receptors bound)
    pd_response: np.ndarray                 # 0–1  (normalised net pharmacological effect)
    disease_severity: np.ndarray            # 0–1  (trajectory of first active disease)
    pk_stats: Dict[str, float]              # Cmax, Tmax, AUClast, t½, Ctrough, Css_avg
    organ_conc_at_end: Dict[str, float] = field(default_factory=dict)  # snapshot at t_final

    # Optional: ML residual correction metadata
    ml_corrected: bool = False
    ml_pk_mean: Optional[np.ndarray] = None
    ml_pk_upper: Optional[np.ndarray] = None
    ml_pk_lower: Optional[np.ndarray] = None

    # Optional: GNN organ-resolved PD predictions (Phase 5)
    gnn_available: bool = False
    gnn_organ_pd: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MechanisticPKPDEngine:
    """
    Hybrid mechanistic PK/PD simulation engine.

    Designed to be instantiated once and reused across many simulations.
    Thread-safe for read operations (no mutable state after __init__).
    """

    # ── Organ compositions for Poulin-Theil tissue:plasma partitioning ──────
    #   (water fraction, lipid fraction, protein fraction)  — ICRP 2002 data
    _ORGAN_COMPOSITIONS: Dict[str, Tuple[float, float, float]] = {
        "brain":   (0.77, 0.11, 0.08),
        "heart":   (0.76, 0.07, 0.17),
        "liver":   (0.71, 0.05, 0.18),
        "kidney":  (0.73, 0.05, 0.15),
        "lung":    (0.77, 0.03, 0.16),
        "muscle":  (0.76, 0.03, 0.16),
        "fat":     (0.17, 0.79, 0.02),
        "skin":    (0.71, 0.06, 0.12),
        "gut":     (0.74, 0.04, 0.35),
        "spleen":  (0.77, 0.02, 0.14),
    }

    # ── Disease-specific PK/PD modifiers ────────────────────────────────────
    #   Applied as fractional multipliers scaled by disease severity (0–1).
    #   At severity = 1.0 → full modifier value.
    #   At severity = 0.0 → modifier = 1.0  (no effect).
    _DISEASE_PK_MAP: Dict[str, Dict[str, float]] = {
        "Chronic Kidney Disease":        {"clearance_factor": 0.50, "vd_factor": 1.25, "pd_sensitivity": 1.25},
        "Heart Failure (Mild)":          {"clearance_factor": 0.85, "vd_factor": 1.10, "pd_sensitivity": 1.15},
        "Heart Failure (Moderate)":      {"clearance_factor": 0.70, "vd_factor": 1.20, "pd_sensitivity": 1.20},
        "Heart Failure (Severe)":        {"clearance_factor": 0.55, "vd_factor": 1.35, "pd_sensitivity": 1.30},
        "Type 2 Diabetes (Mild)":        {"clearance_factor": 0.90, "vd_factor": 1.05, "pd_sensitivity": 1.10},
        "Type 2 Diabetes (Moderate)":    {"clearance_factor": 0.80, "vd_factor": 1.10, "pd_sensitivity": 1.15},
        "Type 2 Diabetes (Severe)":      {"clearance_factor": 0.65, "vd_factor": 1.20, "pd_sensitivity": 1.25},
        "Hypertension":                  {"clearance_factor": 0.92, "vd_factor": 1.05, "pd_sensitivity": 1.10},
        "COPD (Moderate)":               {"clearance_factor": 0.88, "vd_factor": 1.10, "pd_sensitivity": 1.12},
        "COPD (Severe)":                 {"clearance_factor": 0.72, "vd_factor": 1.18, "pd_sensitivity": 1.22},
        "Arthritis":                     {"clearance_factor": 0.95, "vd_factor": 1.02, "pd_sensitivity": 1.05},
        "Depression":                    {"clearance_factor": 0.92, "vd_factor": 1.05, "pd_sensitivity": 1.10},
        "Anxiety Disorder":              {"clearance_factor": 0.92, "vd_factor": 1.05, "pd_sensitivity": 1.10},
        "Hypothyroidism":                {"clearance_factor": 0.75, "vd_factor": 1.15, "pd_sensitivity": 1.20},
        "Bacterial Infection":           {"clearance_factor": 1.10, "vd_factor": 1.15, "pd_sensitivity": 0.90},
    }

    # Dosing interval (hours) by frequency label
    _INTERVAL: Dict[str, float] = {"daily": 24.0, "bid": 12.0, "tid": 8.0, "qid": 6.0}

    # ── Signal transduction parameters ──────────────────────────────────────
    #   These mimic typical G-protein coupled or nuclear receptor kinetics.
    _K_ON    = 0.40   # 1/h  response build-up (half-max in ~2.5 h)
    _K_OFF   = 0.08   # 1/h  response decay on drug removal
    _K_ADAPT = 0.04   # 1/h  adaptation / tolerance development
    _K_TOL   = 0.30   # –    fraction of adaptation that dampens response

    # ── Constructor ─────────────────────────────────────────────────────────

    def __init__(self) -> None:
        """Initialise engine and pre-load optional ML and GNN models."""
        self._ml_model = None
        self._gnn_model = None
        self._graph_builder = None
        self._try_load_ml_model()
        self._try_load_gnn_model()

    def _try_load_ml_model(self) -> None:
        """Silently attempt to load the staged PK/PD checkpoint for residual correction."""
        import os, sys
        base = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints")
        )
        staged_ckpt = os.path.join(base, "staged_pkpd_model.pth")
        simple_ckpt = os.path.join(base, "simple_mlp_model.pth")
        try:
            import torch
            sys_src = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
            if sys_src not in sys.path:
                sys.path.insert(0, sys_src)
            if os.path.exists(staged_ckpt):
                from models.pkpd_staged import StagedPKPDModel
                m = StagedPKPDModel(input_dim=32)
                m.load_state_dict(torch.load(staged_ckpt, map_location="cpu", weights_only=False))
                m.eval()
                self._ml_model = m
            elif os.path.exists(simple_ckpt):
                from models.mlp_architectures import SimpleMLP
                m = SimpleMLP(input_dim=32, hidden_dim=128, output_dim=2)
                state = torch.load(simple_ckpt, map_location="cpu", weights_only=False)
                m.load_state_dict(state)
                m.eval()
                self._ml_model = m
        except Exception as exc:
            logger.debug("ML residual model not loaded: %s", exc)

    def _try_load_gnn_model(self) -> None:
        """Silently attempt to load PKPDGNN checkpoint and build OrganGraphBuilder."""
        import os, sys
        base = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints")
        )
        gnn_ckpt = os.path.join(base, "gnn_pkpd.pth")
        try:
            import torch
            sys_src = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
            if sys_src not in sys.path:
                sys.path.insert(0, sys_src)
            from models.gnn_pkpd import PKPDGNN
            from simulation.organ_graph_builder import OrganGraphBuilder
            self._graph_builder = OrganGraphBuilder()
            if os.path.exists(gnn_ckpt):
                m = PKPDGNN(
                    organ_features_dim=16, drug_features_dim=8,
                    hidden_dim=64, num_layers=3, num_heads=4,
                )
                ckpt = torch.load(gnn_ckpt, map_location="cpu", weights_only=False)
                state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
                m.load_state_dict(state)
                m.eval()
                self._gnn_model = m
                logger.info("GNN model loaded from %s", gnn_ckpt)
            else:
                logger.debug("No GNN checkpoint found at %s; GNN inference disabled", gnn_ckpt)
        except Exception as exc:
            logger.debug("GNN model not loaded: %s", exc)

    # =========================================================================
    # Public API
    # =========================================================================    

    def simulate_drug(
        self,
        dash_drug,                 # dashboard DrugProperties dataclass
        dash_bio,                  # dashboard BiologicalParameters dataclass
        dash_diseases: list,       # list of dashboard DiseaseState instances
        time_points: np.ndarray,   # 1-D array of simulation time points (h)
        apply_ml_correction: bool = True,
    ) -> EngineResult:
        """
        Full mechanistic simulation for one drug across the given time horizon.

        Parameters
        ----------
        dash_drug : DrugProperties (from advanced_dashboard.py)
        dash_bio  : BiologicalParameters (from advanced_dashboard.py)
        dash_diseases : list of DiseaseState (from advanced_dashboard.py)
        time_points : np.ndarray  —  monotonically increasing time grid (h)
        apply_ml_correction : bool  —  whether to compute MC±σ ML overlay

        Returns
        -------
        EngineResult
        """
        time_points = np.asarray(time_points, dtype=float)

        # ── 1. Aggregate PK/PD modifiers from diseases + patient organ function
        pk_mods = self._aggregate_pk_modifiers(dash_diseases, dash_bio)

        # ── 2. Effective 1-compartment PK parameters
        cl_eff = float(dash_drug.clearance) * pk_mods["clearance_factor"]   # L/h
        vd_eff = float(dash_drug.volume_distribution) * pk_mods["vd_factor"] # L
        ke     = cl_eff / max(vd_eff, 1e-6)                                   # 1/h

        # ── 3. Full PK timecourse (plasma + organ estimates)
        plasma_conc, organ_concs = self._simulate_pk(
            dose        = max(float(dash_drug.dose_mg), 1e-6),
            bioavail    = float(dash_drug.bioavailability),
            ke          = ke,
            vd          = vd_eff,
            route       = getattr(dash_drug, "administration_route", "oral"),
            frequency   = getattr(dash_drug, "dosing_frequency", "daily"),
            logp        = getattr(dash_drug, "logp", 1.0),
            time        = time_points,
        )

        # ── 4. Free-drug receptor occupancy  (equilibrium Hill model)
        protein_b    = float(dash_drug.protein_binding)
        free_plasma  = plasma_conc * (1.0 - protein_b)
        ec50         = self._get_ec50(dash_drug)
        receptor_occ = free_plasma / (ec50 + free_plasma + 1e-12)

        # ── 5. Signal transduction → net PD response with delay + adaptation
        pd_response = self._signal_transduction_ode(receptor_occ, time_points)
        pd_response = np.clip(pd_response * pk_mods["pd_sensitivity"], 0.0, 1.0)

        # ── 6. Dynamic disease severity trajectory
        disease_sev = self._dynamic_disease_progression(
            dash_diseases, time_points, pd_response
        )

        # ── 7. PK statistics (NCA)
        pk_stats = self._compute_pk_stats(plasma_conc, time_points, dash_drug, cl_eff, vd_eff)

        # ── 8. Optional ML residual correction (MC Dropout uncertainty bands)
        ml_corrected, ml_mean, ml_upper, ml_lower = False, None, None, None
        if apply_ml_correction and self._ml_model is not None:
            ml_mean, ml_upper, ml_lower = self._ml_residual_correction(
                dash_drug, dash_bio, time_points, plasma_conc
            )
            ml_corrected = True

        organ_at_end = {
            org: float(arr[-1]) if len(arr) > 0 else 0.0
            for org, arr in organ_concs.items()
        }

        # ── 9. GNN organ-resolved PD inference (if checkpoint available) ──
        gnn_organ_pd: Optional[Dict[str, float]] = None
        gnn_available = False
        if self._gnn_model is not None and self._graph_builder is not None:
            try:
                tmax_idx = int(np.argmax(plasma_conc))
                organ_at_tmax = {
                    org: float(arr[tmax_idx])
                    for org, arr in organ_concs.items()
                }
                organ_at_tmax["blood"] = float(plasma_conc[tmax_idx])
                avg_disease_sev = float(np.mean(disease_sev)) if len(disease_sev) else 0.0
                graph_data = self._graph_builder.build_graph(
                    organ_conc_at_t=organ_at_tmax,
                    drug_mw=float(getattr(dash_drug, "molecular_weight", 300.0)),
                    drug_logp=float(getattr(dash_drug, "logp", 1.0)),
                    drug_ba=float(dash_drug.bioavailability),
                    drug_pb=float(dash_drug.protein_binding),
                    drug_cl=float(dash_drug.clearance),
                    drug_vd=float(dash_drug.volume_distribution),
                    drug_hl=float(dash_drug.half_life),
                    drug_dose=float(dash_drug.dose_mg),
                    disease_severity=avg_disease_sev,
                    patient_weight=float(getattr(dash_bio, "weight", 70.0)),
                )
                from simulation.organ_graph_builder import gnn_infer_organ_pd
                gnn_organ_pd = gnn_infer_organ_pd(self._gnn_model, graph_data)
                if gnn_organ_pd is not None:
                    gnn_available = True
            except Exception as _gnn_exc:
                logger.debug("GNN inference skipped: %s", _gnn_exc)

        return EngineResult(
            time              = time_points,
            plasma_conc       = plasma_conc,
            organ_conc        = organ_concs,
            receptor_occupancy= receptor_occ,
            pd_response       = pd_response,
            disease_severity  = disease_sev,
            pk_stats          = pk_stats,
            organ_conc_at_end = organ_at_end,
            ml_corrected      = ml_corrected,
            ml_pk_mean        = ml_mean,
            gnn_available     = gnn_available,
            gnn_organ_pd      = gnn_organ_pd,
            ml_pk_upper       = ml_upper,
            ml_pk_lower       = ml_lower,
        )

    # =========================================================================
    # PK Layer
    # =========================================================================

    def _simulate_pk(
        self,
        dose: float,
        bioavail: float,
        ke: float,
        vd: float,
        route: str,
        frequency: str,
        logp: float,
        time: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute plasma concentration timecourse using superposition of
        1-compartment analytical profiles (exact for linear PK systems).

        Oral route: Bateman equation  C(t) = (F·D·ka)/(Vd·(ka-ke))·(e^-ke·t − e^-ka·t)
        IV route  : Simple exponential C(t) = D/Vd · e^-ke·t

        Multiple doses are summed by superposition (dose at t_i contributes
        C(t - t_i) for all t ≥ t_i).
        """
        is_oral = route.lower() in ("oral", "po", "enteral")
        interval_h = self._INTERVAL.get(frequency, 24.0)

        # Absorption rate constant — drug-specific from logP
        # More lipophilic → faster membrane permeation → higher ka
        ka = float(np.clip(0.8 + 0.25 * logp, 0.3, 4.0))

        # Guard against ka ≈ ke (avoid division by zero in Bateman)
        if is_oral and abs(ka - ke) < 1e-4:
            ka += 1e-3

        absorbed_dose = dose * bioavail  # mg

        # Dose administration times within simulation window
        dose_times = np.arange(0.0, time[-1] + 1e-6, interval_h)

        plasma_total = np.zeros(len(time), dtype=float)
        for t_dose in dose_times:
            dt = time - t_dose
            mask = dt >= 0.0
            if not np.any(mask):
                continue
            dt_m = dt[mask]

            if is_oral and ka > ke:
                contrib = (absorbed_dose * ka) / (vd * (ka - ke)) * (
                    np.exp(-ke * dt_m) - np.exp(-ka * dt_m)
                )
            else:
                # IV bolus or fallback
                contrib = (absorbed_dose / vd) * np.exp(-ke * dt_m)

            plasma_total[mask] += contrib

        plasma_total = np.clip(plasma_total, 0.0, None)
        organ_concs  = self._estimate_organ_concentrations(plasma_total, logp)

        return plasma_total, organ_concs

    def _estimate_organ_concentrations(
        self, plasma: np.ndarray, logp: float
    ) -> Dict[str, np.ndarray]:
        """
        Estimate organ-tissue concentrations from plasma via Poulin-Theil
        tissue:plasma partitioning.

            K_tp = f_water·1 + f_lipid·10^logP + f_protein·1

        Lipophilic drugs (logP > 2) accumulate heavily in fat and brain.
        Hydrophilic drugs (logP < 0) stay in water-rich tissues.
        Clamp Ktp to [0.05, 100] to prevent physiologically impossible values.
        """
        k_lipid = float(10 ** logp)
        organ_concs = {}
        for organ, (fw, fl, fp) in self._ORGAN_COMPOSITIONS.items():
            ktp = fw * 1.0 + fl * k_lipid + fp * 1.0
            ktp = float(np.clip(ktp, 0.05, 100.0))
            organ_concs[organ] = plasma * ktp
        organ_concs["blood"] = plasma.copy()
        return organ_concs

    # =========================================================================
    # PD Layer
    # =========================================================================

    def _get_ec50(self, drug) -> float:
        """
        Estimate EC50 (mg/L) as geometric mean of a drug's therapeutic window.

        Therapeutic window: (C_min_effective, C_max_tolerated) in mg/L.
        EC50 ≈ sqrt(lower × upper) is a standard pharmacological approximation
        (the concentration producing 50% of maximal effect sits in the middle of
        the therapeutic range on a log-scale).
        """
        try:
            tw  = drug.therapeutic_window
            ec50 = float(np.sqrt(max(float(tw[0]), 0.001) * max(float(tw[1]), 0.001)))
        except Exception:
            ec50 = 10.0
        return max(ec50, 0.001)

    def _signal_transduction_ode(
        self, occupancy: np.ndarray, time: np.ndarray
    ) -> np.ndarray:
        """
        First-order receptor–effector coupling with adaptation.

        ODEs:
            dR/dt = k_on · occ(t) · (1 − R) − k_off · R      (response build-up)
            dA/dt = k_adapt · (R − A)                           (tolerance/adaptation)

        Net effect = max(0, R − k_tol · A)

        This captures:
          • Delayed onset  — R reaches plateau after several half-times of k_on⁻¹
          • Drug removal   — R decays at k_off⁻¹ hours after drug cessation
          • Adaptation     — sustained high occupancy generates tolerance (A tracks R)
        """
        occ_func = interp1d(
            time, occupancy, kind="linear",
            fill_value="extrapolate", bounds_error=False
        )

        def odes(y: list, t: float) -> list:
            R, A = y
            occ  = float(np.clip(occ_func(t), 0.0, 1.0))
            dR   = self._K_ON * occ * (1.0 - R) - self._K_OFF * R
            dA   = self._K_ADAPT * (R - A)
            return [dR, dA]

        sol = odeint(odes, [0.0, 0.0], time, rtol=1e-5, atol=1e-7)
        response = np.clip(sol[:, 0] - self._K_TOL * sol[:, 1], 0.0, 1.0)
        return response

    # =========================================================================
    # Disease Layer
    # =========================================================================

    def _aggregate_pk_modifiers(
        self, dash_diseases: list, dash_bio
    ) -> Dict[str, float]:
        """
        Combine PK/PD modifiers from all active diseases and patient organ function.

        Organ function (kidney/liver from BiologicalParameters) modifies clearance
        physiologically:
          • Hepatic impairment → reduced metabolic clearance
          • Renal impairment  → reduced renal excretion

        Disease modifiers are severity-weighted: at severity=1.0 the tabulated
        modifier is fully applied; at severity=0.0 it is not applied at all.
        """
        kf   = float(dash_bio.kidney_function)   # 0–1
        lf   = float(dash_bio.liver_function)     # 0–1

        # Organ function → clearance reduction  (linear interpolation)
        clearance_factor = 0.5 * lf + 0.5 * kf   # combined metabolic + renal scaling
        # Clamp away from zero but keep full scaling intact
        clearance_factor = float(np.clip(clearance_factor, 0.05, 1.0))

        vd_factor      = 1.0
        pd_sensitivity = 1.0

        for disease in dash_diseases:
            name = str(getattr(disease, "name", ""))
            sev  = float(np.clip(getattr(disease, "severity", 0.5), 0.0, 1.0))

            mods = self._DISEASE_PK_MAP.get(name)
            if mods:
                # Severity-weighted modifier: lerp between 1.0 and tabulated value
                eff_cl  = 1.0 - (1.0 - mods["clearance_factor"]) * sev
                eff_vd  = 1.0 + (mods["vd_factor"] - 1.0) * sev
                eff_pd  = 1.0 + (mods["pd_sensitivity"] - 1.0) * sev
            else:
                # Generic: infer from affected organs
                affected = getattr(disease, "affected_organs", [])
                eff_cl  = 1.0
                if any(o in affected for o in ("kidneys", "kidney")):
                    eff_cl *= 1.0 - 0.35 * sev
                if "liver" in affected:
                    eff_cl *= 1.0 - 0.45 * sev
                eff_vd = 1.0 + 0.10 * sev
                eff_pd = 1.0 + 0.15 * sev

            clearance_factor *= float(np.clip(eff_cl, 0.05, 1.5))
            vd_factor        *= float(np.clip(eff_vd, 0.5,  3.0))
            pd_sensitivity   *= float(np.clip(eff_pd, 0.5,  3.0))

        return {
            "clearance_factor": float(np.clip(clearance_factor, 0.03, 1.5)),
            "vd_factor":        float(np.clip(vd_factor,        0.5,  4.0)),
            "pd_sensitivity":   float(np.clip(pd_sensitivity,   0.5,  4.0)),
        }

    def _dynamic_disease_progression(
        self,
        dash_diseases: list,
        time_points: np.ndarray,
        pd_response: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate disease severity trajectory during the simulation.

        Mechanics:
          - Severity increases each timestep at the disease's `progression_rate` (per hour)
          - A therapeutic drug response suppresses and partially reverses progression
          - Suppression scales with `pd_response` (0 = no drug effect, 1 = full effect)
          - Maximum drug-mediated reversal is capped at 40 % of the progression rate

        Returns the severity trajectory of the *first* active disease (or zeros).
        """
        if not dash_diseases:
            return np.zeros(len(time_points))

        disease = dash_diseases[0]
        sev     = float(np.clip(getattr(disease, "severity", 0.5), 0.0, 1.0))
        rate    = float(np.clip(getattr(disease, "progression_rate", 0.005), 0.0, 0.1))

        pd_func = interp1d(
            time_points, pd_response,
            kind="linear", fill_value="extrapolate", bounds_error=False
        )

        severity_t       = np.zeros(len(time_points))
        severity_t[0]    = sev
        dt               = float(time_points[1] - time_points[0]) if len(time_points) > 1 else 1.0

        for i in range(1, len(time_points)):
            drug_effect   = float(np.clip(pd_func(time_points[i]), 0.0, 1.0))
            natural_prog  = rate * dt
            drug_reversal = drug_effect * rate * 0.40 * dt
            sev           = float(np.clip(sev + natural_prog - drug_reversal, 0.0, 1.0))
            severity_t[i] = sev

        return severity_t

    # =========================================================================
    # ML Residual Correction
    # =========================================================================

    def _ml_residual_correction(
        self,
        dash_drug,
        dash_bio,
        time_points: np.ndarray,
        plasma_conc: np.ndarray,
        n_samples: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply MC Dropout to the loaded ML checkpoint to generate a
        mean ± 1σ prediction band overlaying the mechanistic PK curve.

        Returns (mean, upper, lower) arrays of same length as time_points.
        Model output is normalised (0–1 scale); scale back by Cmax of
        mechanistic curve for interpretability.
        """
        try:
            import torch

            model = self._ml_model

            # Enable Dropout for Monte Carlo sampling
            for m in model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.train()

            # Build feature vector for each time point
            features_list = []
            for t in time_points:
                feat = self._build_feature_vector(dash_drug, dash_bio, t)
                features_list.append(feat)
            features = torch.tensor(np.array(features_list), dtype=torch.float32)  # (T, 32)

            all_preds = []
            with torch.no_grad():
                for _ in range(n_samples):
                    preds = model(features).numpy()[:, 0]   # PK output, shape (T,)
                    all_preds.append(preds)

            model.eval()  # restore eval mode

            all_preds = np.array(all_preds)     # (n_samples, T)
            mean_pred  = all_preds.mean(axis=0)
            std_pred   = all_preds.std(axis=0)

            # Scale: model output is normalised; rescale relative to mechanistic Cmax
            cmax_mech   = float(np.max(plasma_conc)) if np.max(plasma_conc) > 0 else 1.0
            scale       = cmax_mech / max(float(np.max(mean_pred)), 1e-6)
            ml_mean     = mean_pred * scale
            ml_upper    = (mean_pred + std_pred) * scale
            ml_lower    = np.clip((mean_pred - std_pred) * scale, 0.0, None)

            return ml_mean, ml_upper, ml_lower

        except Exception as exc:
            logger.debug("ML residual correction failed: %s", exc)
            return None, None, None

    # =========================================================================
    # NCA PK Statistics
    # =========================================================================

    def _compute_pk_stats(
        self,
        plasma: np.ndarray,
        time: np.ndarray,
        drug,
        cl_eff: float,
        vd_eff: float,
    ) -> Dict[str, float]:
        """
        Non-compartmental analysis (NCA) of the concentration–time profile.

        Computed parameters:
          Cmax       — peak plasma concentration (mg/L)
          Tmax       — time to peak (h)
          AUClast    — AUC trapezoidal to last time point (mg·h/L)
          t½         — terminal elimination half-life (h) from log-linear regression
          Ctrough    — concentration at final time point (mg/L)
          Css_avg    — predicted average steady-state concentration at current dose (mg/L)
          Vd_eff     — effective volume of distribution used (L)
          CL_eff     — effective clearance used (L/h)
        """
        if len(plasma) == 0:
            return {}

        cmax_idx  = int(np.argmax(plasma))
        cmax      = float(plasma[cmax_idx])
        tmax      = float(time[cmax_idx])
        auc_last  = float(np.trapezoid(plasma, time))
        c_trough  = float(plasma[-1])

        # Terminal t½: log-linear regression on the elimination phase
        try:
            elim_plasma = plasma[cmax_idx:]
            elim_time   = time[cmax_idx:]
            threshold   = 0.05 * cmax if cmax > 0 else 1e-6
            pos_mask    = elim_plasma > threshold
            if np.sum(pos_mask) >= 3:
                slope, _ = np.polyfit(elim_time[pos_mask], np.log(elim_plasma[pos_mask]), 1)
                t_half   = float(-np.log(2.0) / slope) if slope < -1e-9 else float(drug.half_life)
            else:
                t_half = float(drug.half_life)
        except Exception:
            t_half = float(drug.half_life)

        # Steady-state average concentration: Css = F·D / (CL · tau)
        interval_h = self._INTERVAL.get(getattr(drug, "dosing_frequency", "daily"), 24.0)
        try:
            css_avg = float(drug.dose_mg * drug.bioavailability / (cl_eff * interval_h))
        except Exception:
            css_avg = auc_last / max(float(time[-1]), 1.0)

        return {
            "Cmax (mg/L)":      round(cmax, 4),
            "Tmax (h)":         round(tmax, 2),
            "AUClast (mg·h/L)": round(auc_last, 3),
            "t½ (h)":           round(t_half, 2),
            "Ctrough (mg/L)":   round(c_trough, 4),
            "Css_avg (mg/L)":   round(max(css_avg, 0.0), 4),
            "CL_eff (L/h)":     round(cl_eff, 3),
            "Vd_eff (L)":       round(vd_eff, 2),
        }

    # =========================================================================
    # Feature vector builder (for ML residual correction)
    # =========================================================================

    def _build_feature_vector(self, drug, bio, t: float) -> np.ndarray:
        """
        Build the 32-dimensional feature vector expected by trained checkpoints.

        Layout (must match scripts/train.py create_features()):
          [0:10]  Drug props
          [10:20] Organ/disease features  (liver defaults)
          [20:29] Patient features
          [29:32] Pre-computed PK
        """
        dose     = float(drug.dose_mg)
        F        = float(drug.bioavailability)
        cl       = float(drug.clearance)
        vd       = float(drug.volume_distribution)
        k_el     = cl / max(vd, 1e-6)
        c_max    = dose * F / max(vd, 1e-6)
        auc_24   = dose * F / max(cl, 1e-6)
        mw       = float(getattr(drug, "molecular_weight", 300.0))
        logp     = float(getattr(drug, "logp", 1.0))
        pb       = float(drug.protein_binding)
        hl       = float(drug.half_life)
        route    = getattr(drug, "administration_route", "oral")

        feat = np.array([
            # Drug (10)
            mw / 1000.0, logp / 10.0, F, pb, cl / 5.0, vd / 10.0,
            hl / 100.0, 1.0, 1.0 if route == "oral" else 0.0, dose / 1000.0,
            # Organ – liver defaults (10)
            1.575 / 50.0, 1500.0 / 2000.0, 1.16, 1.0,
            7.4 / 14.0, (37.5 - 35.0) / 5.0, 0.71, 0.05, 0.24, 0.0,
            # Patient (9)
            float(bio.age) / 100.0, float(bio.weight) / 200.0, float(bio.sex),
            0.0, 0.0, 0.0, 0.0,       # ethnicity one-hot (unknown)
            t / 24.0, 0.0,             # time + adverse event flag
            # PK computed (3)
            float(np.clip(k_el,        0.0, 5.0)),
            float(np.clip(c_max / 1000.0,  0.0, 10.0)),
            float(np.clip(auc_24 / 10000.0, 0.0, 10.0)),
        ], dtype=np.float32)

        return feat
