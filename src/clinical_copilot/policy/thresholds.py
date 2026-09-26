"""Clinical thresholds, early warning scores (NEWS2, qSOFA, Shock Index), and syndrome heuristics."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from ..schemas import PatientContext, VitalReading


def compute_news2(vitals: VitalReading, context: Optional[PatientContext] = None) -> tuple[int, dict[str, int]]:
    """
    Computes NEWS2 score based on Royal College of Physicians (2017) guidance.
    Accounts for COPD / hypercapnic respiratory failure using SpO2 Scale 2.
    """
    score = 0
    breakdown: dict[str, int] = {}

    # 1. Respiration Rate (breaths/min)
    # <=8: 3, 9-11: 1, 12-20: 0, 21-24: 2, >=25: 3
    rr = vitals.respiratory_rate
    if rr <= 8.5:
        s_rr = 3
    elif rr <= 11.5:
        s_rr = 1
    elif rr <= 20.5:
        s_rr = 0
    elif rr <= 24.5:
        s_rr = 2
    else:
        s_rr = 3
    score += s_rr
    breakdown["respiratory_rate"] = s_rr

    # 2. Oxygen Saturation (SpO2)
    # Scale 2 applies if patient has confirmed COPD or hypercapnic risk
    is_copd = False
    if context and ("COPD" in context.relevant_history or "copd" in [h.lower() for h in context.relevant_history]):
        is_copd = True

    spo2 = vitals.spo2
    if is_copd:
        # Scale 2: <=83: 3, 84-85: 2, 86-87: 1, 88-92 on air: 0, 93-94: 1, >=95: 2
        if spo2 <= 83.5:
            s_spo2 = 3
        elif spo2 <= 85.5:
            s_spo2 = 2
        elif spo2 <= 87.5:
            s_spo2 = 1
        elif spo2 <= 92.5:
            s_spo2 = 0
        elif spo2 <= 94.5:
            s_spo2 = 1 if vitals.fio2 > 0.21 else 0
        else:
            s_spo2 = 2 if vitals.fio2 > 0.21 else 0
    else:
        # Scale 1: <=91: 3, 92-93: 2, 94-95: 1, >=96: 0
        if spo2 <= 91.5:
            s_spo2 = 3
        elif spo2 <= 93.5:
            s_spo2 = 2
        elif spo2 <= 95.5:
            s_spo2 = 1
        else:
            s_spo2 = 0
    score += s_spo2
    breakdown["spo2"] = s_spo2

    # 3. Supplemental Oxygen
    # Any supplemental oxygen (FiO2 > 0.21): 2
    s_o2 = 2 if vitals.fio2 > 0.21 else 0
    score += s_o2
    breakdown["supplemental_o2"] = s_o2

    # 4. Systolic Blood Pressure (mmHg)
    # <=90: 3, 91-100: 2, 101-110: 1, 111-219: 0, >=220: 3
    sbp = vitals.systolic_bp
    if sbp <= 90.5:
        s_sbp = 3
    elif sbp <= 100.5:
        s_sbp = 2
    elif sbp <= 110.5:
        s_sbp = 1
    elif sbp <= 219.5:
        s_sbp = 0
    else:
        s_sbp = 3
    score += s_sbp
    breakdown["systolic_bp"] = s_sbp

    # 5. Heart Rate (beats/min)
    # <=40: 3, 41-50: 1, 51-90: 0, 91-110: 1, 111-130: 2, >=131: 3
    hr = vitals.heart_rate
    if hr <= 40.5:
        s_hr = 3
    elif hr <= 50.5:
        s_hr = 1
    elif hr <= 90.5:
        s_hr = 0
    elif hr <= 110.5:
        s_hr = 1
    elif hr <= 130.5:
        s_hr = 2
    else:
        s_hr = 3
    score += s_hr
    breakdown["heart_rate"] = s_hr

    return score, breakdown


def compute_qsofa(vitals: VitalReading, altered_mentation: bool = False) -> tuple[int, list[str]]:
    """
    Computes qSOFA score (Singer et al., JAMA 2016):
    - Respiratory Rate >= 22 /min (+1)
    - Systolic BP <= 100 mmHg (+1)
    - Altered mentation (+1)
    """
    score = 0
    criteria = []
    if vitals.respiratory_rate >= 21.5:
        score += 1
        criteria.append(f"Tachypnea (RR {vitals.respiratory_rate:.1f} >= 22)")
    if vitals.systolic_bp <= 100.5:
        score += 1
        criteria.append(f"Hypotension (SBP {vitals.systolic_bp:.1f} <= 100)")
    if altered_mentation:
        score += 1
        criteria.append("Altered mentation reported")
    return score, criteria


def compute_shock_index(vitals: VitalReading) -> float:
    """Shock Index = Heart Rate / Systolic Blood Pressure. Normal range: 0.5 - 0.7."""
    if vitals.systolic_bp > 0:
        return vitals.heart_rate / vitals.systolic_bp
    return 2.0


def identify_clinical_syndrome(
    current: VitalReading,
    history: list[VitalReading],
    context: Optional[PatientContext] = None,
) -> tuple[str, list[str], float]:
    """
    Identifies specific clinical deterioration archetypes:
    - haemodynamic_shock
    - respiratory_decline
    - beta_blocked_sepsis
    - opioid_respiratory_depression
    - masked_hypoxemia
    - sepsis_like
    - artifact_or_transient
    - transient_exertion
    - stable
    Returns (syndrome_name, contributing_factors, confidence)
    """
    factors: list[str] = []
    news2, _ = compute_news2(current, context)
    qsofa, q_crit = compute_qsofa(current)
    si = compute_shock_index(current)

    is_copd = False
    is_beta_blocked = False
    if context:
        is_copd = any("copd" in h.lower() for h in context.relevant_history)
        is_beta_blocked = any("beta blocker" in m.lower() for m in context.current_medications)

    # Calculate baseline differences if history exists
    # Baseline from context if provided, else initial history
    base_dict = context.baseline_vitals if (context and context.baseline_vitals) else {}
    ref_hr = base_dict.get("heart_rate", history[0].heart_rate if history else current.heart_rate)
    ref_sbp = base_dict.get("systolic_bp", history[0].systolic_bp if history else current.systolic_bp)
    ref_spo2 = base_dict.get("spo2", history[0].spo2 if history else current.spo2)
    ref_rr = base_dict.get("respiratory_rate", history[0].respiratory_rate if history else current.respiratory_rate)

    delta_hr = current.heart_rate - ref_hr
    delta_sbp = current.systolic_bp - ref_sbp
    delta_spo2 = current.spo2 - ref_spo2
    delta_rr = current.respiratory_rate - ref_rr

    # 1. Check for single reading artifact
    if current.quality == "suspect":
        factors.append("Telemetry monitor quality flag suspect: possible sensor disconnect or motion artifact")
        return "artifact_or_transient", factors, 0.98

    if history:
        prev = history[-1]
        # Isolated single-point sudden spike/drop that doesn't fit physiological trajectory
        if abs(current.spo2 - prev.spo2) > 8 or abs(current.heart_rate - prev.heart_rate) > 25:
            factors.append("Sudden non-physiological jump from preceding reading; filtering as artifact")
            return "artifact_or_transient", factors, 0.95

    # 2. Check for benign transient exertion / ambulation / anxiety
    # Blood pressure is preserved or elevated, oxygenation is normal, no shock
    if delta_hr > 10 and delta_rr > 3 and current.systolic_bp >= (ref_sbp - 3.0) and current.spo2 >= 94.0 and si < 0.85:
        factors.append(f"Tachycardia (+{delta_hr:.0f} bpm) and tachypnea (+{delta_rr:.0f}/min) with preserved BP and normal SpO2 (exertion pattern)")
        return "transient_exertion", factors, 0.90

    # 3. Opioid-Induced Respiratory Depression
    # Characterized by Bradypnea (falling RR), hypoxemia (falling SpO2), and blunted/low HR
    if current.respiratory_rate <= 11.5 and (current.spo2 <= 93.0 or delta_spo2 <= -4.0) and current.heart_rate <= 78.0:
        factors.append(f"Depressed respiration rate ({current.respiratory_rate:.1f}/min)")
        factors.append(f"Falling SpO2 ({current.spo2:.1f}%)")
        factors.append(f"Low or blunted heart rate ({current.heart_rate:.1f} bpm)")
        return "opioid_respiratory_depression", factors, 0.92

    # 4. Masked Hypoxemia
    # Elevated FiO2 used to prop up SpO2, but patient has persistent tachypnea & tachycardia
    if current.fio2 > 0.25 and current.respiratory_rate >= 21.0 and current.heart_rate >= 85.0:
        factors.append(f"Elevated supplemental oxygen (FiO2 {current.fio2*100:.0f}%) masking underlying hypoxemia")
        factors.append(f"Persistent tachypnea despite oxygen therapy (RR {current.respiratory_rate:.1f}/min)")
        factors.append(f"Tachycardia (HR {current.heart_rate:.1f} bpm)")
        return "masked_hypoxemia", factors, 0.90

    # 5. Beta-Blocked Sepsis / Deterioration
    # Patient on beta blockers: blood pressure drops, but HR is blunted/normal
    if is_beta_blocked and (current.systolic_bp <= 85.0 or ((current.systolic_bp <= 102.0 or delta_sbp <= -15.0) and current.respiratory_rate >= 20.0 and current.heart_rate <= 92.0)):
        factors.append("Patient prescribed beta-blocker: compensatory tachycardia blunted")
        factors.append(f"Hypotension (SBP {current.systolic_bp:.1f} mmHg, drop: {delta_sbp:+.1f} mmHg)")
        if current.respiratory_rate >= 20.0:
            factors.append(f"Tachypnea (RR {current.respiratory_rate:.1f}/min)")
        factors.extend(q_crit)
        return "beta_blocked_sepsis", factors, 0.92

    # 6. Hemodynamic Shock
    # Severe drop in SBP (or SBP <= 80), tachycardia, elevated Shock Index >= 0.88
    if (current.systolic_bp <= 80.0) or ((current.systolic_bp <= 96.0 or delta_sbp <= -18.0) and current.heart_rate >= 90.0 and si >= 0.88):
        factors.append(f"Hypotension (SBP {current.systolic_bp:.1f} mmHg, baseline drop: {delta_sbp:+.1f} mmHg)")
        if current.heart_rate >= 90.0:
            factors.append(f"Compensatory tachycardia (HR {current.heart_rate:.1f} bpm)")
        factors.append(f"Elevated Shock Index ({si:.2f})")
        factors.extend(q_crit)
        return "haemodynamic_shock", factors, 0.95

    # 7. Respiratory Decline
    # Falling SpO2 (accounting for COPD baseline), rising RR
    spo2_threshold = 86.0 if is_copd else 91.5
    if (current.spo2 <= spo2_threshold or delta_spo2 <= -5.0) and current.respiratory_rate >= 21.5:
        factors.append(f"Significant hypoxemia (SpO2 {current.spo2:.1f}% {'[COPD adjusted]' if is_copd else ''})")
        factors.append(f"Tachypnea (RR {current.respiratory_rate:.1f}/min)")
        if current.heart_rate >= 85.0:
            factors.append(f"Compensatory tachycardia (HR {current.heart_rate:.1f} bpm)")
        return "respiratory_decline", factors, 0.92

    # 8. Sepsis-like / Systemic Deterioration
    # Meets qSOFA >= 2 or NEWS2 >= 6 with multi-system involvement
    if qsofa >= 2 or (news2 >= 6 and (current.respiratory_rate >= 22.0 or current.systolic_bp <= 100.0 or current.heart_rate >= 100.0)):
        factors.extend(q_crit)
        factors.append(f"Elevated aggregate NEWS2 score: {news2}")
        if si >= 0.85:
            factors.append(f"Elevated Shock Index ({si:.2f})")
        return "sepsis_like", factors, 0.88

    # Normal / Stable
    if news2 <= 2 and si < 0.85:
        return "stable", ["Vitals within acceptable reference range"], 0.95

    return "indeterminate", [f"Mild abnormal parameters (NEWS2: {news2})"], 0.60
