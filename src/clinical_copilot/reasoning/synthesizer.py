"""Clinical reasoning synthesizer that crafts plain-language explanations and evidence-grounded recommendations."""
from __future__ import annotations

from typing import List, Optional
from ..schemas import (
    AlertPriority,
    AlertRecommendation,
    PatientContext,
    RAGCitation,
    VitalReading,
)


class ClinicalReasoningSynthesizer:
    """Synthesizes patient context, vital trends, and retrieved clinical guidelines."""

    def explain(
        self,
        patient_id: str,
        current: VitalReading,
        syndrome: str,
        priority: AlertPriority,
        contributing_factors: list[str],
        news2: int,
        si: float,
        context: Optional[PatientContext] = None,
    ) -> str:
        """Produces a clear, plain-language clinical explanation of why the patient's state is concerning."""
        history_str = ", ".join(context.relevant_history) if context and context.relevant_history else "No prior history"
        meds_str = ", ".join(context.current_medications) if context and context.current_medications else "None"

        lines = [
            f"Patient {patient_id} demonstrates significant clinical deterioration classified as [{syndrome.upper().replace('_', ' ')}].",
            f"Current Vitals: HR {current.heart_rate:.0f} bpm, BP {current.systolic_bp:.0f}/{current.diastolic_bp:.0f} mmHg (MAP {current.map:.0f}), SpO2 {current.spo2:.0f}%, RR {current.respiratory_rate:.0f}/min.",
            f"Scoring Assessment: NEWS2 aggregate score = {news2} (Priority: {priority.value}), Shock Index = {si:.2f}.",
            f"Context: Baseline medical history ({history_str}), Medications ({meds_str}).",
            "Key Contributing Factors:",
        ]
        for f in contributing_factors:
            lines.append(f"  • {f}")

        if syndrome == "beta_blocked_sepsis":
            lines.append("Clinical Note: Tachycardia is blunted due to concurrent beta-blocker therapy. Do not rely on heart rate to rule out septic shock.")
        elif syndrome == "opioid_respiratory_depression":
            lines.append("Clinical Note: Progressive bradypnea and falling oxygenation suggest opioid toxicity or central respiratory drive depression.")
        elif syndrome == "masked_hypoxemia":
            lines.append("Clinical Note: Normal SpO2 is artificially maintained via high FiO2 while tachypnea indicates impending respiratory exhaustion.")

        return "\n".join(lines)

    def recommend(
        self,
        syndrome: str,
        priority: AlertPriority,
        citations: list[RAGCitation],
        context: Optional[PatientContext] = None,
    ) -> AlertRecommendation:
        """Formulates targeted clinical recommendations grounded in cited guidelines."""
        actions: list[str] = []
        rationale: str = ""

        if syndrome in {"sepsis_like", "beta_blocked_sepsis"}:
            actions = [
                "Urgent medical review by Rapid Response / Critical Care Team.",
                "Initiate Sepsis-3 resuscitation bundle: obtain serum lactate and blood cultures immediately.",
                "Administer intravenous broad-spectrum antimicrobials within 1 hour.",
                "Initiate 30 mL/kg IV crystalloid fluid challenge for hypotension (target MAP >= 65 mmHg).",
                "Continuous telemetry and arterial line monitoring if hypotension persists.",
            ]
            rationale = "Grounded in Sepsis-3 consensus and NICE NG51 guidelines for suspected severe infection and organ dysfunction."

        elif syndrome == "haemodynamic_shock":
            actions = [
                "Immediate bedside hemodynamic evaluation.",
                "Place patient in supine position with legs elevated (passive leg raise test).",
                "Administer IV fluid resuscitation (500 mL crystalloid bolus over 15 minutes).",
                "Evaluate for active hemorrhage, anaphylaxis, or acute cardiac dysfunction.",
                "Prepare vasopressor infusion (noradrenaline) if fluid refractory.",
            ]
            rationale = "Critical hypotension (Shock Index >= 0.9) reflects impending or manifest cardiovascular collapse."

        elif syndrome in {"respiratory_decline", "masked_hypoxemia"}:
            actions = [
                "Immediate airway and breathing evaluation; ensure patent airway.",
                "Titrate supplemental oxygen to maintain target SpO2 (88-92% if COPD, 94-98% otherwise).",
                "Perform urgent Arterial Blood Gas (ABG) analysis to assess PaO2, PaCO2, and pH.",
                "Obtain urgent portable chest radiograph.",
                "Prepare high-flow nasal cannula (HFNC) or non-invasive ventilation (NIV) if work of breathing persists.",
            ]
            rationale = "NEWS2 guidance indicates escalating respiratory failure requiring immediate oxygen titration and senior review."

        elif syndrome == "opioid_respiratory_depression":
            actions = [
                "Immediate airway assessment and physical stimulation.",
                "Stop any continuous opioid infusions or PCA pumps.",
                "Prepare and administer Naloxone (0.1 - 0.4 mg IV, titrate to respiratory rate >= 10).",
                "Provide bag-valve-mask assistance or supplemental oxygen as needed.",
                "Monitor continuously for renarcotization due to short half-life of naloxone.",
            ]
            rationale = "Severe bradypnea and falling oxygenation indicate life-threatening opioid-induced central respiratory failure."

        else:
            actions = [
                "Repeat full vital signs in 15 minutes.",
                "Perform bedside clinical assessment and review fluid balance.",
                "Escalate to attending physician if NEWS2 remains elevated.",
            ]
            rationale = "Standard early warning monitoring based on Royal College of Physicians NEWS2 protocol."

        return AlertRecommendation(
            actions=actions,
            rationale=rationale,
            citations=citations,
        )
