"""Core Alert Policy Engine coordinating decisions (ALERT / DEFER / SUPPRESS), priorities, and thresholds."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional
from ..reasoning.evidence_retriever import EvidenceRetriever
from ..reasoning.synthesizer import ClinicalReasoningSynthesizer
from ..schemas import (
    AlertDecision,
    AlertEvaluationResult,
    AlertPriority,
    PatientContext,
    VitalReading,
)
from .cooldown import CooldownManager
from .thresholds import (
    compute_news2,
    compute_qsofa,
    compute_shock_index,
    identify_clinical_syndrome,
)


class AlertPolicyEngine:
    """
    Evaluates streaming patient vitals to determine:
    - ALERT: Immediate escalation warranted
    - DEFER: Ambiguous/mild signal requiring short observation window
    - SUPPRESS: Benign artifact, known baseline, or within cooldown
    """

    def __init__(
        self,
        cooldown_seconds: int = 1200,
        corpus_path: Optional[str] = None,
        deferral_window: int = 3,
    ) -> None:
        self.cooldown_manager = CooldownManager(cooldown_seconds=cooldown_seconds)
        self.evidence_retriever = EvidenceRetriever(corpus_path=corpus_path)
        self.synthesizer = ClinicalReasoningSynthesizer()
        self.deferral_window = deferral_window

        # Per-patient state
        # patient_id -> list of VitalReading
        self.patient_histories: dict[str, list[VitalReading]] = defaultdict(list)
        # patient_id -> PatientContext
        self.patient_contexts: dict[str, PatientContext] = {}
        # patient_id -> count of deferred readings
        self.deferred_counts: dict[str, int] = defaultdict(int)
        self.consecutive_deteriorating: dict[str, int] = defaultdict(int)

    def register_context(self, context: PatientContext) -> None:
        """Registers static EHR patient profile."""
        self.patient_contexts[context.patient_id] = context

    def evaluate_reading(
        self,
        reading: VitalReading,
        patient_id: Optional[str] = None,
    ) -> AlertEvaluationResult:
        """
        Evaluates a single vital reading for a patient in temporal context.
        """
        pid = patient_id or getattr(reading, "patient_id", "UNKNOWN")
        context = self.patient_contexts.get(pid)
        history = self.patient_histories[pid]

        news2, breakdown = compute_news2(reading, context)
        qsofa, q_crit = compute_qsofa(reading)
        si = compute_shock_index(reading)
        syndrome, factors, confidence = identify_clinical_syndrome(reading, history, context)

        # 1. Handle Known Artifact / Sensor Quality
        if syndrome == "artifact_or_transient" or reading.quality == "suspect":
            self.consecutive_deteriorating[pid] = 0
            history.append(reading)
            if len(history) > 30:
                history.pop(0)
            return AlertEvaluationResult(
                patient_id=pid,
                timestamp=reading.timestamp,
                sequence=reading.sequence,
                decision=AlertDecision.SUPPRESS,
                priority=None,
                news2_score=news2,
                qsofa_score=qsofa,
                shock_index=si,
                syndrome="artifact_suppressed",
                contributing_factors=factors or ["Sensor artifact filtered by quality detector"],
                explanation="Isolated sensor artifact or suspect telemetry suppressed to prevent alarm fatigue.",
                recommendation=None,
                is_artifact=True,
                cooldown_active=False,
                deduplicated=False,
                confidence=confidence,
            )

        # 2. Determine raw priority based on multi-parameter severity
        raw_priority: Optional[AlertPriority] = None
        is_named_deterioration = syndrome in {
            "haemodynamic_shock",
            "beta_blocked_sepsis",
            "respiratory_decline",
            "opioid_respiratory_depression",
            "masked_hypoxemia",
            "sepsis_like",
        }

        if news2 >= 8 or si >= 1.1 or reading.systolic_bp <= 82.0 or reading.spo2 <= 84.0:
            raw_priority = AlertPriority.CRITICAL
        elif is_named_deterioration and (news2 >= 5 or si >= 0.90 or qsofa >= 2 or reading.systolic_bp <= 95.0 or reading.spo2 <= 90.0):
            raw_priority = AlertPriority.HIGH
        elif is_named_deterioration or news2 >= 5:
            raw_priority = AlertPriority.MEDIUM
        elif news2 >= 3:
            raw_priority = AlertPriority.LOW

        # 3. Check for Normal / Stable State or non-specific low-score noise
        if raw_priority is None or syndrome == "stable" or (syndrome == "indeterminate" and news2 < 7):
            self.deferred_counts[pid] = 0
            self.consecutive_deteriorating[pid] = 0
            history.append(reading)
            if len(history) > 30:
                history.pop(0)
            if len(history) >= 5 and all(compute_news2(h, context)[0] <= 2 for h in history[-5:]):
                self.cooldown_manager.resolve_if_stable(pid)

            return AlertEvaluationResult(
                patient_id=pid,
                timestamp=reading.timestamp,
                sequence=reading.sequence,
                decision=AlertDecision.SUPPRESS,
                priority=None,
                news2_score=news2,
                qsofa_score=qsofa,
                shock_index=si,
                syndrome=syndrome if syndrome != "indeterminate" else "stable",
                contributing_factors=["Parameters within acceptable baseline limits"] if syndrome != "indeterminate" else factors,
                explanation="Patient is clinically stable. Physiological fluctuations within normal AR(1) noise bounds.",
                recommendation=None,
                is_artifact=False,
                cooldown_active=False,
                deduplicated=False,
                confidence=confidence,
            )

        # 4. Check for Transient Exertion / Ambulation
        if syndrome == "transient_exertion":
            self.deferred_counts[pid] = 0
            self.consecutive_deteriorating[pid] = 0
            history.append(reading)
            if len(history) > 30:
                history.pop(0)
            return AlertEvaluationResult(
                patient_id=pid,
                timestamp=reading.timestamp,
                sequence=reading.sequence,
                decision=AlertDecision.SUPPRESS,
                priority=None,
                news2_score=news2,
                qsofa_score=qsofa,
                shock_index=si,
                syndrome="transient_exertion",
                contributing_factors=factors,
                explanation="Transient elevation consistent with benign exertion or post-physiotherapy. Blood pressure preserved.",
                recommendation=None,
                is_artifact=False,
                cooldown_active=False,
                deduplicated=False,
                confidence=confidence,
            )

        # 5. Require at least 2 consecutive deteriorating readings to confirm a sustained trajectory
        # This suppresses isolated 1-minute AR(1) fluctuations and cuff artifacts.
        self.consecutive_deteriorating[pid] = self.consecutive_deteriorating.get(pid, 0) + 1

        if self.consecutive_deteriorating[pid] < 2:
            history.append(reading)
            if len(history) > 30:
                history.pop(0)
            return AlertEvaluationResult(
                patient_id=pid,
                timestamp=reading.timestamp,
                sequence=reading.sequence,
                decision=AlertDecision.DEFER,
                priority=raw_priority,
                news2_score=news2,
                qsofa_score=qsofa,
                shock_index=si,
                syndrome=syndrome,
                contributing_factors=factors,
                explanation=f"Initial parameter breach ({syndrome.replace('_', ' ')}). Deferring alert to confirm sustained trajectory on next reading.",
                recommendation=None,
                is_artifact=False,
                cooldown_active=False,
                deduplicated=False,
                confidence=0.75,
            )

        # If it reached here, an ALERT is warranted.
        # Now check Cooldown and Deduplication
        should_alert, is_escalation, reason, ep_id = self.cooldown_manager.check_alert(
            patient_id=pid,
            timestamp=reading.timestamp,
            priority=raw_priority,
            syndrome=syndrome,
        )

        history.append(reading)
        if len(history) > 30:
            history.pop(0)

        if not should_alert:
            # Suppressed by Cooldown or Deduplicated
            return AlertEvaluationResult(
                patient_id=pid,
                timestamp=reading.timestamp,
                sequence=reading.sequence,
                decision=AlertDecision.SUPPRESS,
                priority=raw_priority,
                news2_score=news2,
                qsofa_score=qsofa,
                shock_index=si,
                syndrome=syndrome,
                contributing_factors=factors,
                explanation=f"Alert suppressed: {reason}",
                recommendation=None,
                is_artifact=False,
                cooldown_active=True,
                deduplicated=True,
                episode_id=ep_id,
                confidence=confidence,
            )

        # ALERT decision approved!
        # Retrieve RAG clinical guidance and formulate explanation + recommendations
        citations = self.evidence_retriever.retrieve(syndrome=syndrome, keywords=factors)
        explanation = self.synthesizer.explain(
            patient_id=pid,
            current=reading,
            syndrome=syndrome,
            priority=raw_priority,
            contributing_factors=factors,
            news2=news2,
            si=si,
            context=context,
        )
        if is_escalation:
            explanation = f"⚠️ [CRITICAL ESCALATION OVERRIDE] {explanation}"

        recommendation = self.synthesizer.recommend(
            syndrome=syndrome,
            priority=raw_priority,
            citations=citations,
            context=context,
        )

        return AlertEvaluationResult(
            patient_id=pid,
            timestamp=reading.timestamp,
            sequence=reading.sequence,
            decision=AlertDecision.ALERT,
            priority=raw_priority,
            news2_score=news2,
            qsofa_score=qsofa,
            shock_index=si,
            syndrome=syndrome,
            contributing_factors=factors,
            explanation=explanation,
            recommendation=recommendation,
            is_artifact=False,
            cooldown_active=False,
            deduplicated=False,
            episode_id=ep_id,
            confidence=confidence,
        )
