"""Deduplication and Cooldown handling with Escalation Override support."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple
from ..schemas import AlertDecision, AlertPriority


@dataclass
class ActiveEpisode:
    patient_id: str
    episode_id: str
    syndrome: str
    initial_alert_time: str
    last_alert_time: str
    highest_priority: AlertPriority
    last_priority: AlertPriority
    alert_count: int = 1
    suppressed_count: int = 0
    clinician_action: Optional[str] = None
    resolved: bool = False


class CooldownManager:
    """
    Manages alert deduplication and cooldown windows per patient.
    Enforces the clinical principle that repetitive alerts on an evolving episode
    must be suppressed to prevent alarm fatigue, while critical priority escalations
    must immediately break cooldown.
    """

    def __init__(self, cooldown_seconds: int = 1200) -> None:
        """
        cooldown_seconds: Default 1200s (20 minutes). In simulation, time differences
        are parsed from event timestamps.
        """
        self.cooldown_seconds = cooldown_seconds
        # patient_id -> ActiveEpisode
        self.active_episodes: dict[str, ActiveEpisode] = {}
        self.priority_order = {
            AlertPriority.LOW: 1,
            AlertPriority.MEDIUM: 2,
            AlertPriority.HIGH: 3,
            AlertPriority.CRITICAL: 4,
        }

    def _parse_time(self, t_str: str) -> float:
        try:
            # Handles ISO format
            clean_str = t_str.replace("Z", "+00:00")
            return datetime.fromisoformat(clean_str).timestamp()
        except Exception:
            return 0.0

    def check_alert(
        self,
        patient_id: str,
        timestamp: str,
        priority: AlertPriority,
        syndrome: str,
        proposed_episode_id: Optional[str] = None,
    ) -> tuple[bool, bool, str, Optional[str]]:
        """
        Evaluates whether a proposed alert should proceed, be deduplicated, or be suppressed by cooldown.
        Returns:
            should_alert: bool (True if alert should fire, False if suppressed)
            is_escalation: bool (True if this alert broke cooldown due to escalation)
            reason: str (Diagnostic reason for the decision)
            active_episode_id: Optional[str]
        """
        t_current = self._parse_time(timestamp)
        ep = self.active_episodes.get(patient_id)

        if not ep or ep.resolved:
            # Brand new alert episode
            ep_id = proposed_episode_id or f"EP-{patient_id}-{int(t_current)}"
            self.active_episodes[patient_id] = ActiveEpisode(
                patient_id=patient_id,
                episode_id=ep_id,
                syndrome=syndrome,
                initial_alert_time=timestamp,
                last_alert_time=timestamp,
                highest_priority=priority,
                last_priority=priority,
                alert_count=1,
            )
            return True, False, "New alert episode initialized", ep_id

        # Existing episode exists. Check time elapsed since last alert
        t_last = self._parse_time(ep.last_alert_time)
        time_elapsed = max(0.0, t_current - t_last)
        current_rank = self.priority_order.get(priority, 1)
        highest_rank = self.priority_order.get(ep.highest_priority, 1)
        last_rank = self.priority_order.get(ep.last_priority, 1)

        # 1. Escalation Override:
        # If patient's condition escalates to a strictly higher priority than previously seen in this episode,
        # BREAK cooldown immediately!
        if current_rank > highest_rank:
            ep.highest_priority = priority
            ep.last_priority = priority
            ep.last_alert_time = timestamp
            ep.alert_count += 1
            ep.syndrome = syndrome
            return True, True, f"Escalation override: Priority worsened from {ep.highest_priority.value} to {priority.value}", ep.episode_id

        # 2. Check if within Cooldown Window
        if time_elapsed < self.cooldown_seconds:
            ep.suppressed_count += 1
            remaining = int(self.cooldown_seconds - time_elapsed)
            return False, False, f"Suppressed by cooldown: Active episode {ep.episode_id} ({remaining}s remaining)", ep.episode_id

        # 3. Cooldown has elapsed -> Sustained ongoing alert
        ep.last_priority = priority
        ep.last_alert_time = timestamp
        ep.alert_count += 1
        return True, False, f"Cooldown refreshed for ongoing episode {ep.episode_id}", ep.episode_id

    def record_clinician_action(self, patient_id: str, action: str) -> None:
        """Updates episode state when a clinician responds."""
        if patient_id in self.active_episodes:
            self.active_episodes[patient_id].clinician_action = action
            if action.upper() in {"DISMISS", "RESOLVE"}:
                self.active_episodes[patient_id].resolved = True

    def resolve_if_stable(self, patient_id: str) -> None:
        """Marks active episode as resolved when patient vitals normalize."""
        if patient_id in self.active_episodes:
            self.active_episodes[patient_id].resolved = True

    def get_episode(self, patient_id: str) -> Optional[ActiveEpisode]:
        return self.active_episodes.get(patient_id)
