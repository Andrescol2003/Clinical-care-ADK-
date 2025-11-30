"""
Clinical Care Tools Package

Custom tools for Google ADK agents.
"""

from .clinical_tools import (
    # Core tools
    assess_symptoms,
    lookup_patient,
    check_drug_interactions,
    schedule_appointment,
    send_alert,
    generate_clinical_note,
    
    # Tool getters by agent
    get_triage_tools,
    get_diagnosis_tools,
    get_treatment_tools,
    get_scheduling_tools,
    get_documentation_tools,
    get_followup_tools,
    get_all_tools
)

__all__ = [
    "assess_symptoms",
    "lookup_patient",
    "check_drug_interactions",
    "schedule_appointment",
    "send_alert",
    "generate_clinical_note",
    "get_triage_tools",
    "get_diagnosis_tools",
    "get_treatment_tools",
    "get_scheduling_tools",
    "get_documentation_tools",
    "get_followup_tools",
    "get_all_tools"
]
