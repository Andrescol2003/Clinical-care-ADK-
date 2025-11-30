"""
Custom Tools for Clinical Care Agents - Google ADK Compatible

KEY CONCEPT: TOOLS IN GOOGLE ADK
================================
Tools extend agent capabilities beyond text generation.
In Google ADK, tools are functions that agents can call.

Tools for our clinical system:
1. Medical assessment tool (uses BioGPT)
2. Patient lookup tool
3. Drug interaction checker
4. Appointment scheduler
5. Alert notification tool

These tools follow Google ADK patterns and can be passed
to Agent() during initialization.
"""

from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
import json


# =========================================================
# TOOL DEFINITIONS FOR GOOGLE ADK
# =========================================================

def assess_symptoms(
    symptoms: str,
    patient_age: Optional[int] = None,
    patient_gender: Optional[str] = None,
    medical_history: Optional[str] = None
) -> Dict[str, Any]:
    """
    Assess patient symptoms using medical knowledge.
    
    This tool analyzes symptoms and provides an initial assessment
    including urgency level and recommended next steps.
    
    Args:
        symptoms: Description of patient symptoms
        patient_age: Patient's age in years
        patient_gender: Patient's gender (M/F/Other)
        medical_history: Relevant medical history
        
    Returns:
        Assessment including urgency level and recommendations
    """
    # In production, this would call BioGPT
    # For now, return structured assessment
    
    urgency_keywords = {
        "high": ["chest pain", "difficulty breathing", "severe", "unconscious", "bleeding heavily"],
        "medium": ["fever", "pain", "vomiting", "dizziness", "weakness"],
        "low": ["mild", "minor", "slight", "occasional"]
    }
    
    symptoms_lower = symptoms.lower()
    
    # Determine urgency
    urgency = "medium"  # default
    for level, keywords in urgency_keywords.items():
        if any(kw in symptoms_lower for kw in keywords):
            urgency = level
            break
    
    urgency_mapping = {"high": 1, "medium": 3, "low": 5}
    
    return {
        "status": "success",
        "assessment": {
            "symptoms_analyzed": symptoms,
            "urgency_level": urgency_mapping[urgency],
            "urgency_description": urgency,
            "patient_context": {
                "age": patient_age,
                "gender": patient_gender,
                "history": medical_history
            },
            "recommendation": f"Patient requires {'immediate' if urgency == 'high' else 'standard'} evaluation",
            "timestamp": datetime.now().isoformat()
        }
    }


def lookup_patient(
    patient_id: str
) -> Dict[str, Any]:
    """
    Look up patient information from the medical records system.
    
    Args:
        patient_id: Unique patient identifier
        
    Returns:
        Patient demographics, medical history, and current medications
    """
    # Mock patient database - in production, this queries actual EMR
    mock_patients = {
        "P001": {
            "patient_id": "P001",
            "name": "John Smith",
            "age": 45,
            "gender": "M",
            "medical_history": ["Hypertension", "Type 2 Diabetes"],
            "current_medications": ["Metformin 500mg", "Lisinopril 10mg"],
            "allergies": ["Penicillin"],
            "last_visit": "2024-10-15"
        },
        "P002": {
            "patient_id": "P002",
            "name": "Sarah Johnson",
            "age": 32,
            "gender": "F",
            "medical_history": ["Asthma"],
            "current_medications": ["Albuterol inhaler PRN"],
            "allergies": [],
            "last_visit": "2024-11-01"
        }
    }
    
    if patient_id in mock_patients:
        return {
            "status": "success",
            "patient": mock_patients[patient_id]
        }
    else:
        return {
            "status": "not_found",
            "message": f"Patient {patient_id} not found in system"
        }


def check_drug_interactions(
    proposed_medications: List[str],
    current_medications: List[str] = None,
    allergies: List[str] = None
) -> Dict[str, Any]:
    """
    Check for drug interactions and allergy conflicts.
    
    Args:
        proposed_medications: Medications being considered
        current_medications: Patient's current medication list
        allergies: Known drug allergies
        
    Returns:
        Interaction warnings and safety assessment
    """
    current_medications = current_medications or []
    allergies = allergies or []
    
    warnings = []
    
    # Mock interaction database
    known_interactions = {
        ("warfarin", "aspirin"): "Increased bleeding risk",
        ("metformin", "contrast dye"): "Risk of lactic acidosis",
        ("lisinopril", "potassium"): "Risk of hyperkalemia"
    }
    
    # Check interactions
    all_meds = [m.lower() for m in proposed_medications + current_medications]
    
    for (drug1, drug2), warning in known_interactions.items():
        if drug1 in ' '.join(all_meds) and drug2 in ' '.join(all_meds):
            warnings.append({
                "type": "interaction",
                "drugs": [drug1, drug2],
                "warning": warning,
                "severity": "moderate"
            })
    
    # Check allergies
    for med in proposed_medications:
        for allergy in allergies:
            if allergy.lower() in med.lower():
                warnings.append({
                    "type": "allergy",
                    "drug": med,
                    "allergen": allergy,
                    "severity": "high"
                })
    
    return {
        "status": "success",
        "safe_to_prescribe": len([w for w in warnings if w["severity"] == "high"]) == 0,
        "warnings": warnings,
        "medications_checked": proposed_medications,
        "timestamp": datetime.now().isoformat()
    }


def schedule_appointment(
    patient_id: str,
    appointment_type: str,
    preferred_date: Optional[str] = None,
    urgency: str = "routine"
) -> Dict[str, Any]:
    """
    Schedule an appointment for a patient.
    
    Args:
        patient_id: Patient identifier
        appointment_type: Type of appointment (follow_up, specialist, lab, imaging)
        preferred_date: Preferred date (ISO format)
        urgency: Urgency level (routine, urgent, emergency)
        
    Returns:
        Appointment confirmation details
    """
    # Calculate appointment date based on urgency
    urgency_days = {
        "emergency": 0,
        "urgent": 1,
        "routine": 7
    }
    
    days_out = urgency_days.get(urgency, 7)
    
    if preferred_date:
        try:
            apt_date = datetime.fromisoformat(preferred_date)
        except ValueError:
            apt_date = datetime.now() + timedelta(days=days_out)
    else:
        apt_date = datetime.now() + timedelta(days=days_out)
    
    # Mock available time slot
    apt_time = apt_date.replace(hour=10, minute=0, second=0)
    
    confirmation = {
        "status": "success",
        "appointment": {
            "confirmation_number": f"APT-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "patient_id": patient_id,
            "type": appointment_type,
            "datetime": apt_time.isoformat(),
            "location": "Main Clinic - Room 101",
            "provider": "Dr. Available",
            "instructions": get_appointment_instructions(appointment_type)
        }
    }
    
    return confirmation


def get_appointment_instructions(appointment_type: str) -> str:
    """Get preparation instructions for appointment type."""
    instructions = {
        "follow_up": "Please bring your current medication list and any questions.",
        "specialist": "Bring referral paperwork and relevant medical records.",
        "lab": "Fast for 8-12 hours before your appointment. Water is OK.",
        "imaging": "Wear comfortable clothing without metal. Arrive 15 minutes early."
    }
    return instructions.get(appointment_type, "Please arrive 15 minutes early.")


def send_alert(
    alert_type: str,
    patient_id: str,
    message: str,
    severity: str = "info",
    recipient: str = "care_team"
) -> Dict[str, Any]:
    """
    Send an alert to the care team or patient.
    
    Args:
        alert_type: Type of alert (clinical, appointment, followup)
        patient_id: Patient this alert concerns
        message: Alert message
        severity: Severity level (info, warning, urgent, critical)
        recipient: Who should receive (care_team, patient, both)
        
    Returns:
        Alert confirmation
    """
    alert = {
        "status": "sent",
        "alert": {
            "alert_id": f"ALT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "type": alert_type,
            "patient_id": patient_id,
            "message": message,
            "severity": severity,
            "recipient": recipient,
            "timestamp": datetime.now().isoformat(),
            "acknowledged": False
        }
    }
    
    return alert


def generate_clinical_note(
    patient_id: str,
    note_type: str,
    content: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a clinical documentation note.
    
    Args:
        patient_id: Patient identifier
        note_type: Type of note (soap, progress, discharge)
        content: Note content sections
        
    Returns:
        Formatted clinical note
    """
    note = {
        "status": "success",
        "note": {
            "note_id": f"NOTE-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "patient_id": patient_id,
            "type": note_type,
            "content": content,
            "created_at": datetime.now().isoformat(),
            "created_by": "Clinical AI System",
            "status": "draft",
            "requires_signature": True
        }
    }
    
    return note


# =========================================================
# TOOL REGISTRY FOR ADK
# =========================================================

def get_triage_tools() -> List[Callable]:
    """Returns tools available to the Triage Agent."""
    return [
        assess_symptoms,
        lookup_patient,
        send_alert
    ]


def get_diagnosis_tools() -> List[Callable]:
    """Returns tools available to the Diagnosis Agent."""
    return [
        assess_symptoms,
        lookup_patient,
        check_drug_interactions
    ]


def get_treatment_tools() -> List[Callable]:
    """Returns tools available to the Treatment Agent."""
    return [
        lookup_patient,
        check_drug_interactions,
        generate_clinical_note
    ]


def get_scheduling_tools() -> List[Callable]:
    """Returns tools available to the Scheduling Agent."""
    return [
        lookup_patient,
        schedule_appointment,
        send_alert
    ]


def get_documentation_tools() -> List[Callable]:
    """Returns tools available to the Documentation Agent."""
    return [
        lookup_patient,
        generate_clinical_note
    ]


def get_followup_tools() -> List[Callable]:
    """Returns tools available to the Follow-up Agent."""
    return [
        lookup_patient,
        schedule_appointment,
        send_alert
    ]


def get_all_tools() -> List[Callable]:
    """Returns all available tools."""
    return [
        assess_symptoms,
        lookup_patient,
        check_drug_interactions,
        schedule_appointment,
        send_alert,
        generate_clinical_note
    ]
