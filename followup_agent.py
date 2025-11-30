"""
Follow-up & Monitoring Agent - Google ADK Implementation

KEY CONCEPT: LOOP AGENT PATTERN
===============================
This agent demonstrates the LOOP pattern:
- Continuous monitoring cycles
- Periodic check-ins with patients
- Alert generation when thresholds exceeded
- Re-engagement for missed appointments

Unlike one-shot agents, this runs in cycles.
"""

from google.adk.agents import Agent
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from tools.clinical_tools import (
    lookup_patient,
    schedule_appointment,
    send_alert,
    get_followup_tools
)


# =========================================================
# FOLLOW-UP AGENT CONFIGURATION
# =========================================================

FOLLOWUP_INSTRUCTION = """You are a patient follow-up and monitoring AI assistant.

## YOUR ROLE
You ensure continuity of care through proactive monitoring. Your job is to:
1. Track patient progress after treatment
2. Monitor for warning signs
3. Ensure appointment adherence
4. Facilitate communication between patients and care team
5. Escalate concerns when needed

## MONITORING RESPONSIBILITIES

### 1. Appointment Adherence
- Track scheduled appointments
- Identify missed appointments
- Arrange rescheduling
- Note patterns of non-adherence

### 2. Treatment Compliance
- Check medication adherence
- Monitor for side effects
- Assess understanding of treatment plan

### 3. Clinical Monitoring
- Track symptom progression
- Watch for red flags
- Monitor for complications
- Note improvement or deterioration

### 4. Patient Communication
- Regular check-ins
- Answer questions
- Provide reminders
- Offer support

## TOOLS AVAILABLE
- `lookup_patient`: Get patient information
- `schedule_appointment`: Reschedule as needed
- `send_alert`: Notify care team of concerns

## MONITORING SCHEDULE

### Timing Based on Urgency:
- URGENT cases: Check daily for first week
- ROUTINE cases: Check at 1 week, 2 weeks, 4 weeks
- CHRONIC management: Monthly or as specified

### What to Monitor:
- Appointment attendance
- Symptom status (better/worse/same)
- Medication compliance
- Side effects
- Questions or concerns
- Warning signs

## ALERT CRITERIA

### Generate IMMEDIATE Alert:
- Symptoms significantly worsening
- New emergency symptoms
- Severe side effects
- Suicidal ideation
- Unable to contact patient for extended period

### Generate ROUTINE Alert:
- Missed appointment
- Mild side effects
- Questions needing physician input
- Requests for medication changes

## CHECK-IN PROCESS

### 1. Review Patient Status
- Last contact date
- Scheduled appointments
- Treatment plan
- Previous concerns

### 2. Assess Current State
- Symptom status
- Medication adherence
- Any concerns
- Overall wellbeing

### 3. Take Action
- Schedule if needed
- Send alerts if concerning
- Document interaction
- Plan next check-in

### 4. Document
- Record check-in
- Note any changes
- Update monitoring plan

## OUTPUT FORMAT
```
FOLLOW-UP CHECK-IN
=================
Patient ID: [ID]
Check-in Date: [Date]
Days Since Treatment: [X days]

STATUS REVIEW:
- Last Contact: [Date]
- Appointments: [Attended/Missed]
- Symptoms: [Better/Worse/Same]
- Medication Adherence: [Good/Partial/Poor]

CURRENT ASSESSMENT:
[Summary of current patient status]

CONCERNS IDENTIFIED:
- [List any concerns]

ACTIONS TAKEN:
- [What was done]

ALERTS GENERATED:
- [Any alerts sent]

NEXT CHECK-IN:
- Scheduled: [Date]
- Focus: [What to monitor]
```

## COMMUNICATION TEMPLATES

### Routine Check-in:
"Hi [Name], this is a follow-up from [Clinic]. How are you feeling since your visit? Are you taking your medications as prescribed? Any concerns?"

### Missed Appointment:
"Hi [Name], we noticed you missed your appointment on [Date]. We want to make sure you're okay. Please call us to reschedule."

### Concerning Symptoms:
"[Name], I'm sorry to hear your symptoms are worse. This is important - please contact your doctor today or go to urgent care if symptoms are severe."

## ESCALATION GUIDELINES

### Escalate to Physician When:
- Symptoms worsening despite treatment
- New symptoms develop
- Patient requesting medication changes
- Safety concerns
- Unable to reach patient for 1+ week

### Escalate to Emergency When:
- Life-threatening symptoms
- Suicidal ideation
- Severe allergic reaction
- Patient in crisis

Remember: Consistent follow-up improves outcomes and patient satisfaction.
"""


def create_followup_agent(
    model: str = "gemini-2.0-flash",
    custom_tools: List = None
) -> Agent:
    """
    Creates and configures the Follow-up Agent.
    """
    tools = custom_tools if custom_tools is not None else get_followup_tools()
    
    agent = Agent(
        name="followup_agent",
        model=model,
        instruction=FOLLOWUP_INSTRUCTION,
        tools=tools
    )
    
    return agent


# =========================================================
# MONITORING STATE MANAGEMENT
# =========================================================

class MonitoringState:
    """
    Manages monitoring state for patients.
    
    KEY CONCEPT: LOOP STATE
    =======================
    The follow-up agent maintains state across cycles.
    This class tracks what we're monitoring and when.
    """
    
    def __init__(self):
        self.active_monitors: Dict[str, Dict[str, Any]] = {}
    
    def add_patient(
        self,
        patient_id: str,
        treatment_plan: Dict[str, Any],
        appointments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Add a patient to monitoring."""
        monitor = {
            "patient_id": patient_id,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "treatment_reference": treatment_plan.get("treatment_id"),
            "diagnosis": treatment_plan.get("diagnosis_treated"),
            "scheduled_appointments": appointments,
            "check_interval_days": 7,
            "next_check": (datetime.now() + timedelta(days=7)).isoformat(),
            "check_history": [],
            "alerts_generated": [],
            "last_contact": datetime.now().isoformat()
        }
        
        self.active_monitors[patient_id] = monitor
        return monitor
    
    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get monitoring state for a patient."""
        return self.active_monitors.get(patient_id)
    
    def update_patient(
        self,
        patient_id: str,
        updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update monitoring state."""
        if patient_id in self.active_monitors:
            self.active_monitors[patient_id].update(updates)
            return self.active_monitors[patient_id]
        return None
    
    def get_due_for_check(self) -> List[str]:
        """Get list of patients due for check-in."""
        due = []
        now = datetime.now()
        
        for patient_id, monitor in self.active_monitors.items():
            if monitor["status"] != "active":
                continue
            
            next_check = datetime.fromisoformat(monitor["next_check"])
            if now >= next_check:
                due.append(patient_id)
        
        return due
    
    def record_check(
        self,
        patient_id: str,
        check_result: Dict[str, Any]
    ) -> None:
        """Record a monitoring check."""
        if patient_id in self.active_monitors:
            monitor = self.active_monitors[patient_id]
            
            # Add to history
            monitor["check_history"].append({
                "timestamp": datetime.now().isoformat(),
                "result": check_result
            })
            
            # Keep history bounded
            if len(monitor["check_history"]) > 20:
                monitor["check_history"] = monitor["check_history"][-10:]
            
            # Update last contact
            monitor["last_contact"] = datetime.now().isoformat()
            
            # Schedule next check
            interval = monitor["check_interval_days"]
            monitor["next_check"] = (
                datetime.now() + timedelta(days=interval)
            ).isoformat()


# Global monitoring state (in production, use database)
monitoring_state = MonitoringState()


# =========================================================
# FOLLOW-UP SPECIFIC FUNCTIONS
# =========================================================

def initialize_monitoring(
    patient_id: str,
    treatment_result: Dict[str, Any],
    scheduling_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Initialize monitoring for a patient after treatment.
    
    KEY CONCEPT: LOOP INITIALIZATION
    ================================
    This sets up the monitoring loop for a patient.
    """
    appointments = scheduling_result.get("confirmed_appointments", [])
    
    monitor = monitoring_state.add_patient(
        patient_id=patient_id,
        treatment_plan=treatment_result,
        appointments=appointments
    )
    
    # Generate initial check-in message
    initial_message = generate_checkin_message(
        patient_id=patient_id,
        message_type="initial",
        context={"diagnosis": treatment_result.get("diagnosis_treated")}
    )
    
    return {
        "status": "monitoring_initialized",
        "patient_id": patient_id,
        "monitor_config": {
            "check_interval_days": monitor["check_interval_days"],
            "next_check": monitor["next_check"],
            "appointments_tracked": len(appointments)
        },
        "initial_message": initial_message
    }


def run_monitoring_check(patient_id: str) -> Dict[str, Any]:
    """
    Run a monitoring check for a patient.
    
    KEY CONCEPT: LOOP ITERATION
    ===========================
    This is one iteration of the monitoring loop.
    """
    monitor = monitoring_state.get_patient(patient_id)
    
    if not monitor:
        return {"status": "error", "message": "Patient not in monitoring"}
    
    # Assess current status
    status = assess_patient_status(monitor)
    
    # Check for alerts
    alerts = check_alert_conditions(monitor, status)
    
    # Generate appropriate outreach
    if alerts:
        message_type = "concern"
    elif should_routine_checkin(monitor):
        message_type = "routine"
    else:
        message_type = None
    
    outreach_message = None
    if message_type:
        outreach_message = generate_checkin_message(
            patient_id=patient_id,
            message_type=message_type,
            context={"status": status, "alerts": alerts}
        )
    
    # Record the check
    check_result = {
        "status": status,
        "alerts": alerts,
        "outreach_sent": message_type is not None
    }
    monitoring_state.record_check(patient_id, check_result)
    
    # Send any alerts
    for alert in alerts:
        send_alert(
            alert_type="followup",
            patient_id=patient_id,
            message=alert["message"],
            severity=alert["severity"]
        )
    
    return {
        "patient_id": patient_id,
        "check_timestamp": datetime.now().isoformat(),
        "status_assessment": status,
        "alerts_generated": alerts,
        "outreach_message": outreach_message,
        "next_check": monitoring_state.get_patient(patient_id)["next_check"]
    }


def assess_patient_status(monitor: Dict[str, Any]) -> Dict[str, Any]:
    """Assess current patient status based on monitoring data."""
    appointments = monitor.get("scheduled_appointments", [])
    now = datetime.now()
    
    # Check appointment adherence
    past_appointments = [
        apt for apt in appointments
        if datetime.fromisoformat(apt["datetime"]) < now
    ]
    
    attended = sum(1 for apt in past_appointments if apt.get("attended", True))
    total = len(past_appointments)
    
    # Calculate days since last contact
    last_contact = datetime.fromisoformat(monitor["last_contact"])
    days_since_contact = (now - last_contact).days
    
    return {
        "appointment_adherence": {
            "attended": attended,
            "total": total,
            "rate": attended / total if total > 0 else 1.0
        },
        "days_since_contact": days_since_contact,
        "monitoring_status": monitor["status"],
        "check_count": len(monitor.get("check_history", []))
    }


def check_alert_conditions(
    monitor: Dict[str, Any],
    status: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Check if any alert conditions are met."""
    alerts = []
    
    # Check for missed appointments
    adherence = status["appointment_adherence"]
    if adherence["total"] > 0 and adherence["rate"] < 0.8:
        missed = adherence["total"] - adherence["attended"]
        alerts.append({
            "type": "missed_appointments",
            "severity": "warning" if missed < 2 else "urgent",
            "message": f"Patient has missed {missed} appointment(s)"
        })
    
    # Check for extended no contact
    days_since = status["days_since_contact"]
    if days_since > 14:
        alerts.append({
            "type": "no_contact",
            "severity": "warning",
            "message": f"No patient contact in {days_since} days"
        })
    
    return alerts


def should_routine_checkin(monitor: Dict[str, Any]) -> bool:
    """Determine if routine check-in is due."""
    last_contact = datetime.fromisoformat(monitor["last_contact"])
    days_since = (datetime.now() - last_contact).days
    return days_since >= monitor["check_interval_days"]


def generate_checkin_message(
    patient_id: str,
    message_type: str,
    context: Dict[str, Any] = None
) -> str:
    """Generate appropriate check-in message."""
    context = context or {}
    
    if message_type == "initial":
        return (
            f"Hello! This is a follow-up from your recent visit. "
            f"We're checking in to see how you're doing with your treatment for "
            f"{context.get('diagnosis', 'your condition')}. "
            f"Please let us know if you have any questions or concerns."
        )
    
    elif message_type == "routine":
        return (
            f"Hi! Just checking in to see how you're feeling. "
            f"Are you taking your medications as prescribed? "
            f"Any new symptoms or concerns? We're here to help!"
        )
    
    elif message_type == "concern":
        alerts = context.get("alerts", [])
        if any(a["type"] == "missed_appointments" for a in alerts):
            return (
                f"We noticed you may have missed a recent appointment. "
                f"We want to make sure you're doing well. "
                f"Please call us to reschedule - your health is important to us."
            )
        else:
            return (
                f"We haven't heard from you in a while and want to check in. "
                f"How are you feeling? Any concerns about your treatment? "
                f"Please let us know so we can help."
            )
    
    return "How are you doing today?"


def format_followup_result(
    patient_id: str,
    action: str,
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """Format follow-up result."""
    return {
        "followup_id": f"FU-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "patient_id": patient_id,
        "action": action,
        "result": result,
        "timestamp": datetime.now().isoformat(),
        "agent": "followup_agent"
    }
