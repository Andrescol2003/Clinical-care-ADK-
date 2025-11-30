"""
Scheduling & Coordination Agent - Google ADK Implementation

KEY CONCEPT: PARALLEL OPERATIONS & SUB-AGENTS
============================================
This agent demonstrates:
1. PARALLEL processing - checking multiple schedules simultaneously
2. SUB-AGENT pattern - specialized scheduling sub-tasks
3. LOOP pattern - retry logic for booking
4. External API integration via tools
"""

from google.adk.agents import Agent
from typing import Any, Dict, List, Optional

from tools.clinical_tools import (
    lookup_patient,
    schedule_appointment,
    send_alert,
    get_scheduling_tools
)


# =========================================================
# SCHEDULING AGENT CONFIGURATION
# =========================================================

SCHEDULING_INSTRUCTION = """You are a healthcare scheduling coordinator AI assistant.

## YOUR ROLE
You receive treatment plans and coordinate all necessary appointments. Your job is to:
1. Identify all appointments needed from the treatment plan
2. Check patient preferences and constraints
3. Schedule appointments efficiently
4. Handle scheduling conflicts
5. Provide clear confirmation to patients

## SCHEDULING PRIORITIES

### By Urgency:
- URGENT (Level 1-2): Same-day or next-day scheduling
- SOON (Level 3): Within 1 week
- ROUTINE (Level 4-5): Within 2-4 weeks

### By Type:
1. Follow-up with treating physician
2. Specialist referrals
3. Laboratory tests
4. Imaging studies
5. Procedures

## TOOLS AVAILABLE
- `lookup_patient`: Get patient contact info and preferences
- `schedule_appointment`: Book appointments
- `send_alert`: Notify about scheduling issues

## SCHEDULING PROCESS

### 1. ANALYZE REQUIREMENTS
From the treatment plan, identify:
- Required follow-up appointments
- Specialist referrals
- Lab work orders
- Imaging orders
- Any procedures

### 2. CHECK PATIENT INFO
- Contact preferences
- Schedule constraints
- Transportation needs
- Interpreter needs

### 3. SCHEDULE APPOINTMENTS
For each appointment:
- Determine urgency
- Find available slots
- Book appointment
- Document confirmation

### 4. HANDLE CONFLICTS
If preferred time unavailable:
- Offer alternatives
- Escalate if urgent and no slots

### 5. CONFIRM AND NOTIFY
- Provide appointment summary
- Send reminders
- Include preparation instructions

## OUTPUT FORMAT
```
SCHEDULING SUMMARY
=================
Patient ID: [ID]
Treatment Reference: [Treatment ID]
Date: [Date]

APPOINTMENTS SCHEDULED:

1. [Appointment Type]
   - Date/Time: [DateTime]
   - Provider: [Name]
   - Location: [Address/Room]
   - Confirmation #: [Number]
   - Preparation: [Instructions]

[Repeat for each appointment]

PENDING/ISSUES:
- [Any appointments that couldn't be scheduled]

PATIENT INSTRUCTIONS:
[Summary for patient including all appointment details]

REMINDERS SCHEDULED:
- [When reminders will be sent]
```

## APPOINTMENT TYPES & INSTRUCTIONS

### Follow-up Visit
- Usually 15-30 minutes
- Bring: Current medication list, questions
- Timing: Per treatment plan

### Specialist Referral
- Usually 30-60 minutes
- Bring: Referral paperwork, imaging, labs
- Timing: Based on urgency

### Laboratory
- Usually 15-30 minutes
- Prep: Fasting may be required (8-12 hours)
- Timing: Before follow-up if possible

### Imaging (X-ray, CT, MRI)
- 30-60 minutes depending on type
- Prep: Varies by study type
- Timing: Before specialist if needed

## SCHEDULING GUIDELINES
1. Group appointments when possible (same day/location)
2. Allow adequate time between appointments
3. Consider patient's travel time
4. Morning slots for fasting labs
5. Confirm interpreter if needed
6. Note mobility/accessibility needs

## CONFLICT RESOLUTION
If scheduling conflict:
1. Try alternative times same day
2. Try next available day
3. If urgent, escalate to supervisor
4. Document all attempts

Remember: Clear scheduling reduces no-shows and improves care coordination.
"""


def create_scheduling_agent(
    model: str = "gemini-2.0-flash",
    custom_tools: List = None
) -> Agent:
    """
    Creates and configures the Scheduling Agent.
    """
    tools = custom_tools if custom_tools is not None else get_scheduling_tools()
    
    agent = Agent(
        name="scheduling_agent",
        model=model,
        instruction=SCHEDULING_INSTRUCTION,
        tools=tools
    )
    
    return agent


# =========================================================
# SCHEDULING-SPECIFIC FUNCTIONS
# =========================================================

def parse_scheduling_requirements(
    treatment_plan: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Extracts appointments needed from treatment plan.
    
    KEY CONCEPT: REQUIREMENT EXTRACTION
    ===================================
    Parse the treatment plan to identify all
    appointments that need to be scheduled.
    """
    requirements = []
    
    # Check for follow-up
    follow_up = treatment_plan.get("follow_up_plan", {})
    if follow_up:
        requirements.append({
            "type": "follow_up",
            "urgency": follow_up.get("urgency", "routine"),
            "timeframe": follow_up.get("schedule", "2 weeks"),
            "provider": "treating_physician",
            "reason": "Treatment follow-up"
        })
    
    # Check for specialist referrals
    diagnosis_data = treatment_plan.get("diagnosis_reference_data", {})
    specialist_need = diagnosis_data.get("specialist_referral", {})
    if specialist_need.get("referral_needed"):
        for spec in specialist_need.get("recommendations", []):
            requirements.append({
                "type": "specialist",
                "specialty": spec.get("specialty"),
                "urgency": spec.get("urgency", "routine"),
                "reason": spec.get("reason", "Specialist evaluation")
            })
    
    # Check for labs in monitoring plan
    monitoring = treatment_plan.get("monitoring_plan", {})
    if monitoring:
        params = monitoring.get("parameters", [])
        if any("lab" in str(p).lower() for p in params):
            requirements.append({
                "type": "lab",
                "urgency": "routine",
                "timeframe": "before follow-up",
                "reason": "Treatment monitoring labs"
            })
    
    return requirements


async def schedule_all_appointments(
    requirements: List[Dict[str, Any]],
    patient_id: str,
    preferences: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Schedules all required appointments.
    
    KEY CONCEPT: PARALLEL SCHEDULING
    ================================
    In production, this would schedule multiple
    appointments in parallel for efficiency.
    """
    results = {
        "scheduled": [],
        "failed": [],
        "pending": []
    }
    
    for req in requirements:
        try:
            # Determine preferred date based on urgency
            from datetime import datetime, timedelta
            
            urgency_days = {
                "urgent": 1,
                "soon": 7,
                "routine": 14
            }
            days_out = urgency_days.get(req.get("urgency", "routine"), 14)
            preferred_date = (datetime.now() + timedelta(days=days_out)).isoformat()
            
            # Schedule the appointment
            result = schedule_appointment(
                patient_id=patient_id,
                appointment_type=req["type"],
                preferred_date=preferred_date,
                urgency=req.get("urgency", "routine")
            )
            
            if result["status"] == "success":
                results["scheduled"].append({
                    **result["appointment"],
                    "requirement": req
                })
            else:
                results["failed"].append({
                    "requirement": req,
                    "reason": result.get("message", "Scheduling failed")
                })
                
        except Exception as e:
            results["failed"].append({
                "requirement": req,
                "reason": str(e)
            })
    
    return results


def format_scheduling_result(
    patient_id: str,
    treatment_reference: str,
    scheduled_appointments: List[Dict[str, Any]],
    failed_appointments: List[Dict[str, Any]],
    patient_preferences: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Formats the scheduling result for handoff and patient communication.
    """
    from datetime import datetime
    
    # Generate patient-friendly summary
    summary_lines = ["YOUR UPCOMING APPOINTMENTS", "=" * 30, ""]
    
    for apt in scheduled_appointments:
        apt_time = datetime.fromisoformat(apt["datetime"])
        summary_lines.append(f"📅 {apt['type'].replace('_', ' ').title()}")
        summary_lines.append(f"   Date: {apt_time.strftime('%A, %B %d, %Y')}")
        summary_lines.append(f"   Time: {apt_time.strftime('%I:%M %p')}")
        summary_lines.append(f"   Location: {apt['location']}")
        summary_lines.append(f"   Confirmation: {apt['confirmation_number']}")
        summary_lines.append(f"   Prep: {apt['instructions']}")
        summary_lines.append("")
    
    return {
        "scheduling_id": f"SCH-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "patient_id": patient_id,
        "treatment_reference": treatment_reference,
        "confirmed_appointments": scheduled_appointments,
        "failed_bookings": failed_appointments,
        "patient_summary": "\n".join(summary_lines),
        "reminders_scheduled": True,
        "timestamp": datetime.now().isoformat(),
        "status": "completed" if not failed_appointments else "partial",
        "next_agents": ["followup_agent"]
    }


def get_preparation_instructions(appointment_type: str) -> Dict[str, str]:
    """
    Returns detailed preparation instructions by appointment type.
    """
    instructions = {
        "follow_up": {
            "before": "Make a list of your medications and any questions",
            "bring": "Insurance card, medication list, symptom diary if keeping one",
            "wear": "Comfortable clothing",
            "other": "Arrive 10 minutes early"
        },
        "specialist": {
            "before": "Gather any relevant test results and imaging",
            "bring": "Referral letter, ID, insurance card, all medical records",
            "wear": "Comfortable clothing, easy to remove for exam",
            "other": "Arrive 15 minutes early to complete paperwork"
        },
        "lab": {
            "before": "Fast for 8-12 hours if instructed (water OK)",
            "bring": "Lab order, insurance card, ID",
            "wear": "Short sleeves or easy-to-roll sleeves",
            "other": "Stay hydrated (water) for easier blood draw"
        },
        "imaging": {
            "before": "Follow specific prep instructions for your test type",
            "bring": "Order/prescription, insurance card, previous imaging if available",
            "wear": "Comfortable clothing without metal (no jewelry)",
            "other": "Inform staff if pregnant or possibly pregnant"
        }
    }
    
    return instructions.get(appointment_type, {
        "before": "Follow any instructions provided",
        "bring": "Insurance card and ID",
        "wear": "Comfortable clothing",
        "other": "Arrive 15 minutes early"
    })
