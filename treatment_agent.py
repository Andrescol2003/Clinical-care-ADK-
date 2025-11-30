"""
Treatment Planning Agent - Google ADK Implementation

KEY CONCEPT: TOOL-AUGMENTED AGENT
=================================
This agent demonstrates heavy tool usage:
- Drug interaction checking
- Patient record lookup
- Clinical documentation generation

Treatment planning requires external data sources,
making it a perfect showcase for ADK tool integration.
"""

from google.adk.agents import Agent
from google.adk.tools import google_search
from typing import Any, Dict, List, Optional

from tools.clinical_tools import (
    lookup_patient,
    check_drug_interactions,
    generate_clinical_note,
    get_treatment_tools
)


# =========================================================
# TREATMENT AGENT CONFIGURATION
# =========================================================

TREATMENT_INSTRUCTION = """You are a treatment planning AI assistant supporting clinical care coordination.

## YOUR ROLE
You receive diagnostic assessments and create evidence-based treatment plans. Your job is to:
1. Review the diagnosis and clinical context
2. Generate appropriate treatment recommendations
3. Ensure medication safety (check interactions and allergies)
4. Create clear plans for the care team
5. Prepare patient education materials

## TREATMENT PLANNING PRINCIPLES

### SAFETY FIRST
- ALWAYS check for drug allergies before recommending medications
- ALWAYS check for drug-drug interactions
- Consider patient-specific factors (age, renal function, etc.)
- When in doubt, recommend safer alternatives

### EVIDENCE-BASED
- Recommend treatments with proven efficacy
- Follow clinical guidelines when available
- Consider risk/benefit for each recommendation

### PATIENT-CENTERED
- Consider patient factors (age, comorbidities)
- Account for medication burden
- Include non-pharmacological options
- Provide clear patient instructions

## TOOLS AVAILABLE
- `lookup_patient`: Get patient records, allergies, current medications
- `check_drug_interactions`: CRITICAL - use before any medication recommendation
- `generate_clinical_note`: Create documentation
- `google_search`: Look up treatment guidelines

## TREATMENT PLAN STRUCTURE

### 1. DIAGNOSIS CONFIRMATION
- Confirm the diagnosis being treated
- Note any secondary diagnoses to consider

### 2. TREATMENT GOALS
- What are we trying to achieve?
- Measurable outcomes if possible

### 3. PHARMACOLOGICAL TREATMENT
For EACH medication:
- Drug name (generic preferred)
- Dose and frequency
- Duration of treatment
- Rationale for choice
- Monitoring requirements
- Key side effects to watch

### 4. NON-PHARMACOLOGICAL TREATMENT
- Lifestyle modifications
- Physical therapy if indicated
- Dietary changes
- Activity modifications

### 5. MONITORING PLAN
- What parameters to monitor
- How often
- Target values
- When to escalate

### 6. FOLLOW-UP
- When to reassess
- Criteria for treatment success
- When to modify treatment

### 7. PATIENT EDUCATION
- Key points for patient
- Warning signs to watch
- When to seek emergency care

## OUTPUT FORMAT
```
TREATMENT PLAN
=============
Patient ID: [ID]
Diagnosis Reference: [Diagnosis ID]
Diagnosis: [What we're treating]
Date: [Date]

SAFETY CHECKS:
- Allergies Verified: [Yes/No - List allergies]
- Interactions Checked: [Yes/No - Any concerns]
- Contraindications: [None / List any]

TREATMENT GOALS:
1. [Primary goal]
2. [Secondary goals]

PHARMACOLOGICAL TREATMENT:
1. [Medication Name]
   - Dose: [Dose and frequency]
   - Duration: [How long]
   - Rationale: [Why this drug]
   - Monitoring: [What to watch]
   - Cautions: [Side effects/warnings]

[Repeat for each medication]

NON-PHARMACOLOGICAL:
- [Lifestyle/therapy recommendations]

MONITORING PLAN:
- Parameters: [What to monitor]
- Frequency: [How often]
- Targets: [Goal values]

FOLLOW-UP:
- Schedule: [When]
- Success Criteria: [How we know it's working]

PATIENT EDUCATION:
[Key points in patient-friendly language]

WARNING SIGNS - SEEK CARE IF:
- [Red flags for patient]

PHYSICIAN APPROVAL REQUIRED: Yes
```

## CRITICAL SAFETY RULES
1. NEVER recommend a medication without checking allergies
2. ALWAYS run drug interaction check
3. Adjust doses for age, weight, organ function
4. Consider pregnancy/breastfeeding status
5. Document all safety checks performed

## WORKFLOW
1. Review diagnosis and clinical picture
2. Look up patient record (allergies, current meds)
3. Generate treatment recommendations
4. Check ALL medications for interactions
5. If interactions found, modify plan
6. Generate final treatment plan
7. Create patient education materials

Remember: Treatment plans require physician approval. Be thorough and safe.
"""


def create_treatment_agent(
    model: str = "gemini-2.0-flash",
    custom_tools: List = None
) -> Agent:
    """
    Creates and configures the Treatment Planning Agent.
    """
    tools = custom_tools if custom_tools is not None else get_treatment_tools()
    tools.append(google_search)
    
    agent = Agent(
        name="treatment_agent",
        model=model,
        instruction=TREATMENT_INSTRUCTION,
        tools=tools
    )
    
    return agent


# =========================================================
# TREATMENT-SPECIFIC FUNCTIONS
# =========================================================

def run_safety_checks(
    proposed_medications: List[str],
    patient_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Runs comprehensive safety checks before finalizing treatment.
    
    KEY CONCEPT: SAFETY GATE
    ========================
    This is a critical checkpoint. Treatment cannot proceed
    until safety is verified.
    """
    safety_result = {
        "passed": True,
        "checks_performed": [],
        "warnings": [],
        "blockers": []
    }
    
    allergies = patient_data.get("allergies", [])
    current_meds = patient_data.get("current_medications", [])
    
    # Check 1: Allergy verification
    safety_result["checks_performed"].append("allergy_check")
    for med in proposed_medications:
        for allergy in allergies:
            if allergy.lower() in med.lower():
                safety_result["blockers"].append({
                    "type": "allergy_conflict",
                    "medication": med,
                    "allergen": allergy,
                    "action": "DO NOT PRESCRIBE"
                })
                safety_result["passed"] = False
    
    # Check 2: Drug interactions (using tool)
    safety_result["checks_performed"].append("interaction_check")
    interaction_result = check_drug_interactions(
        proposed_medications=proposed_medications,
        current_medications=current_meds,
        allergies=allergies
    )
    
    if interaction_result["warnings"]:
        for warning in interaction_result["warnings"]:
            if warning["severity"] == "high":
                safety_result["blockers"].append(warning)
                safety_result["passed"] = False
            else:
                safety_result["warnings"].append(warning)
    
    # Check 3: Special populations
    safety_result["checks_performed"].append("population_check")
    if patient_data.get("pregnant"):
        safety_result["warnings"].append({
            "type": "pregnancy",
            "message": "Verify all medications are pregnancy-safe"
        })
    
    if patient_data.get("age", 0) > 65:
        safety_result["warnings"].append({
            "type": "elderly",
            "message": "Consider dose adjustments for elderly patient"
        })
    
    return safety_result


def format_treatment_plan(
    patient_id: str,
    diagnosis_reference: str,
    diagnosis: str,
    medications: List[Dict[str, Any]],
    non_pharm: List[str],
    monitoring: Dict[str, Any],
    followup: Dict[str, Any],
    patient_education: str,
    safety_checks: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Formats the complete treatment plan for handoff.
    """
    from datetime import datetime
    
    return {
        "treatment_id": f"TX-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "patient_id": patient_id,
        "diagnosis_reference": diagnosis_reference,
        "diagnosis_treated": diagnosis,
        "safety_verification": {
            "passed": safety_checks["passed"],
            "checks_performed": safety_checks["checks_performed"],
            "warnings": safety_checks["warnings"],
            "blockers": safety_checks["blockers"]
        },
        "pharmacological_treatment": medications,
        "non_pharmacological_treatment": non_pharm,
        "monitoring_plan": monitoring,
        "follow_up_plan": followup,
        "patient_education": patient_education,
        "timestamp": datetime.now().isoformat(),
        "status": "pending_approval" if safety_checks["passed"] else "safety_hold",
        "requires_physician_approval": True,
        "next_agents": ["documentation_agent", "scheduling_agent"]
    }


def generate_patient_education(
    diagnosis: str,
    medications: List[Dict[str, Any]],
    warnings: List[str]
) -> str:
    """
    Generates patient-friendly education materials.
    
    KEY CONCEPT: MULTI-AUDIENCE OUTPUT
    ==================================
    Same information, different presentation:
    - Clinical plan for healthcare team
    - Simplified version for patient
    """
    education = []
    
    education.append("UNDERSTANDING YOUR TREATMENT")
    education.append("=" * 30)
    education.append(f"\nYour Condition: {diagnosis}\n")
    
    education.append("YOUR MEDICATIONS:")
    for med in medications:
        education.append(f"• {med.get('name', 'Medication')}")
        education.append(f"  - Take: {med.get('dose', 'As directed')}")
        education.append(f"  - Why: {med.get('purpose', 'To help your condition')}")
        education.append("")
    
    education.append("IMPORTANT - CALL YOUR DOCTOR IF:")
    for warning in warnings:
        education.append(f"• {warning}")
    
    education.append("\nQUESTIONS?")
    education.append("Don't hesitate to call our office if you have any concerns.")
    
    return "\n".join(education)


# =========================================================
# BIOGPT INTEGRATION FOR TREATMENT
# =========================================================

async def generate_treatment_with_biogpt(
    diagnosis: str,
    patient_context: Dict[str, Any],
    biogpt_wrapper
) -> str:
    """
    Uses BioGPT to generate treatment recommendations.
    """
    prompt = f"""KEYWORDS: treatment plan, {diagnosis}

Patient Context:
- Age: {patient_context.get('age', 'Unknown')}
- Diagnosis: {diagnosis}
- Medical History: {patient_context.get('medical_history', 'None')}
- Current Medications: {patient_context.get('current_medications', 'None')}
- Allergies: {patient_context.get('allergies', 'None')}

Generate a comprehensive treatment plan including:
1. Recommended medications with dosing
2. Non-pharmacological interventions
3. Monitoring parameters
4. Follow-up schedule
5. Patient education points

TRANSCRIPTION:"""
    
    response = await biogpt_wrapper.agenerate(prompt)
    return response
