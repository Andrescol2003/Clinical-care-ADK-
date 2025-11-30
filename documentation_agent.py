"""
Documentation Agent - Google ADK Implementation

KEY CONCEPT: AGGREGATION & STRUCTURED OUTPUT
============================================
This agent demonstrates:
1. Aggregating data from multiple sources (all previous agents)
2. Generating multiple output formats (SOAP, progress notes, etc.)
3. Template-guided generation
4. Compliance documentation
"""

from google.adk.agents import Agent
from typing import Any, Dict, List, Optional

from tools.clinical_tools import (
    lookup_patient,
    generate_clinical_note,
    get_documentation_tools
)


# =========================================================
# DOCUMENTATION AGENT CONFIGURATION
# =========================================================

DOCUMENTATION_INSTRUCTION = """You are a clinical documentation AI assistant.

## YOUR ROLE
You aggregate information from the clinical care workflow and generate accurate, professional documentation. Your job is to:
1. Collect data from triage, diagnosis, and treatment agents
2. Generate appropriate clinical documentation
3. Ensure compliance with documentation standards
4. Create clear, complete records

## DOCUMENTATION PRINCIPLES

### Accuracy
- Document exactly what occurred
- Use specific, measurable terms
- Include all relevant findings
- Note any limitations or uncertainties

### Completeness
- Include all essential elements
- Don't omit negative findings if relevant
- Document patient responses
- Record all interventions

### Timeliness
- Document promptly
- Note actual times when relevant
- Use current date/time stamps

### Professional Language
- Use standard medical terminology
- Be objective and factual
- Avoid judgmental language
- Write clearly and concisely

## TOOLS AVAILABLE
- `lookup_patient`: Get patient demographics
- `generate_clinical_note`: Create formatted notes

## SUPPORTED DOCUMENTATION TYPES

### 1. SOAP NOTE
Standard format for clinical encounters:
- S (Subjective): What the patient reports
- O (Objective): What we observe/measure
- A (Assessment): Our clinical impression
- P (Plan): What we're going to do

### 2. PROGRESS NOTE
For ongoing care documentation:
- Current status
- Changes since last visit
- Response to treatment
- Updated plan

### 3. DISCHARGE SUMMARY
For care transitions:
- Reason for encounter
- Course of care
- Discharge diagnosis
- Follow-up instructions

### 4. REFERRAL LETTER
For specialist communication:
- Reason for referral
- Relevant history
- Specific questions

## OUTPUT FORMATS

### SOAP NOTE FORMAT
```
CLINICAL NOTE - SOAP FORMAT
===========================
Patient: [Name/ID]
Date: [Date]
Provider: [Provider]

SUBJECTIVE:
Chief Complaint: [CC]
History of Present Illness: [HPI]
Review of Systems: [ROS if done]
Past Medical History: [Relevant PMH]
Medications: [Current meds]
Allergies: [Allergies]

OBJECTIVE:
Vital Signs: [If available]
Physical Exam: [Relevant findings]
Diagnostic Results: [Labs/imaging if available]

ASSESSMENT:
[Numbered list of diagnoses/problems]
1. [Primary diagnosis]
2. [Secondary diagnoses]

PLAN:
[For each diagnosis]
1. [Diagnosis]: [Plan]
   - Medications: [Prescriptions]
   - Tests: [Orders]
   - Follow-up: [Timing]

Patient Education: [What was discussed]
Disposition: [Where patient goes next]

Electronically signed by: Clinical AI System
Status: DRAFT - Requires Physician Attestation
```

### PROGRESS NOTE FORMAT
```
PROGRESS NOTE
=============
Patient: [Name/ID]
Date: [Date]

Interval History:
[What happened since last visit]

Current Status:
[How patient is doing]

Assessment:
[Clinical impression]

Plan Updates:
[Changes to treatment plan]

Next Steps:
[Follow-up plan]
```

## DOCUMENTATION GUIDELINES

### What to Include:
- Patient identification
- Date and time
- Chief complaint
- Relevant history
- Exam findings
- Assessment/diagnosis
- Treatment plan
- Follow-up plan
- Who documented

### What to Avoid:
- Vague terms ("some improvement")
- Judgmental language
- Speculation without documentation
- Incomplete sentences
- Abbreviations that could be misunderstood

## COMPLIANCE REQUIREMENTS
- All notes require physician review
- Mark AI-generated notes as DRAFT
- Include timestamp
- Include all required elements
- Flag any missing information

Remember: Documentation is the legal record of care. Be accurate and complete.
"""


def create_documentation_agent(
    model: str = "gemini-2.0-flash",
    custom_tools: List = None
) -> Agent:
    """
    Creates and configures the Documentation Agent.
    """
    tools = custom_tools if custom_tools is not None else get_documentation_tools()
    
    agent = Agent(
        name="documentation_agent",
        model=model,
        instruction=DOCUMENTATION_INSTRUCTION,
        tools=tools
    )
    
    return agent


# =========================================================
# DOCUMENTATION-SPECIFIC FUNCTIONS
# =========================================================

def aggregate_clinical_data(
    triage_result: Dict[str, Any] = None,
    diagnosis_result: Dict[str, Any] = None,
    treatment_result: Dict[str, Any] = None,
    scheduling_result: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Aggregates data from all agents for documentation.
    
    KEY CONCEPT: DATA AGGREGATION
    =============================
    Documentation needs the complete picture.
    This function pulls together all available data.
    """
    aggregated = {
        "patient_id": None,
        "encounter_date": None,
        "subjective": {},
        "objective": {},
        "assessment": {},
        "plan": {}
    }
    
    # Extract from triage
    if triage_result:
        aggregated["patient_id"] = triage_result.get("patient_id")
        aggregated["encounter_date"] = triage_result.get("timestamp")
        aggregated["subjective"]["chief_complaint"] = triage_result.get("chief_complaint")
        aggregated["subjective"]["triage_assessment"] = triage_result.get("assessment")
        aggregated["objective"]["urgency_level"] = triage_result.get("urgency_level")
        aggregated["objective"]["red_flags"] = triage_result.get("red_flags", [])
    
    # Extract from diagnosis
    if diagnosis_result:
        aggregated["patient_id"] = aggregated["patient_id"] or diagnosis_result.get("patient_id")
        clinical_picture = diagnosis_result.get("clinical_picture", {})
        
        aggregated["subjective"]["medical_history"] = clinical_picture.get("medical_history", [])
        aggregated["subjective"]["medications"] = clinical_picture.get("current_medications", [])
        aggregated["subjective"]["allergies"] = clinical_picture.get("allergies", [])
        
        aggregated["assessment"]["primary_diagnosis"] = diagnosis_result.get("primary_diagnosis")
        aggregated["assessment"]["differential"] = diagnosis_result.get("differential_diagnoses", [])
        aggregated["assessment"]["clinical_reasoning"] = diagnosis_result.get("clinical_reasoning")
        
        aggregated["plan"]["investigations"] = diagnosis_result.get("investigations", {})
        aggregated["plan"]["specialist_referral"] = diagnosis_result.get("specialist_referral", {})
    
    # Extract from treatment
    if treatment_result:
        aggregated["plan"]["medications"] = treatment_result.get("pharmacological_treatment", [])
        aggregated["plan"]["non_pharm"] = treatment_result.get("non_pharmacological_treatment", [])
        aggregated["plan"]["monitoring"] = treatment_result.get("monitoring_plan", {})
        aggregated["plan"]["follow_up"] = treatment_result.get("follow_up_plan", {})
        aggregated["plan"]["patient_education"] = treatment_result.get("patient_education")
    
    # Extract from scheduling
    if scheduling_result:
        aggregated["plan"]["scheduled_appointments"] = scheduling_result.get("confirmed_appointments", [])
    
    return aggregated


def generate_soap_note(aggregated_data: Dict[str, Any]) -> str:
    """
    Generates a SOAP note from aggregated data.
    
    KEY CONCEPT: TEMPLATE-BASED GENERATION
    ======================================
    Using structured templates ensures consistent,
    complete documentation.
    """
    from datetime import datetime
    
    note = []
    
    # Header
    note.append("=" * 60)
    note.append("CLINICAL NOTE - SOAP FORMAT")
    note.append("=" * 60)
    note.append(f"Patient ID: {aggregated_data.get('patient_id', 'Unknown')}")
    note.append(f"Date: {aggregated_data.get('encounter_date', datetime.now().isoformat())}")
    note.append(f"Generated: {datetime.now().isoformat()}")
    note.append("")
    
    # SUBJECTIVE
    note.append("SUBJECTIVE:")
    note.append("-" * 40)
    subj = aggregated_data.get("subjective", {})
    note.append(f"Chief Complaint: {subj.get('chief_complaint', 'Not documented')}")
    note.append("")
    note.append(f"History of Present Illness:")
    note.append(f"  {subj.get('triage_assessment', 'See triage assessment')}")
    note.append("")
    
    if subj.get("medical_history"):
        note.append(f"Past Medical History: {', '.join(subj['medical_history'])}")
    if subj.get("medications"):
        note.append(f"Current Medications: {', '.join(subj['medications'])}")
    if subj.get("allergies"):
        note.append(f"Allergies: {', '.join(subj['allergies'])}")
    else:
        note.append("Allergies: NKDA (No Known Drug Allergies)")
    note.append("")
    
    # OBJECTIVE
    note.append("OBJECTIVE:")
    note.append("-" * 40)
    obj = aggregated_data.get("objective", {})
    note.append(f"Triage Urgency Level: {obj.get('urgency_level', 'Not assigned')}")
    
    red_flags = obj.get("red_flags", [])
    if red_flags:
        note.append(f"Red Flags Identified: {', '.join(red_flags)}")
    else:
        note.append("Red Flags: None identified")
    note.append("")
    
    # ASSESSMENT
    note.append("ASSESSMENT:")
    note.append("-" * 40)
    assess = aggregated_data.get("assessment", {})
    note.append(f"Primary Diagnosis: {assess.get('primary_diagnosis', 'Pending')}")
    
    differential = assess.get("differential", [])
    if differential:
        note.append("Differential Diagnoses:")
        for i, dx in enumerate(differential[:5], 1):
            if isinstance(dx, dict):
                note.append(f"  {i}. {dx.get('diagnosis', 'Unknown')}")
            else:
                note.append(f"  {i}. {dx}")
    
    if assess.get("clinical_reasoning"):
        note.append(f"\nClinical Reasoning: {assess['clinical_reasoning'][:300]}...")
    note.append("")
    
    # PLAN
    note.append("PLAN:")
    note.append("-" * 40)
    plan = aggregated_data.get("plan", {})
    
    # Medications
    meds = plan.get("medications", [])
    if meds:
        note.append("Medications:")
        for med in meds:
            if isinstance(med, dict):
                note.append(f"  - {med.get('name', 'Unknown')}: {med.get('dose', 'See prescription')}")
            else:
                note.append(f"  - {med}")
    
    # Non-pharmacological
    non_pharm = plan.get("non_pharm", [])
    if non_pharm:
        note.append("\nNon-pharmacological:")
        for item in non_pharm:
            note.append(f"  - {item}")
    
    # Investigations
    investigations = plan.get("investigations", {})
    if investigations:
        note.append("\nInvestigations Ordered:")
        if isinstance(investigations, dict):
            for category, tests in investigations.items():
                note.append(f"  {category}: {tests}")
        else:
            note.append(f"  {investigations}")
    
    # Follow-up
    follow_up = plan.get("follow_up", {})
    if follow_up:
        note.append(f"\nFollow-up: {follow_up.get('schedule', 'To be scheduled')}")
    
    # Scheduled appointments
    appointments = plan.get("scheduled_appointments", [])
    if appointments:
        note.append("\nScheduled Appointments:")
        for apt in appointments:
            note.append(f"  - {apt.get('type', 'Appointment')}: {apt.get('datetime', 'TBD')}")
    
    note.append("")
    
    # Footer
    note.append("=" * 60)
    note.append("Status: DRAFT - Requires Physician Review and Attestation")
    note.append("Generated by: Clinical Care AI System")
    note.append("=" * 60)
    
    return "\n".join(note)


def format_documentation_result(
    patient_id: str,
    document_type: str,
    document_content: str,
    source_references: List[str]
) -> Dict[str, Any]:
    """
    Formats the documentation result.
    """
    from datetime import datetime
    
    return {
        "document_id": f"DOC-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "patient_id": patient_id,
        "document_type": document_type,
        "content": document_content,
        "source_references": source_references,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "created_by": "documentation_agent",
            "status": "draft",
            "requires_attestation": True,
            "word_count": len(document_content.split())
        },
        "compliance": {
            "required_elements_present": True,
            "hipaa_compliant": True,
            "physician_review_required": True
        }
    }


# =========================================================
# BIOGPT INTEGRATION FOR DOCUMENTATION
# =========================================================

async def enhance_documentation_with_biogpt(
    document_section: str,
    clinical_context: Dict[str, Any],
    biogpt_wrapper
) -> str:
    """
    Uses BioGPT to enhance documentation with medical language.
    """
    prompt = f"""KEYWORDS: clinical documentation, {document_section}

Clinical Context:
{clinical_context}

Enhance this clinical documentation section using appropriate medical terminology.
Maintain accuracy while improving clarity and completeness.
Use standard medical documentation conventions.

TRANSCRIPTION:"""
    
    response = await biogpt_wrapper.agenerate(prompt)
    return response
