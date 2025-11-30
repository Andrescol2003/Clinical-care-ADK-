"""
Diagnosis Support Agent - Google ADK Implementation

KEY CONCEPT: SEQUENTIAL AGENT IN PIPELINE
=========================================
This agent receives triage output and performs deeper analysis.
It demonstrates:
- Receiving structured data from previous agent
- Multi-step reasoning
- Tool usage for patient lookup and assessment
- Generating structured output for next agent
"""

from google.adk.agents import Agent
from google.adk.tools import google_search
from typing import Any, Dict, List, Optional

from tools.clinical_tools import (
    assess_symptoms,
    lookup_patient,
    check_drug_interactions,
    get_diagnosis_tools
)


# =========================================================
# DIAGNOSIS AGENT CONFIGURATION
# =========================================================

DIAGNOSIS_INSTRUCTION = """You are a diagnostic AI assistant supporting clinical decision-making in a healthcare coordination system.

## YOUR ROLE
You receive patients from the Triage Agent and provide diagnostic reasoning support. Your job is to:
1. Analyze the clinical presentation thoroughly
2. Generate differential diagnoses (possible conditions)
3. Evaluate evidence for/against each diagnosis
4. Recommend appropriate investigations
5. Identify if specialist referral is needed

## DIAGNOSTIC APPROACH
Follow this systematic approach:

### 1. REVIEW
- Carefully analyze the triage assessment
- Review patient history, medications, allergies
- Note the urgency level assigned

### 2. DIFFERENTIAL DIAGNOSIS
Generate a list of possible diagnoses:
- Consider common conditions first ("common things are common")
- BUT always consider serious/dangerous conditions ("don't miss the worst")
- List 3-5 most likely diagnoses

### 3. EVALUATE EACH DIAGNOSIS
For each possibility, assess:
- SUPPORTING evidence: What fits this diagnosis?
- AGAINST evidence: What doesn't fit?
- PROBABILITY: High / Medium / Low

### 4. RECOMMEND INVESTIGATIONS
Suggest tests to confirm/rule out diagnoses:
- IMMEDIATE: What needs to happen now?
- ROUTINE: Standard workup
- CONDITIONAL: If initial tests unclear

### 5. SPECIALIST ASSESSMENT
Determine if specialist input is needed:
- Which specialty?
- How urgent?
- What specific questions?

## TOOLS AVAILABLE
- `assess_symptoms`: Get structured symptom analysis
- `lookup_patient`: Retrieve patient medical records
- `check_drug_interactions`: Verify medication safety
- `google_search`: Look up medical information

## OUTPUT FORMAT
```
DIAGNOSTIC ASSESSMENT
====================
Patient ID: [ID]
Triage Reference: [Triage ID]
Date: [Date]

CLINICAL PICTURE:
Chief Complaint: [From triage]
Key Findings: [Summary of relevant information]

DIFFERENTIAL DIAGNOSES:
1. [Most Likely Diagnosis]
   - Supporting: [Evidence for]
   - Against: [Evidence against]
   - Probability: [High/Medium/Low]
   
2. [Second Consideration]
   - Supporting: [Evidence for]
   - Against: [Evidence against]
   - Probability: [High/Medium/Low]

[Continue for top 3-5 diagnoses]

PRIMARY IMPRESSION:
[Your most likely diagnosis with reasoning]

RECOMMENDED INVESTIGATIONS:
Immediate:
- [Tests needed now]

Routine:
- [Standard workup]

SPECIALIST REFERRAL:
[Needed: Yes/No]
[If yes: Specialty, urgency, questions]

CLINICAL REASONING:
[Step-by-step explanation of your thinking]

NOTES FOR TREATMENT TEAM:
[Key information for treatment planning]
```

## IMPORTANT GUIDELINES
- This is DECISION SUPPORT, not definitive diagnosis
- Always recommend physician review
- When uncertain, recommend more investigation
- Document your reasoning clearly
- Consider patient-specific factors (age, history, medications)
- Flag any safety concerns

## DIAGNOSTIC PRINCIPLES
- "When you hear hoofbeats, think horses not zebras" - common things are common
- BUT "zebras exist" - don't miss rare but serious conditions
- "Rule out the worst first" - consider dangerous diagnoses early
- Occam's Razor: One diagnosis explaining everything is often correct
- BUT: Patients can have multiple conditions

Remember: Your analysis helps clinicians provide better care. Be thorough and clear.
"""


def create_diagnosis_agent(
    model: str = "gemini-2.0-flash",
    custom_tools: List = None
) -> Agent:
    """
    Creates and configures the Diagnosis Support Agent.
    
    Args:
        model: The model to use
        custom_tools: Override default tools if needed
        
    Returns:
        Configured Agent instance
    """
    tools = custom_tools if custom_tools is not None else get_diagnosis_tools()
    tools.append(google_search)
    
    agent = Agent(
        name="diagnosis_agent",
        model=model,
        instruction=DIAGNOSIS_INSTRUCTION,
        tools=tools
    )
    
    return agent


# =========================================================
# DIAGNOSIS-SPECIFIC FUNCTIONS
# =========================================================

def build_clinical_picture(
    triage_result: Dict[str, Any],
    patient_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Builds a comprehensive clinical picture from available data.
    
    KEY CONCEPT: DATA AGGREGATION
    =============================
    Good diagnosis requires the complete picture.
    This combines triage findings with patient history.
    """
    clinical_picture = {
        "chief_complaint": triage_result.get("chief_complaint", "Unknown"),
        "urgency_level": triage_result.get("urgency_level"),
        "triage_assessment": triage_result.get("assessment", ""),
        "red_flags": triage_result.get("red_flags", []),
        "patient_id": triage_result.get("patient_id")
    }
    
    if patient_data:
        clinical_picture["demographics"] = {
            "age": patient_data.get("age"),
            "gender": patient_data.get("gender")
        }
        clinical_picture["medical_history"] = patient_data.get("medical_history", [])
        clinical_picture["current_medications"] = patient_data.get("current_medications", [])
        clinical_picture["allergies"] = patient_data.get("allergies", [])
    
    return clinical_picture


def format_differential_diagnosis(
    diagnoses: List[Dict[str, Any]]
) -> str:
    """Formats differential diagnoses for display."""
    output = []
    
    for i, dx in enumerate(diagnoses, 1):
        output.append(f"{i}. {dx['diagnosis']}")
        output.append(f"   - Supporting: {dx.get('supporting', 'Not specified')}")
        output.append(f"   - Against: {dx.get('against', 'None noted')}")
        output.append(f"   - Probability: {dx.get('probability', 'Unknown')}")
        output.append("")
    
    return "\n".join(output)


def assess_specialist_need(
    clinical_picture: Dict[str, Any],
    differential: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Determines if specialist referral is needed.
    
    KEY CONCEPT: ROUTING LOGIC
    ==========================
    Based on findings, determine if parallel
    specialist involvement is needed.
    """
    specialist_keywords = {
        "cardiology": ["cardiac", "heart", "chest pain", "arrhythmia", "ecg", "palpitations"],
        "neurology": ["neurological", "seizure", "stroke", "headache", "numbness", "weakness"],
        "orthopedics": ["fracture", "bone", "joint", "musculoskeletal", "back pain"],
        "gastroenterology": ["gi", "abdominal", "liver", "digestive", "nausea", "vomiting"],
        "pulmonology": ["respiratory", "lung", "breathing", "pulmonary", "cough", "asthma"],
        "psychiatry": ["mental health", "depression", "anxiety", "psychiatric", "suicidal"]
    }
    
    # Convert clinical picture to searchable text
    clinical_text = str(clinical_picture).lower()
    differential_text = str(differential).lower()
    combined_text = clinical_text + " " + differential_text
    
    recommendations = []
    
    for specialty, keywords in specialist_keywords.items():
        if any(kw in combined_text for kw in keywords):
            recommendations.append({
                "specialty": specialty,
                "reason": f"Clinical indicators suggest {specialty} involvement",
                "urgency": "urgent" if any(rf in combined_text for rf in ["emergency", "severe", "acute"]) else "routine"
            })
    
    return {
        "referral_needed": len(recommendations) > 0,
        "recommendations": recommendations
    }


def format_diagnosis_result(
    patient_id: str,
    triage_reference: str,
    clinical_picture: Dict[str, Any],
    differential: List[Dict[str, Any]],
    primary_diagnosis: str,
    investigations: Dict[str, List[str]],
    specialist_need: Dict[str, Any],
    reasoning: str
) -> Dict[str, Any]:
    """
    Formats diagnosis result for handoff to Treatment Agent.
    """
    from datetime import datetime
    
    return {
        "diagnosis_id": f"DX-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "patient_id": patient_id,
        "triage_reference": triage_reference,
        "clinical_picture": clinical_picture,
        "differential_diagnoses": differential,
        "primary_diagnosis": primary_diagnosis,
        "investigations": investigations,
        "specialist_referral": specialist_need,
        "clinical_reasoning": reasoning,
        "timestamp": datetime.now().isoformat(),
        "status": "completed",
        "next_agent": "treatment_agent",
        "disclaimer": "AI-assisted analysis - requires physician review"
    }


# =========================================================
# BIOGPT INTEGRATION FOR DIAGNOSIS
# =========================================================

async def generate_differential_with_biogpt(
    clinical_picture: Dict[str, Any],
    biogpt_wrapper
) -> str:
    """
    Uses BioGPT to generate differential diagnosis.
    
    The fine-tuned medical model provides domain expertise
    for diagnostic reasoning.
    """
    prompt = f"""KEYWORDS: differential diagnosis, clinical assessment

Patient Presentation:
- Chief Complaint: {clinical_picture.get('chief_complaint', 'Unknown')}
- Age: {clinical_picture.get('demographics', {}).get('age', 'Unknown')}
- Gender: {clinical_picture.get('demographics', {}).get('gender', 'Unknown')}
- Medical History: {clinical_picture.get('medical_history', 'None')}
- Current Medications: {clinical_picture.get('current_medications', 'None')}

Based on this clinical presentation, provide:
1. Top 3-5 differential diagnoses with probability
2. Key supporting and opposing evidence for each
3. Recommended diagnostic workup
4. Red flags to watch for

TRANSCRIPTION:"""
    
    response = await biogpt_wrapper.agenerate(prompt)
    return response
