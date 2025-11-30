"""
Triage Agent - Google ADK Implementation

KEY CONCEPT: GOOGLE ADK AGENT STRUCTURE
=======================================
In Google ADK, an Agent is created with:
- name: Unique identifier
- model: The LLM to use (we'll use Gemini + BioGPT hybrid)
- instruction: System prompt defining behavior
- tools: List of callable functions

The Agent handles:
- Conversation management
- Tool calling
- Response generation

We add our BioGPT integration for medical-specific reasoning.
"""

from google.adk.agents import Agent
from google.adk.tools import google_search
from typing import Any, Dict, List, Optional
import json

# Import our custom tools
from tools.clinical_tools import (
    assess_symptoms,
    lookup_patient,
    send_alert,
    get_triage_tools
)


# =========================================================
# TRIAGE AGENT CONFIGURATION
# =========================================================

TRIAGE_INSTRUCTION = """You are an experienced triage nurse AI assistant in a clinical care coordination system.

## YOUR ROLE
You are the FIRST point of contact for patients entering the clinical care system. Your job is to:
1. Assess patient symptoms quickly and accurately
2. Determine urgency level (1-5 scale)
3. Identify any RED FLAGS requiring immediate attention
4. Gather essential information for the diagnosis team
5. Route patients appropriately

## URGENCY LEVELS
- Level 1 (IMMEDIATE): Life-threatening - needs attention NOW
  Examples: chest pain with shortness of breath, severe bleeding, unconsciousness
- Level 2 (EMERGENCY): Severe condition requiring rapid intervention
  Examples: severe pain, high fever with altered mental status
- Level 3 (URGENT): Serious but stable, needs attention within 30 min
  Examples: moderate pain, persistent vomiting, concerning symptoms
- Level 4 (LESS URGENT): Can wait 1-2 hours safely
  Examples: minor injuries, mild symptoms, routine concerns
- Level 5 (NON-URGENT): Routine care, can be scheduled
  Examples: medication refills, minor complaints, follow-up questions

## RED FLAGS - ALWAYS ESCALATE
- Chest pain or pressure
- Difficulty breathing / shortness of breath
- Severe bleeding that won't stop
- Loss of consciousness or confusion
- Signs of stroke (face drooping, arm weakness, speech difficulty)
- Severe allergic reaction (swelling, difficulty breathing)
- Suicidal thoughts or self-harm
- Severe trauma or injury

## YOUR PROCESS
1. GREET the patient and acknowledge their concern
2. GATHER information about:
   - Chief complaint (main symptom/reason for visit)
   - Duration and progression
   - Pain level (1-10) if applicable
   - Associated symptoms
   - Relevant medical history
3. CHECK for red flags IMMEDIATELY
4. USE TOOLS when needed:
   - Use `assess_symptoms` for structured assessment
   - Use `lookup_patient` to get patient history
   - Use `send_alert` for urgent situations
5. DETERMINE urgency level
6. PROVIDE clear next steps

## OUTPUT FORMAT
After assessment, provide:
```
TRIAGE ASSESSMENT
================
Patient ID: [ID]
Chief Complaint: [Main symptom]
Urgency Level: [1-5] - [Description]

Assessment Summary:
[Your clinical assessment]

Red Flags: [None / List any found]

Recommended Action:
[Next steps - who should see this patient]

Notes for Diagnosis Team:
[Key information to pass along]
```

## IMPORTANT GUIDELINES
- When in doubt, assign HIGHER urgency (safety first)
- Be empathetic but efficient
- Don't diagnose - that's for the Diagnosis Agent
- Always document your reasoning
- If RED FLAGS present, alert immediately and assign Level 1-2

Remember: Your assessment directly impacts patient safety. Be thorough but quick.
"""


def create_triage_agent(
    model: str = "gemini-2.0-flash",
    custom_tools: List = None
) -> Agent:
    """
    Creates and configures the Triage Agent.
    
    KEY CONCEPT: AGENT FACTORY PATTERN
    ==================================
    Using a factory function allows:
    - Easy configuration changes
    - Testing with different models
    - Dependency injection of tools
    
    Args:
        model: The model to use (default: gemini-2.0-flash)
        custom_tools: Override default tools if needed
        
    Returns:
        Configured Agent instance
    """
    # Get tools for triage
    tools = custom_tools if custom_tools is not None else get_triage_tools()
    
    # Add google_search for looking up medical information
    tools.append(google_search)
    
    # Create the agent
    agent = Agent(
        name="triage_agent",
        model=model,
        instruction=TRIAGE_INSTRUCTION,
        tools=tools
    )
    
    return agent


# =========================================================
# TRIAGE-SPECIFIC FUNCTIONS
# =========================================================

def check_red_flags(symptoms: str) -> Dict[str, Any]:
    """
    Quick check for red flag symptoms.
    
    This is a fast-path check before full assessment.
    If red flags found, we can escalate immediately.
    """
    red_flags = [
        "chest pain", "chest pressure", "difficulty breathing",
        "shortness of breath", "severe bleeding", "unconscious",
        "loss of consciousness", "confusion", "stroke", 
        "face drooping", "arm weakness", "speech difficulty",
        "severe allergic", "anaphylaxis", "suicidal", "self-harm",
        "severe trauma", "not breathing", "no pulse"
    ]
    
    symptoms_lower = symptoms.lower()
    found_flags = [flag for flag in red_flags if flag in symptoms_lower]
    
    return {
        "has_red_flags": len(found_flags) > 0,
        "red_flags_found": found_flags,
        "recommended_urgency": 1 if found_flags else None
    }


def format_triage_result(
    patient_id: str,
    symptoms: str,
    urgency: int,
    assessment: str,
    red_flags: List[str] = None,
    notes: str = ""
) -> Dict[str, Any]:
    """
    Formats triage result for handoff to Diagnosis Agent.
    
    KEY CONCEPT: STRUCTURED HANDOFFS
    ================================
    When passing data between agents, use structured formats.
    This ensures the receiving agent has all needed information.
    """
    urgency_descriptions = {
        1: "IMMEDIATE - Life threatening",
        2: "EMERGENCY - Severe condition",
        3: "URGENT - Serious but stable",
        4: "LESS URGENT - Can wait",
        5: "NON-URGENT - Routine care"
    }
    
    return {
        "triage_id": f"TRG-{patient_id}-{__import__('datetime').datetime.now().strftime('%Y%m%d%H%M%S')}",
        "patient_id": patient_id,
        "chief_complaint": symptoms,
        "urgency_level": urgency,
        "urgency_description": urgency_descriptions.get(urgency, "Unknown"),
        "assessment": assessment,
        "red_flags": red_flags or [],
        "notes_for_diagnosis": notes,
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "status": "completed",
        "next_agent": "diagnosis_agent"
    }


# =========================================================
# CALLBACK FOR BIOGPT INTEGRATION
# =========================================================

async def triage_with_biogpt(
    symptoms: str,
    patient_context: Dict[str, Any],
    biogpt_wrapper
) -> str:
    """
    Generates triage assessment using BioGPT.
    
    KEY CONCEPT: HYBRID MODEL USAGE
    ===============================
    We use BioGPT for medical-specific reasoning,
    while Gemini (via ADK) handles conversation flow
    and tool orchestration.
    
    Args:
        symptoms: Patient symptoms
        patient_context: Additional patient info
        biogpt_wrapper: BioGPT model wrapper instance
        
    Returns:
        Medical assessment text from BioGPT
    """
    prompt = f"""KEYWORDS: triage assessment, {symptoms}

Patient Context:
- Age: {patient_context.get('age', 'Unknown')}
- Gender: {patient_context.get('gender', 'Unknown')}
- Medical History: {patient_context.get('medical_history', 'None provided')}

Provide a clinical triage assessment including:
1. Summary of presenting symptoms
2. Urgency assessment
3. Key concerns
4. Recommended next steps

TRANSCRIPTION:"""
    
    # Generate using BioGPT
    assessment = await biogpt_wrapper.agenerate(prompt)
    
    return assessment
