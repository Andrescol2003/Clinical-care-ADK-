"""
Clinical Care Orchestrator - Google ADK Implementation

KEY CONCEPT: ADK RUNNER & MULTI-AGENT ORCHESTRATION
===================================================
Google ADK provides InMemoryRunner for agent execution.
This orchestrator coordinates multiple agents through:
1. Sequential workflows (Triage -> Diagnosis -> Treatment)
2. Parallel execution (Documentation + Scheduling)
3. Loop patterns (Follow-up monitoring)
4. Session management for conversation context

This is the main entry point for the clinical care system.
"""

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.sessions import Session
from google.genai import types
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import asyncio

# Import agent creators
from agents import (
    create_triage_agent,
    create_diagnosis_agent,
    create_treatment_agent,
    create_scheduling_agent,
    create_documentation_agent,
    create_followup_agent,
    # Utilities
    check_red_flags,
    format_triage_result,
    build_clinical_picture,
    run_safety_checks,
    aggregate_clinical_data,
    generate_soap_note,
    initialize_monitoring
)

# Import BioGPT wrapper
from models.biogpt_wrapper import BioGPTWrapper, create_biogpt_wrapper


class ClinicalCareOrchestrator:
    """
    Main orchestrator for the Clinical Care Coordination System.
    
    KEY CONCEPT: CENTRALIZED COORDINATION WITH ADK
    ==============================================
    The orchestrator:
    - Creates and manages all agents
    - Routes patient data through workflows
    - Manages sessions for conversation context
    - Integrates BioGPT for medical reasoning
    - Provides observability and logging
    """
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        biogpt_path: str = None,
        use_mock_biogpt: bool = True
    ):
        """
        Initialize the orchestrator.
        
        Args:
            model: Gemini model to use for agents
            biogpt_path: Path to fine-tuned BioGPT model
            use_mock_biogpt: Use mock BioGPT for testing
        """
        self.model = model
        
        # Initialize BioGPT
        self.biogpt = create_biogpt_wrapper(
            model_path=biogpt_path,
            use_mock=use_mock_biogpt
        )
        
        # Initialize agents
        self.agents: Dict[str, Agent] = {}
        self._initialize_agents()
        
        # Initialize runners for each agent
        self.runners: Dict[str, InMemoryRunner] = {}
        self._initialize_runners()
        
        # Session management
        self.sessions: Dict[str, Session] = {}
        
        # Workflow tracking
        self.workflow_history: List[Dict[str, Any]] = []
        
        print(f"Clinical Care Orchestrator initialized with {len(self.agents)} agents")
    
    def _initialize_agents(self) -> None:
        """Initialize all department agents."""
        self.agents["triage"] = create_triage_agent(model=self.model)
        self.agents["diagnosis"] = create_diagnosis_agent(model=self.model)
        self.agents["treatment"] = create_treatment_agent(model=self.model)
        self.agents["scheduling"] = create_scheduling_agent(model=self.model)
        self.agents["documentation"] = create_documentation_agent(model=self.model)
        self.agents["followup"] = create_followup_agent(model=self.model)
    
    def _initialize_runners(self) -> None:
        """
        Initialize InMemoryRunner for each agent.
        
        KEY CONCEPT: ADK RUNNERS
        ========================
        Runners handle:
        - Agent execution
        - Session management
        - Tool calling
        - Response streaming
        """
        for name, agent in self.agents.items():
            self.runners[name] = InMemoryRunner(agent=agent)
    
    # =========================================================
    # WORKFLOW EXECUTION
    # =========================================================
    
    async def run_full_workflow(
        self,
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run the complete patient care workflow.
        
        KEY CONCEPT: SEQUENTIAL + PARALLEL + LOOP
        =========================================
        1. SEQUENTIAL: Triage -> Diagnosis -> Treatment
        2. PARALLEL: After Treatment, run Doc + Scheduling together
        3. LOOP: Initialize follow-up monitoring
        
        Args:
            patient_data: Patient information including symptoms
            
        Returns:
            Complete workflow result with all agent outputs
        """
        workflow_id = f"WF-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        patient_id = patient_data.get("patient_id", "unknown")
        
        result = {
            "workflow_id": workflow_id,
            "patient_id": patient_id,
            "started_at": datetime.now().isoformat(),
            "steps": {},
            "status": "running"
        }
        
        try:
            # ==========================================
            # STEP 1: TRIAGE (Sequential)
            # ==========================================
            print(f"\n{'='*50}")
            print("STEP 1: TRIAGE")
            print(f"{'='*50}")
            
            triage_result = await self._run_triage(patient_data)
            result["steps"]["triage"] = triage_result
            
            # Check for emergency
            if triage_result.get("urgency_level", 5) == 1:
                result["status"] = "emergency_escalation"
                print("⚠️ EMERGENCY - Escalating immediately")
                return result
            
            # ==========================================
            # STEP 2: DIAGNOSIS (Sequential)
            # ==========================================
            print(f"\n{'='*50}")
            print("STEP 2: DIAGNOSIS")
            print(f"{'='*50}")
            
            diagnosis_input = {**patient_data, **triage_result}
            diagnosis_result = await self._run_diagnosis(diagnosis_input)
            result["steps"]["diagnosis"] = diagnosis_result
            
            # ==========================================
            # STEP 3: TREATMENT (Sequential)
            # ==========================================
            print(f"\n{'='*50}")
            print("STEP 3: TREATMENT PLANNING")
            print(f"{'='*50}")
            
            treatment_input = {**diagnosis_input, **diagnosis_result}
            treatment_result = await self._run_treatment(treatment_input)
            result["steps"]["treatment"] = treatment_result
            
            # Check for safety hold
            if treatment_result.get("status") == "safety_hold":
                result["status"] = "safety_review_required"
                print("⚠️ SAFETY HOLD - Requires physician review")
                return result
            
            # ==========================================
            # STEP 4: PARALLEL - Documentation + Scheduling
            # ==========================================
            print(f"\n{'='*50}")
            print("STEP 4: DOCUMENTATION & SCHEDULING (Parallel)")
            print(f"{'='*50}")
            
            parallel_input = {**treatment_input, **treatment_result}
            
            # Run in parallel
            doc_task = self._run_documentation(parallel_input)
            sched_task = self._run_scheduling(parallel_input)
            
            doc_result, sched_result = await asyncio.gather(doc_task, sched_task)
            
            result["steps"]["documentation"] = doc_result
            result["steps"]["scheduling"] = sched_result
            
            # ==========================================
            # STEP 5: FOLLOW-UP SETUP (Loop Initialization)
            # ==========================================
            print(f"\n{'='*50}")
            print("STEP 5: FOLLOW-UP MONITORING SETUP")
            print(f"{'='*50}")
            
            followup_result = await self._setup_followup(
                patient_id=patient_id,
                treatment_result=treatment_result,
                scheduling_result=sched_result
            )
            result["steps"]["followup"] = followup_result
            
            # ==========================================
            # WORKFLOW COMPLETE
            # ==========================================
            result["status"] = "completed"
            result["completed_at"] = datetime.now().isoformat()
            
            # Calculate duration
            started = datetime.fromisoformat(result["started_at"])
            completed = datetime.fromisoformat(result["completed_at"])
            result["duration_seconds"] = (completed - started).total_seconds()
            
            print(f"\n{'='*50}")
            print(f"✅ WORKFLOW COMPLETE - {result['duration_seconds']:.1f}s")
            print(f"{'='*50}")
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"\n❌ WORKFLOW ERROR: {e}")
        
        # Store in history
        self.workflow_history.append(result)
        
        return result
    
    # =========================================================
    # INDIVIDUAL AGENT EXECUTION
    # =========================================================
    
    async def _run_triage(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run triage agent."""
        patient_id = patient_data.get("patient_id", "unknown")
        symptoms = patient_data.get("symptoms", "")
        
        # Quick red flag check
        red_flags = check_red_flags(symptoms)
        
        # Create session for this patient
        session_id = f"triage_{patient_id}"
        
        # Build prompt for agent
        prompt = f"""
Please triage the following patient:

Patient ID: {patient_id}
Symptoms: {symptoms}
Age: {patient_data.get('age', 'Unknown')}
Gender: {patient_data.get('gender', 'Unknown')}
Medical History: {patient_data.get('medical_history', 'None provided')}
Current Medications: {patient_data.get('medications', 'None')}
Allergies: {patient_data.get('allergies', 'None')}

{"⚠️ RED FLAGS DETECTED: " + ', '.join(red_flags['red_flags_found']) if red_flags['has_red_flags'] else ''}

Please provide your triage assessment.
"""
        
        # Run agent
        response = await self._execute_agent("triage", session_id, prompt)
        
        # Format result
        urgency = red_flags.get("recommended_urgency", 3)  # Default to urgent if red flags
        if not red_flags["has_red_flags"]:
            urgency = 4  # Default to less urgent
        
        return format_triage_result(
            patient_id=patient_id,
            symptoms=symptoms,
            urgency=urgency,
            assessment=response,
            red_flags=red_flags.get("red_flags_found", [])
        )
    
    async def _run_diagnosis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run diagnosis agent."""
        patient_id = input_data.get("patient_id", "unknown")
        session_id = f"diagnosis_{patient_id}"
        
        # Build clinical picture
        clinical_picture = build_clinical_picture(input_data, input_data)
        
        # Use BioGPT for medical reasoning
        biogpt_assessment = ""
        if self.biogpt:
            biogpt_prompt = f"Analyze symptoms: {input_data.get('chief_complaint', input_data.get('symptoms', ''))}"
            biogpt_assessment = self.biogpt.generate(biogpt_prompt)
        
        prompt = f"""
Please analyze this patient and provide differential diagnosis:

TRIAGE SUMMARY:
- Chief Complaint: {input_data.get('chief_complaint', input_data.get('symptoms', 'Unknown'))}
- Urgency Level: {input_data.get('urgency_level', 'Unknown')}
- Triage Assessment: {input_data.get('assessment', 'See above')}

PATIENT CONTEXT:
{clinical_picture}

MEDICAL AI ANALYSIS:
{biogpt_assessment}

Please provide your differential diagnosis and recommendations.
"""
        
        response = await self._execute_agent("diagnosis", session_id, prompt)
        
        return {
            "diagnosis_id": f"DX-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "patient_id": patient_id,
            "clinical_picture": clinical_picture,
            "agent_analysis": response,
            "biogpt_assessment": biogpt_assessment,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _run_treatment(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run treatment agent."""
        patient_id = input_data.get("patient_id", "unknown")
        session_id = f"treatment_{patient_id}"
        
        # Run safety checks
        proposed_meds = []  # Would extract from diagnosis
        patient_info = {
            "allergies": input_data.get("allergies", []),
            "current_medications": input_data.get("medications", []),
            "age": input_data.get("age")
        }
        safety_result = run_safety_checks(proposed_meds, patient_info)
        
        prompt = f"""
Please create a treatment plan for this patient:

DIAGNOSIS:
{input_data.get('agent_analysis', 'See diagnosis')}

PATIENT CONTEXT:
- Patient ID: {patient_id}
- Allergies: {input_data.get('allergies', 'None')}
- Current Medications: {input_data.get('medications', 'None')}
- Medical History: {input_data.get('medical_history', 'None')}

SAFETY CHECK RESULTS:
{safety_result}

Please provide a comprehensive treatment plan.
"""
        
        response = await self._execute_agent("treatment", session_id, prompt)
        
        return {
            "treatment_id": f"TX-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "patient_id": patient_id,
            "treatment_plan": response,
            "safety_checks": safety_result,
            "status": "completed" if safety_result["passed"] else "safety_hold",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _run_documentation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run documentation agent."""
        patient_id = input_data.get("patient_id", "unknown")
        session_id = f"documentation_{patient_id}"
        
        # Aggregate all clinical data
        aggregated = aggregate_clinical_data(
            triage_result=input_data,
            diagnosis_result=input_data,
            treatment_result=input_data
        )
        
        # Generate SOAP note
        soap_note = generate_soap_note(aggregated)
        
        prompt = f"""
Please review and enhance this clinical documentation:

{soap_note}

Ensure it is complete, accurate, and follows documentation standards.
"""
        
        response = await self._execute_agent("documentation", session_id, prompt)
        
        return {
            "document_id": f"DOC-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "patient_id": patient_id,
            "soap_note": soap_note,
            "agent_review": response,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _run_scheduling(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run scheduling agent."""
        patient_id = input_data.get("patient_id", "unknown")
        session_id = f"scheduling_{patient_id}"
        
        prompt = f"""
Please schedule appointments for this patient:

Patient ID: {patient_id}
Treatment Plan: {input_data.get('treatment_plan', 'See treatment')}

Schedule:
1. Follow-up appointment (2 weeks)
2. Any specialist referrals needed
3. Lab work if ordered

Please provide scheduling confirmation.
"""
        
        response = await self._execute_agent("scheduling", session_id, prompt)
        
        return {
            "scheduling_id": f"SCH-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "patient_id": patient_id,
            "scheduling_summary": response,
            "confirmed_appointments": [],  # Would be populated by tool calls
            "timestamp": datetime.now().isoformat()
        }
    
    async def _setup_followup(
        self,
        patient_id: str,
        treatment_result: Dict[str, Any],
        scheduling_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Initialize follow-up monitoring."""
        return initialize_monitoring(
            patient_id=patient_id,
            treatment_result=treatment_result,
            scheduling_result=scheduling_result
        )
    
    async def _execute_agent(
        self,
        agent_name: str,
        session_id: str,
        prompt: str
    ) -> str:
        """
        Execute an agent with the given prompt.
        
        KEY CONCEPT: ADK RUNNER EXECUTION
        =================================
        The runner handles:
        - Creating/resuming sessions
        - Executing the agent
        - Managing tool calls
        - Returning responses
        """
        runner = self.runners.get(agent_name)
        if not runner:
            return f"Agent {agent_name} not found"
        
        try:
            # Execute agent
            response = await runner.run(
                session_id=session_id,
                prompt=prompt
            )
            
            # Extract text response
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'content'):
                return str(response.content)
            else:
                return str(response)
                
        except Exception as e:
            return f"Error executing {agent_name}: {str(e)}"
    
    # =========================================================
    # MONITORING LOOP
    # =========================================================
    
    async def run_monitoring_cycle(self) -> Dict[str, Any]:
        """
        Run monitoring cycle for all active patients.
        
        KEY CONCEPT: LOOP PATTERN
        =========================
        This would be called periodically (e.g., daily)
        to check on all patients in monitoring.
        """
        from agents.followup_agent import monitoring_state, run_monitoring_check
        
        due_patients = monitoring_state.get_due_for_check()
        results = []
        
        for patient_id in due_patients:
            result = run_monitoring_check(patient_id)
            results.append(result)
        
        return {
            "cycle_timestamp": datetime.now().isoformat(),
            "patients_checked": len(results),
            "results": results
        }
    
    # =========================================================
    # OBSERVABILITY
    # =========================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "agents_initialized": list(self.agents.keys()),
            "runners_active": list(self.runners.keys()),
            "workflows_completed": len(self.workflow_history),
            "biogpt_loaded": self.biogpt.is_loaded() if self.biogpt else False,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_workflow_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent workflow history."""
        return self.workflow_history[-limit:]


# =========================================================
# CONVENIENCE FUNCTIONS
# =========================================================

def create_orchestrator(
    use_mock_biogpt: bool = True,
    biogpt_path: str = None
) -> ClinicalCareOrchestrator:
    """
    Factory function to create an orchestrator.
    
    Args:
        use_mock_biogpt: Use mock BioGPT for testing
        biogpt_path: Path to fine-tuned BioGPT model
        
    Returns:
        Configured ClinicalCareOrchestrator
    """
    return ClinicalCareOrchestrator(
        use_mock_biogpt=use_mock_biogpt,
        biogpt_path=biogpt_path
    )


async def quick_triage(
    symptoms: str,
    patient_id: str = "QUICK"
) -> Dict[str, Any]:
    """
    Quick triage without full workflow.
    
    Useful for testing or simple assessments.
    """
    orchestrator = create_orchestrator(use_mock_biogpt=True)
    
    return await orchestrator._run_triage({
        "patient_id": patient_id,
        "symptoms": symptoms
    })
