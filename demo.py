"""
Clinical Care Coordination System - Demo Script

This script demonstrates the complete clinical care workflow
using Google ADK agents with BioGPT integration.

REQUIREMENTS:
- pip install google-adk
- pip install transformers torch (for BioGPT)

USAGE:
    python demo.py
    
    # Or with real BioGPT model:
    python demo.py --biogpt-path /path/to/your/model
"""

import asyncio
import argparse
from datetime import datetime


async def run_demo(use_mock: bool = True, biogpt_path: str = None):
    """
    Run the clinical care demonstration.
    
    This shows the complete workflow from patient arrival
    through follow-up monitoring setup.
    """
    from orchestrator import create_orchestrator
    
    print("=" * 60)
    print("CLINICAL CARE COORDINATION SYSTEM - DEMO")
    print("Google ADK + BioGPT Integration")
    print("=" * 60)
    print()
    
    # Create the orchestrator
    print("Initializing system...")
    orchestrator = create_orchestrator(
        use_mock_biogpt=use_mock,
        biogpt_path=biogpt_path
    )
    
    # Check status
    status = orchestrator.get_status()
    print(f"✓ Agents initialized: {', '.join(status['agents_initialized'])}")
    print(f"✓ BioGPT loaded: {status['biogpt_loaded']}")
    print()
    
    # Sample patient data
    patient_data = {
        "patient_id": "P001",
        "symptoms": "Persistent headache for 3 days, mild fever, neck stiffness",
        "age": 35,
        "gender": "F",
        "medical_history": ["Migraines", "Hypertension"],
        "medications": ["Lisinopril 10mg daily"],
        "allergies": ["Penicillin"]
    }
    
    print("PATIENT PRESENTATION:")
    print("-" * 40)
    print(f"Patient ID: {patient_data['patient_id']}")
    print(f"Symptoms: {patient_data['symptoms']}")
    print(f"Age/Gender: {patient_data['age']}/{patient_data['gender']}")
    print(f"History: {', '.join(patient_data['medical_history'])}")
    print(f"Medications: {', '.join(patient_data['medications'])}")
    print(f"Allergies: {', '.join(patient_data['allergies'])}")
    print()
    
    # Run the full workflow
    print("Starting clinical care workflow...")
    print()
    
    result = await orchestrator.run_full_workflow(patient_data)
    
    # Display results
    print()
    print("=" * 60)
    print("WORKFLOW RESULTS")
    print("=" * 60)
    print()
    
    print(f"Workflow ID: {result['workflow_id']}")
    print(f"Status: {result['status']}")
    print(f"Duration: {result.get('duration_seconds', 'N/A')} seconds")
    print()
    
    # Show each step's result
    for step_name, step_result in result.get("steps", {}).items():
        print(f"\n--- {step_name.upper()} ---")
        if isinstance(step_result, dict):
            for key, value in step_result.items():
                if key not in ["agent_analysis", "treatment_plan", "soap_note", "scheduling_summary"]:
                    print(f"  {key}: {value}")
        else:
            print(f"  {step_result}")
    
    print()
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    
    return result


async def run_triage_only_demo():
    """Quick demo of just the triage agent."""
    from orchestrator import quick_triage
    
    print("QUICK TRIAGE DEMO")
    print("-" * 40)
    
    symptoms = "Severe chest pain radiating to left arm, shortness of breath, sweating"
    print(f"Symptoms: {symptoms}")
    print()
    
    result = await quick_triage(symptoms, patient_id="EMERGENCY_TEST")
    
    print("TRIAGE RESULT:")
    print(f"  Urgency Level: {result.get('urgency_level')} - {result.get('urgency_description')}")
    print(f"  Red Flags: {result.get('red_flags', [])}")
    print(f"  Next Agent: {result.get('next_agent')}")


async def demonstrate_patterns():
    """
    Demonstrate the different agent patterns:
    - Sequential
    - Parallel  
    - Loop
    """
    print("=" * 60)
    print("AGENT PATTERN DEMONSTRATIONS")
    print("=" * 60)
    
    # 1. Sequential Pattern
    print("\n1. SEQUENTIAL PATTERN")
    print("   Triage → Diagnosis → Treatment")
    print("   Each agent passes results to the next")
    
    # 2. Parallel Pattern
    print("\n2. PARALLEL PATTERN")
    print("   After Treatment, Documentation and Scheduling run simultaneously")
    print("   Uses asyncio.gather() for concurrent execution")
    
    # 3. Loop Pattern
    print("\n3. LOOP PATTERN")
    print("   Follow-up agent runs monitoring cycles")
    print("   Checks patients periodically, generates alerts")
    
    print("\nSee orchestrator.run_full_workflow() for implementation details")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clinical Care Coordination System Demo"
    )
    parser.add_argument(
        "--biogpt-path",
        type=str,
        default=None,
        help="Path to fine-tuned BioGPT model"
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        default=True,
        help="Use mock BioGPT (default: True)"
    )
    parser.add_argument(
        "--triage-only",
        action="store_true",
        help="Run only triage demo"
    )
    parser.add_argument(
        "--show-patterns",
        action="store_true",
        help="Show agent pattern explanations"
    )
    
    args = parser.parse_args()
    
    if args.show_patterns:
        asyncio.run(demonstrate_patterns())
    elif args.triage_only:
        asyncio.run(run_triage_only_demo())
    else:
        use_mock = args.use_mock if args.biogpt_path is None else False
        asyncio.run(run_demo(
            use_mock=use_mock,
            biogpt_path=args.biogpt_path
        ))


if __name__ == "__main__":
    main()
