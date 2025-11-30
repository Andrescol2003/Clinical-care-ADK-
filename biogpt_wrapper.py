"""
BioGPT Model Integration for Google ADK

This wrapper matches YOUR training code from LLM_model.ipynb
Uses the same loading pattern and generation settings.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, Dict, Optional
import asyncio


class BioGPTModel:
    """
    BioGPT wrapper that matches your training notebook.
    
    This uses the SAME code pattern from your LLM_model.ipynb:
    - AutoTokenizer.from_pretrained()
    - AutoModelForCausalLM.from_pretrained()
    - model.generate() with your settings
    """
    
    def __init__(
        self,
        model_path: str = "./biogpt_medical_finetuned",
        device: str = None
    ):
        """
        Initialize BioGPT model.
        
        Args:
            model_path: Path to your fine-tuned model folder
                       (the biogpt_medical_finetuned folder)
            device: 'cuda', 'mps', or 'cpu' (auto-detect if None)
        """
        self.model_path = model_path
        
        # Device detection (same as your notebook)
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        self.tokenizer = None
        self.model = None
        self._is_loaded = False
    
    def load_model(self):
        """
        Load the model and tokenizer.
        
        This matches your test_generation() function in LLM_model.ipynb
        """
        if self._is_loaded:
            return
        
        print(f"Loading BioGPT from: {self.model_path}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer (same as your code)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model (same as your code)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self._is_loaded = True
        print("✅ BioGPT model loaded successfully!")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text from a prompt.
        
        Uses the SAME generation settings from your test_generation() function:
        - max_length=300
        - temperature=0.8
        - top_p=0.9
        - do_sample=True
        
        Args:
            prompt: The prompt text (e.g., "KEYWORDS: ...\\n\\nTRANSCRIPTION:")
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Top-p sampling
            
        Returns:
            Generated text
        """
        if not self._is_loaded:
            self.load_model()
        
        # Tokenize (same as your code)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate (same settings as your test_generation function)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """
        Async version for Google ADK compatibility.
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.generate(prompt, **kwargs)
        )
        return result
    
    def generate_medical_transcription(
        self,
        keywords: str,
        specialty: str = None
    ) -> str:
        """
        Generate a medical transcription using your training format.
        
        This creates the prompt in the SAME format you trained on:
        "KEYWORDS: ...\\n\\nTRANSCRIPTION:"
        
        Args:
            keywords: Medical keywords (e.g., "chest pain, shortness of breath")
            specialty: Optional specialty (e.g., "Cardiology")
            
        Returns:
            Generated medical transcription
        """
        if specialty:
            prompt = f"KEYWORDS: {keywords}, {specialty}\n\nTRANSCRIPTION:"
        else:
            prompt = f"KEYWORDS: {keywords}\n\nTRANSCRIPTION:"
        
        return self.generate(prompt)
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded


class MockBioGPT:
    """
    Mock BioGPT for testing without loading the real model.
    Useful for development and testing the agent system.
    """
    
    def __init__(self):
        self._is_loaded = True
        print("Using MockBioGPT (no real model loaded)")
    
    def load_model(self):
        pass
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Returns mock medical transcription."""
        
        # Extract keywords from prompt
        if "KEYWORDS:" in prompt:
            keywords = prompt.split("KEYWORDS:")[1].split("\n")[0].strip()
        else:
            keywords = "general symptoms"
        
        return f"""KEYWORDS: {keywords}

TRANSCRIPTION:
The patient presents with {keywords}. Initial assessment was performed.

SUBJECTIVE:
Patient reports symptoms consistent with the presenting complaint. 
Duration and severity were noted.

OBJECTIVE:
Vital signs within normal limits. Physical examination performed.
Relevant findings documented.

ASSESSMENT:
Clinical presentation consistent with the reported symptoms.
Differential diagnosis considered.

PLAN:
1. Continue current management
2. Follow-up as scheduled
3. Return if symptoms worsen

Patient educated on warning signs. Questions answered.
Follow-up appointment scheduled."""
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)
    
    def generate_medical_transcription(self, keywords: str, specialty: str = None) -> str:
        prompt = f"KEYWORDS: {keywords}\n\nTRANSCRIPTION:"
        return self.generate(prompt)
    
    def is_loaded(self) -> bool:
        return True


def load_biogpt(
    model_path: str = "./biogpt_medical_finetuned",
    use_mock: bool = False
):
    """
    Load BioGPT model.
    
    Args:
        model_path: Path to your fine-tuned model folder
        use_mock: If True, use mock model for testing
        
    Returns:
        BioGPTModel or MockBioGPT instance
        
    Example:
        # Load your fine-tuned model
        model = load_biogpt("./biogpt_medical_finetuned")
        
        # Generate
        text = model.generate("KEYWORDS: chest pain\\n\\nTRANSCRIPTION:")
        
        # Or use the helper method
        text = model.generate_medical_transcription("chest pain, shortness of breath")
    """
    if use_mock:
        return MockBioGPT()
    
    model = BioGPTModel(model_path=model_path)
    model.load_model()
    return model


# Backwards compatibility alias
create_biogpt_wrapper = load_biogpt
BioGPTWrapper = BioGPTModel
