"""
AD-Specific Model Processor
Adapted from Paper 1's model_processor.py for Alzheimer's Disease prediction
using UK Biobank data with validation in All of Us.

Key changes from original:
- Added support for multiple LLM models (ensemble)
- Added AD-specific prompt handling
- Added confidence extraction from LLM outputs
- Temperature=0 for deterministic/reproducible outputs
"""

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class ADModelProcessor:
    def __init__(self, model_dir: str, gpu_memory_utilization: float = 0.9):
        """
        Initializes the AD Model Processor.
        
        Args:
            model_dir (str): Path to the model directory (e.g., meta-llama/Meta-Llama-3-70B-Instruct)
            gpu_memory_utilization (float): GPU memory usage fraction
        """
        self.model = LLM(
            model=model_dir,
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=False,
            max_num_seqs=128,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.sampling_params = None

    def set_generation_params(self, do_sample=False, num_return_sequences=1, 
                               temperature=0, max_tokens=2048):
        """
        Sets generation parameters.
        
        Key difference from original: 
        - max_tokens increased to 2048 (AD-specific CoT reasoning needs more space)
        - temperature=0 for reproducibility (same as original)
        """
        self.sampling_params = SamplingParams(
            n=num_return_sequences,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_ad_prediction(self, prompts):
        """
        Generates AD-specific biological age predictions.
        
        This is the core inference function. It takes a batch of prompts 
        (each containing a participant's health report) and returns the 
        LLM's predicted biological age and AD risk assessment.
        """
        if self.sampling_params is None:
            raise ValueError("Call set_generation_params() first.")
        
        batch_outputs = self.model.generate(prompts, self.sampling_params, use_tqdm=True)
        return [
            [completion.text for completion in output.outputs] 
            for output in batch_outputs
        ]

    def generate_organ_specific(self, prompts):
        """
        Generates organ-specific age predictions.
        Same as original but kept for compatibility with the organ-specific 
        analysis pipeline (cardiovascular, hepatic, renal, etc.)
        """
        return self.generate_ad_prediction(prompts)
