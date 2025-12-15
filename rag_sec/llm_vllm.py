from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import json

try:
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover
    LLM = None
    SamplingParams = None

@dataclass
class VLLMGenerator:
    model: str
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    trust_remote_code: bool = True

    def __post_init__(self):
        model_name = os.environ.get("RAG_LLM_MODEL", self.model)
        self.model = model_name
        if LLM is None:
            raise RuntimeError(
                "vLLM is not installed. Install with: pip install -r requirements-gpu-vllm.txt"
            )
        self.llm = LLM(
            model=self.model,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=self.trust_remote_code,
        )

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        # Many instruct models follow ChatML-style; we provide a simple concatenation that works broadly.
        prompt = f"""<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"""
        params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )
        out = self.llm.generate([prompt], params)[0].outputs[0].text.strip()

        # Attempt to extract the first JSON object in the output
        start = out.find("{")
        end = out.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Model did not return JSON. Raw output: {out[:500]}")
        js = out[start:end+1]
        return json.loads(js)
