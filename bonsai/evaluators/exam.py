"""
The Proof Machine: Blind Holdout Exam & Verification Engine for Bonsai.
Evaluates the Untrained Base Model vs. the Newly Cultivated Bonsai Model on unseen exam.jsonl.
Features strict weight isolation and memory hygiene.
"""

from dataclasses import dataclass
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from rich.console import Console

console = Console(force_terminal=True, legacy_windows=False)

@dataclass
class ExamSampleResult:
    prompt: str
    expected_output: str
    base_output: str
    bonsai_output: str
    base_valid_json: bool
    bonsai_valid_json: bool
    base_field_match_pct: float
    bonsai_field_match_pct: float
    base_latency_ms: float
    bonsai_latency_ms: float
    base_tokens_sec: float
    bonsai_tokens_sec: float

@dataclass
class ExamScorecard:
    base_model_name: str
    bonsai_model_name: str
    total_exam_questions: int
    base_schema_validity_pct: float
    bonsai_schema_validity_pct: float
    base_task_accuracy_pct: float
    bonsai_task_accuracy_pct: float
    base_hallucination_pct: float
    bonsai_hallucination_pct: float
    base_avg_latency_ms: float
    bonsai_avg_latency_ms: float
    base_tokens_sec: float
    bonsai_tokens_sec: float
    peak_vram_mb: float
    sample_comparisons: List[ExamSampleResult]

class BonsaiExam:
    def __init__(self, base_model_name: str, adapter_path: str, device: str = "cuda"):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.device = device

    def run_exam(self, exam_path: str) -> ExamScorecard:
        """Runs the unseen exam.jsonl holdout set across both models with strict weight isolation."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        target_device = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
        dtype = torch.float16 if target_device == "cuda" else torch.float32

        # 1. Load exam questions
        questions: List[Dict[str, Any]] = []
        with open(exam_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))

        if not questions:
            raise ValueError(f"No questions found in {exam_path}")

        console.print(f"[bold cyan]🔍 [Bonsai Proof Machine][/bold cyan] Loading Model: [yellow]{self.base_model_name}[/yellow]")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_raw = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(target_device)

        # Attach LoRA adapter
        console.print(f"[bold green]🌿 [Bonsai Proof Machine][/bold green] Attaching Cultivated Adapter: [cyan]{self.adapter_path}[/cyan]")
        model = PeftModel.from_pretrained(base_raw, self.adapter_path).to(target_device)
        model.eval()

        if target_device == "cuda":
            torch.cuda.empty_cache()

        # Step A: Evaluate Base Model by disabling adapter (zero contamination guaranteed)
        console.print(f"[yellow]Evaluating Baseline on {len(questions)} blind holdout questions...[/yellow]")
        base_results = []
        with model.disable_adapter():
            for q in questions:
                res = self._generate_response(model, tokenizer, q, target_device)
                base_results.append(res)

        if target_device == "cuda":
            torch.cuda.empty_cache()

        # Step B: Evaluate Bonsai Cultivated Model with adapter enabled
        console.print(f"[green]Evaluating Bonsai Cultivated Model on {len(questions)} blind holdout questions...[/green]")
        bonsai_results = []
        for q in questions:
            res = self._generate_response(model, tokenizer, q, target_device)
            bonsai_results.append(res)

        # Step C: Compile and grade results
        sample_results: List[ExamSampleResult] = []
        for i, q in enumerate(questions):
            expected = ""
            user_prompt = ""
            for msg in q.get("messages", []):
                if msg["role"] == "user":
                    user_prompt = msg["content"]
                elif msg["role"] == "assistant":
                    expected = msg["content"]

            b_out, b_lat, b_tok = base_results[i]
            p_out, p_lat, p_tok = bonsai_results[i]

            b_valid, b_match = self._evaluate_json_payload(b_out, expected)
            p_valid, p_match = self._evaluate_json_payload(p_out, expected)

            sample_results.append(ExamSampleResult(
                prompt=user_prompt,
                expected_output=expected,
                base_output=b_out,
                bonsai_output=p_out,
                base_valid_json=b_valid,
                bonsai_valid_json=p_valid,
                base_field_match_pct=b_match,
                bonsai_field_match_pct=p_match,
                base_latency_ms=b_lat,
                bonsai_latency_ms=p_lat,
                base_tokens_sec=b_tok,
                bonsai_tokens_sec=p_tok
            ))

        total = len(sample_results)
        base_valid_count = sum(1 for s in sample_results if s.base_valid_json)
        bonsai_valid_count = sum(1 for s in sample_results if s.bonsai_valid_json)

        base_schema_pct = round((base_valid_count / total) * 100, 1)
        bonsai_schema_pct = round((bonsai_valid_count / total) * 100, 1)

        base_accuracy_pct = round(sum(s.base_field_match_pct for s in sample_results) / total, 1)
        bonsai_accuracy_pct = round(sum(s.bonsai_field_match_pct for s in sample_results) / total, 1)

        base_hallucination_pct = round(100.0 - base_schema_pct, 1)
        bonsai_hallucination_pct = round(100.0 - bonsai_schema_pct, 1)

        base_avg_lat = round(sum(s.base_latency_ms for s in sample_results) / total, 1)
        bonsai_avg_lat = round(sum(s.bonsai_latency_ms for s in sample_results) / total, 1)

        base_avg_tok = round(sum(s.base_tokens_sec for s in sample_results) / total, 1)
        bonsai_avg_tok = round(sum(s.bonsai_tokens_sec for s in sample_results) / total, 1)

        peak_vram = 0.0
        if target_device == "cuda":
            peak_vram = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
            torch.cuda.empty_cache()

        return ExamScorecard(
            base_model_name=self.base_model_name,
            bonsai_model_name="Bonsai Cultivated Micro-AI",
            total_exam_questions=total,
            base_schema_validity_pct=base_schema_pct,
            bonsai_schema_validity_pct=bonsai_schema_pct,
            base_task_accuracy_pct=base_accuracy_pct,
            bonsai_task_accuracy_pct=bonsai_accuracy_pct,
            base_hallucination_pct=base_hallucination_pct,
            bonsai_hallucination_pct=bonsai_hallucination_pct,
            base_avg_latency_ms=base_avg_lat,
            bonsai_avg_latency_ms=bonsai_avg_lat,
            base_tokens_sec=base_avg_tok,
            bonsai_tokens_sec=bonsai_avg_tok,
            peak_vram_mb=peak_vram,
            sample_comparisons=sample_results
        )

    def _generate_response(self, model, tokenizer, question: Dict[str, Any], device: str) -> Tuple[str, float, float]:
        """Runs single prompt inference and tracks latency and throughput."""
        import torch

        # Include system and user messages
        input_messages = [m for m in question.get("messages", []) if m["role"] != "assistant"]
        text = tokenizer.apply_chat_template(input_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        start = time.perf_counter()
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.perf_counter() - start

        prompt_len = inputs["input_ids"].shape[1]
        gen_tokens = output_tokens[0][prompt_len:]
        num_tokens = len(gen_tokens)
        response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        latency_ms = round(elapsed * 1000, 1)
        tokens_sec = round(num_tokens / elapsed, 1) if elapsed > 0 else 0.0

        return response_text, latency_ms, tokens_sec

    def _evaluate_json_payload(self, output: str, expected_json: str) -> Tuple[bool, float]:
        """Evaluates whether the output is valid JSON and calculates schema match score."""
        expected_obj = {}
        try:
            expected_obj = json.loads(expected_json)
        except Exception:
            pass

        parsed = None
        is_valid = False
        try:
            parsed = json.loads(output)
            is_valid = True
        except Exception:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', output)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                    is_valid = True
                except Exception:
                    pass

        if not is_valid or not isinstance(parsed, dict):
            return False, 0.0

        if not expected_obj:
            return True, 100.0

        expected_keys = set(expected_obj.keys())
        parsed_keys = set(parsed.keys())
        if not expected_keys:
            return True, 100.0

        matched_keys = expected_keys.intersection(parsed_keys)
        key_score = (len(matched_keys) / len(expected_keys)) * 50.0

        val_matches = 0
        for k in matched_keys:
            if str(parsed[k]).strip().lower() == str(expected_obj[k]).strip().lower():
                val_matches += 1
        val_score = (val_matches / len(expected_keys)) * 50.0

        total_match = round(key_score + val_score, 1)
        return True, total_match
