"""
Teacher Engine for Bonsai.
Supports Local Ollama, Cloud APIs (Groq, OpenAI, Gemini, Anthropic),
and an Offline Self-Contained Programmatic Synthesizer for 100% zero-blocker execution.
"""

import json
import os
import re
from typing import List, Dict, Any, Optional
import requests
from bonsai.core.ingest import DocumentBlock
from bonsai.core.constants import DEFAULT_SYSTEM_DIRECTIVE

class CultivatedPair:
    def __init__(self, instruction: str, input_text: str, output_text: str, metadata: Dict[str, Any] = None):
        self.instruction = instruction.strip()
        self.input_text = input_text.strip()
        self.output_text = output_text.strip()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "input": self.input_text,
            "output": self.output_text,
            "metadata": self.metadata
        }

    def to_chat_format(self, system_directive: Optional[str] = DEFAULT_SYSTEM_DIRECTIVE) -> Dict[str, Any]:
        user_msg = f"{self.instruction}\n\nContext:\n{self.input_text}" if self.input_text else self.instruction
        messages = []
        if system_directive:
            messages.append({"role": "system", "content": system_directive})
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": self.output_text})
        return {"messages": messages}

class TeacherEngine:
    """Manages teacher synthesis across Local Ollama, Cloud APIs, or Local Programmatic Fallback."""

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.active_backend = self._detect_backend()

    def _detect_backend(self) -> str:
        # Check Cloud API Keys
        if os.getenv("GROQ_API_KEY"):
            return "groq"
        if os.getenv("GEMINI_API_KEY"):
            return "gemini"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"

        # Check local Ollama
        try:
            res = requests.get(f"{self.ollama_url}/api/tags", timeout=1.5)
            if res.status_code == 200:
                models = res.json().get("models", [])
                if models:
                    return f"ollama:{models[0]['name']}"
        except Exception:
            pass

        return "local_synthesizer"

    def extract_base_scenarios(self, blocks: List[DocumentBlock]) -> List[CultivatedPair]:
        """Extracts unique, un-augmented base scenario pairs from document blocks."""
        pairs: List[CultivatedPair] = []

        for block in blocks:
            # Check pre-formatted JSON / JSONL
            try:
                parsed = json.loads(block.content)
                if isinstance(parsed, dict) and "instruction" in parsed and "output" in parsed:
                    pairs.append(CultivatedPair(
                        instruction=parsed.get("instruction", ""),
                        input_text=parsed.get("input", ""),
                        output_text=parsed.get("output", "") if isinstance(parsed["output"], str) else json.dumps(parsed["output"]),
                        metadata={"source": block.source, "origin": "pre_formatted"}
                    ))
                    continue
            except Exception:
                pass

            # Detect Ops / Incident / Log patterns
            extracted = self._extract_log_incident_patterns(block)
            pairs.extend(extracted)

        return pairs

    def augment_training_scenarios(self, base_pairs: List[CultivatedPair], target_count: int = 24) -> List[CultivatedPair]:
        """Expands base training scenarios with diverse prompt templates while preserving the underlying context."""
        augmented: List[CultivatedPair] = list(base_pairs)
        if not base_pairs:
            return augmented

        instructions_templates = [
            "Extract structured operational telemetry from the provided log in strictly valid JSON format. Required keys: severity, service, event_type, impact_score, recommended_action.",
            "Parse this system event into an actionable JSON payload with severity, service, event_type, impact_score, and recommended_action.",
            "Convert the messy incident report below into a machine-readable JSON object matching the standard incident schema (severity, service, event_type, impact_score, recommended_action).",
            "Analyze the operational alert below. Respond with ONLY a parseable JSON object containing severity, service, event_type, impact_score, recommended_action."
        ]

        idx = 0
        while len(augmented) < target_count and idx < len(base_pairs) * len(instructions_templates):
            src_pair = base_pairs[idx % len(base_pairs)]
            tmpl = instructions_templates[(idx // len(base_pairs)) % len(instructions_templates)]
            augmented.append(CultivatedPair(
                instruction=tmpl,
                input_text=src_pair.input_text,
                output_text=src_pair.output_text,
                metadata={"augmented": True, "template_variant": idx % len(instructions_templates)}
            ))
            idx += 1

        return augmented

    def _extract_log_incident_patterns(self, block: DocumentBlock) -> List[CultivatedPair]:
        """Extracts and formats messy ops logs into strict schema extraction pairs."""
        results = []
        text = block.content

        # Split into distinct log entries
        entries = re.split(r'\n(?=\[\d{4}-\d{2}-\d{2}|\b(?:CRITICAL|ERROR|WARN|INFO|FATAL)\b)', text)
        if len(entries) <= 1:
            entries = [e.strip() for e in text.split("\n\n") if e.strip()]

        for entry in entries:
            entry = entry.strip()
            if not entry or len(entry) < 20:
                continue

            # Determine severity
            severity = "MEDIUM"
            if any(k in entry.upper() for k in ["CRITICAL", "FATAL", "PANIC", "OUTAGE", "EMERGENCY"]):
                severity = "CRITICAL"
            elif any(k in entry.upper() for k in ["ERROR", "FAILED", "DOWN", "TIMEOUT"]):
                severity = "HIGH"
            elif any(k in entry.upper() for k in ["WARN", "SLOW", "DEGRADED", "RETRIED"]):
                severity = "MEDIUM"
            else:
                severity = "LOW"

            # Service name extraction heuristic
            service = "core-service"
            for s in ["auth-api", "payment-gateway", "billing-worker", "db-cluster", "redis-cache", "worker-node", "ingress-controller", "order-service", "inventory-db"]:
                if s.lower() in entry.lower():
                    service = s
                    break
            else:
                match = re.search(r'([a-zA-Z0-9_\-]+(?:-service|-api|-worker|-node|-db|-cluster))', entry, re.IGNORECASE)
                if match:
                    service = match.group(1).lower()

            # Event type heuristic
            event_type = "system_anomaly"
            if "pool exhausted" in entry.lower() or "connection" in entry.lower():
                event_type = "db_pool_exhaustion"
            elif "oom" in entry.lower() or "memory" in entry.lower():
                event_type = "out_of_memory"
            elif "latency" in entry.lower() or "slow" in entry.lower() or "p99" in entry.lower():
                event_type = "latency_spike"
            elif "500" in entry or "503" in entry or "gateway timeout" in entry.lower():
                event_type = "gateway_error"
            elif "disk" in entry.lower() or "storage" in entry.lower():
                event_type = "disk_full"

            # Recommended action heuristic
            action = "Inspect application telemetry and alert on-call engineer."
            if event_type == "db_pool_exhaustion":
                action = "Scale read-replica pool and recycle idle connection handles immediately."
            elif event_type == "out_of_memory":
                action = "Trigger worker container restart and increase pod memory ceiling to 4Gi."
            elif event_type == "latency_spike":
                action = "Drain high-latency node from load balancer rotation and inspect downstream dependencies."
            elif event_type == "gateway_error":
                action = "Failover traffic to standby cluster and verify upstream ingress health."
            elif event_type == "disk_full":
                action = "Purge archived container logs and rotate WAL segments."

            impact_score = 95 if severity == "CRITICAL" else (75 if severity == "HIGH" else 40)

            output_obj = {
                "severity": severity,
                "service": service,
                "event_type": event_type,
                "impact_score": impact_score,
                "recommended_action": action
            }

            instruction = "Analyze the following operational incident log and extract the details into a strict JSON object with fields: severity, service, event_type, impact_score, recommended_action. Output ONLY valid JSON."

            results.append(CultivatedPair(
                instruction=instruction,
                input_text=entry,
                output_text=json.dumps(output_obj, indent=2),
                metadata={"source": block.source, "severity": severity, "service": service}
            ))

        return results
