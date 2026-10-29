# Real-World Scenario: Messy Ops Logs → Strict Structured JSON

## The Problem
Production systems generate chaotic, unstructured log streams across Kubernetes pods, PostgreSQL pools, Kafka topics, and edge gateways. 
General foundation models (and especially untrained micro-models) often fail when prompted to parse these logs: they ramble, emit conversational markdown fillers ("Here is the JSON object:"), hallucinate non-existent schema properties, or generate invalid syntax.

## The Bonsai Distillation Objective
Cultivate a pocket-sized 0.5B micro-model (`Qwen/Qwen2.5-Coder-0.5B-Instruct`) to act as an unshakeable, deterministic incident triage parser on consumer PCs.

### Required Output Schema
```json
{
  "severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW",
  "service": string,
  "event_type": string,
  "impact_score": number (0-100),
  "recommended_action": string
}
```

### Cultivate Command
```bash
uvx bonsai prune ./examples/messy_ops_to_json
```
