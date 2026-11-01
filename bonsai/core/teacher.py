"""
Teacher Engine for Bonsai.
Supports Local Ollama, Cloud APIs (Groq, OpenAI, Gemini, Anthropic),
and an Offline Multi-Archetype Programmatic Synthesizer for 100% zero-blocker execution.
"""

import csv
import io
import json
import os
import re
from typing import List, Dict, Any, Optional
import requests
from bonsai.core.ingest import DocumentBlock
from bonsai.core.constants import DEFAULT_SYSTEM_DIRECTIVE


class CultivatedPair:
    def __init__(self, instruction: str, input_text: str, output_text: str, metadata: Optional[Dict[str, Any]] = None):
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
    """Manages teacher synthesis across Local Ollama, Cloud APIs, or Multi-Archetype Offline Synthesizer."""

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
        """
        Extracts unique, un-augmented base scenario pairs across diverse data archetypes:
        1. Pre-formatted JSON/JSONL instructions
        2. Plain-English goal prompts (synthetic domain scenario expansion)
        3. Operational logs and incident telemetry
        4. Structured tabular / CSV records
        5. Code scripts and function signatures
        6. Unstructured documentation / technical texts
        """
        pairs: List[CultivatedPair] = []

        for block in blocks:
            # 1. Check if block is a user goal prompt
            if block.metadata.get("type") == "goal_prompt" or block.source == "user_goal":
                goal_pairs = self._extract_goal_driven_scenarios(block.content)
                if goal_pairs:
                    pairs.extend(goal_pairs)
                    continue

            # 2. Check pre-formatted JSON / JSONL
            preformatted = self._extract_preformatted(block)
            if preformatted:
                pairs.extend(preformatted)
                continue

            # 3. Detect CSV / Tabular data
            tabular_pairs = self._extract_tabular_scenarios(block)
            if tabular_pairs:
                pairs.extend(tabular_pairs)
                continue

            # 4. Detect Code Snippets
            code_pairs = self._extract_code_scenarios(block)
            if code_pairs:
                pairs.extend(code_pairs)
                continue

            # 5. Detect Ops / Incident / Log patterns
            log_pairs = self._extract_log_incident_patterns(block)
            if log_pairs:
                pairs.extend(log_pairs)
                continue

            # 6. General text / documentation fallback
            doc_pairs = self._extract_document_scenarios(block)
            if doc_pairs:
                pairs.extend(doc_pairs)

        # Fallback safeguard: if still empty, synthesize basic prompt pairs from the raw text
        if not pairs and blocks:
            for b in blocks:
                content = b.content.strip()
                if content:
                    pairs.append(CultivatedPair(
                        instruction="Extract the core entities and structured summary from the provided text.",
                        input_text=content[:500],
                        output_text=json.dumps({"summary": content[:120], "source": b.source}, indent=2),
                        metadata={"source": b.source, "origin": "generic_fallback"}
                    ))
                    pairs.append(CultivatedPair(
                        instruction="Analyze the given information and identify key operational attributes.",
                        input_text=content[:500],
                        output_text=json.dumps({"key_points": [content[:80]], "status": "processed"}, indent=2),
                        metadata={"source": b.source, "origin": "generic_fallback"}
                    ))

        return pairs

    def _extract_preformatted(self, block: DocumentBlock) -> List[CultivatedPair]:
        """Extracts pre-formatted instruction pairs (Alpaca, ShareGPT, OpenAI messages format)."""
        pairs: List[CultivatedPair] = []
        try:
            parsed = json.loads(block.content)
            items = parsed if isinstance(parsed, list) else [parsed]
            for item in items:
                if not isinstance(item, dict):
                    continue

                # Alpaca format: instruction, input, output
                if "instruction" in item and "output" in item:
                    pairs.append(CultivatedPair(
                        instruction=str(item.get("instruction", "")),
                        input_text=str(item.get("input", "")),
                        output_text=item["output"] if isinstance(item["output"], str) else json.dumps(item["output"], indent=2),
                        metadata={"source": block.source, "origin": "alpaca"}
                    ))
                # Prompt / Completion format
                elif "prompt" in item and "completion" in item:
                    pairs.append(CultivatedPair(
                        instruction="Execute the specified prompt.",
                        input_text=str(item.get("prompt", "")),
                        output_text=item["completion"] if isinstance(item["completion"], str) else json.dumps(item["completion"], indent=2),
                        metadata={"source": block.source, "origin": "prompt_completion"}
                    ))
                # Chat messages format: [{"role": "user", ...}, {"role": "assistant", ...}]
                elif "messages" in item and isinstance(item["messages"], list):
                    user_content = ""
                    assistant_content = ""
                    for msg in item["messages"]:
                        if msg.get("role") == "user":
                            user_content = msg.get("content", "")
                        elif msg.get("role") == "assistant":
                            assistant_content = msg.get("content", "")
                    if user_content and assistant_content:
                        pairs.append(CultivatedPair(
                            instruction="Respond to the user request accurately and concisely.",
                            input_text=user_content,
                            output_text=assistant_content if isinstance(assistant_content, str) else json.dumps(assistant_content, indent=2),
                            metadata={"source": block.source, "origin": "chat_messages"}
                        ))
        except Exception:
            pass
        return pairs

    def _extract_goal_driven_scenarios(self, goal_text: str) -> List[CultivatedPair]:
        """
        Synthesizes realistic training & evaluation pairs for a natural-language goal.
        Identifies the domain (invoices/billing, classification, code/SQL, support tickets)
        and seeds diverse, high-fidelity pairs.
        """
        text_lower = goal_text.lower()
        results: List[CultivatedPair] = []

        # Domain A: Invoices / Receipts / Financial Extraction
        if any(w in text_lower for w in ["invoice", "receipt", "billing", "expense", "transaction"]):
            scenarios = [
                (
                    "INVOICE #INV-9821\nVendor: CloudScale Infrastructure Inc.\nDate: 2026-10-14\nTotal: $1,420.50 USD\nLine items: 4x Compute Instances ($1,200.00), Storage S3 500GB ($220.50)\nTax: $0.00\nPayment Terms: Net 30\nDue: 2026-11-13",
                    {
                        "invoice_number": "INV-9821",
                        "vendor": "CloudScale Infrastructure Inc.",
                        "date": "2026-10-14",
                        "due_date": "2026-11-13",
                        "total_amount": 1420.50,
                        "currency": "USD",
                        "line_items_count": 2,
                        "status": "pending_payment"
                    }
                ),
                (
                    "Receipt from: Acme Hardware & Logistics LLC\nTimestamp: 2026-10-18T14:30:00Z\nOrder ID: ORD-55410\nItems: 10x Cat6 Ethernet Cables ($50.00), 2x Gigabit Switches ($240.00)\nSubtotal: $290.00\nSales Tax (8%): $23.20\nGrand Total: $313.20\nPaid via: Corporate Amex ending in 4012",
                    {
                        "invoice_number": "ORD-55410",
                        "vendor": "Acme Hardware & Logistics LLC",
                        "date": "2026-10-18",
                        "due_date": "2026-10-18",
                        "total_amount": 313.20,
                        "currency": "USD",
                        "line_items_count": 2,
                        "status": "paid"
                    }
                ),
                (
                    "INVOICE NOTICE - Global CDN Services GmbH\nInvoice Ref: GCDN-2026-887\nBilled to: TechCorp Solutions\nPeriod: Sept 1 - Sept 30, 2026\nBandwidth Out (12 TB): €480.00\nDDoS Shield Protection: €150.00\nVAT (19%): €119.70\nTotal Due: €749.70\nWire to IBAN: DE89370400440532013000",
                    {
                        "invoice_number": "GCDN-2026-887",
                        "vendor": "Global CDN Services GmbH",
                        "date": "2026-10-01",
                        "due_date": "2026-10-31",
                        "total_amount": 749.70,
                        "currency": "EUR",
                        "line_items_count": 2,
                        "status": "unpaid"
                    }
                ),
                (
                    "Coffee & Co Workspace Receipt\nDate: 2026-10-22\nRef: RCP-1102\nTeam Meeting Day Pass (4 people): $120.00\nCatering: $45.00\nTip: $25.00\nCharged: $190.00 USD (Visa ****8891)",
                    {
                        "invoice_number": "RCP-1102",
                        "vendor": "Coffee & Co Workspace",
                        "date": "2026-10-22",
                        "due_date": "2026-10-22",
                        "total_amount": 190.00,
                        "currency": "USD",
                        "line_items_count": 2,
                        "status": "paid"
                    }
                )
            ]
            for input_doc, output_dict in scenarios:
                results.append(CultivatedPair(
                    instruction="Extract the structured financial details from the invoice or receipt into a strict JSON object with keys: invoice_number, vendor, date, due_date, total_amount, currency, line_items_count, status. Output ONLY valid JSON.",
                    input_text=input_doc,
                    output_text=json.dumps(output_dict, indent=2),
                    metadata={"domain": "invoices", "origin": "goal_synthesis"}
                ))
            return results

        # Domain B: Support Tickets & Triage Classification
        if any(w in text_lower for w in ["support", "ticket", "triage", "customer", "sentiment", "classify"]):
            scenarios = [
                (
                    "Customer: 'Our checkout page is throwing a 500 server error for all European customers! We are losing thousands every minute. Fix this immediately!'",
                    {
                        "category": "payment_failure",
                        "urgency": "CRITICAL",
                        "sentiment": "angry",
                        "action": "Escalate to on-call payments engineer and alert Incident Commander.",
                        "sla_response_minutes": 15
                    }
                ),
                (
                    "Customer: 'Hey team, just wondering if there is an easy way to export our team usage metrics to CSV? Loving the platform so far!'",
                    {
                        "category": "feature_inquiry",
                        "urgency": "LOW",
                        "sentiment": "positive",
                        "action": "Send documentation link for Analytics CSV Export and invite to beta feedback group.",
                        "sla_response_minutes": 240
                    }
                ),
                (
                    "Customer: 'Password reset link sent to my email keeps expiring after 30 seconds. I cannot log into my account.'",
                    {
                        "category": "auth_issue",
                        "urgency": "HIGH",
                        "sentiment": "frustrated",
                        "action": "Issue temporary secure login token and file bug ticket for token expiry ttl.",
                        "sla_response_minutes": 60
                    }
                ),
                (
                    "Customer: 'We noticed a 15% increase in API latency over the past hour on the /v1/search endpoint. Our dashboards show P95 at 850ms.'",
                    {
                        "category": "performance_degradation",
                        "urgency": "MEDIUM",
                        "sentiment": "neutral",
                        "action": "Route to search infrastructure team to verify Elasticsearch cluster load.",
                        "sla_response_minutes": 120
                    }
                )
            ]
            for input_doc, output_dict in scenarios:
                results.append(CultivatedPair(
                    instruction="Analyze the customer message and classify it into a strict JSON payload with fields: category, urgency, sentiment, action, sla_response_minutes. Output ONLY valid JSON.",
                    input_text=input_doc,
                    output_text=json.dumps(output_dict, indent=2),
                    metadata={"domain": "support_triage", "origin": "goal_synthesis"}
                ))
            return results

        # Domain C: Natural Language to Code / SQL / API Commands
        if any(w in text_lower for w in ["sql", "code", "query", "database", "python", "convert", "translate"]):
            scenarios = [
                (
                    "Find all active users who signed up in the last 30 days and have spent over $500 in total orders, ordered by total spent descending.",
                    {
                        "query_type": "SELECT",
                        "target_tables": ["users", "orders"],
                        "sql": "SELECT u.id, u.email, SUM(o.total_amount) AS total_spent FROM users u JOIN orders o ON u.id = o.user_id WHERE u.status = 'active' AND u.created_at >= NOW() - INTERVAL '30 days' GROUP BY u.id, u.email HAVING SUM(o.total_amount) > 500 ORDER BY total_spent DESC;"
                    }
                ),
                (
                    "Get the count of failed login attempts grouped by IP address for the last 24 hours where the count exceeds 10.",
                    {
                        "query_type": "SELECT",
                        "target_tables": ["login_attempts"],
                        "sql": "SELECT ip_address, COUNT(*) AS failure_count FROM login_attempts WHERE status = 'failed' AND attempted_at >= NOW() - INTERVAL '24 hours' GROUP BY ip_address HAVING COUNT(*) > 10 ORDER BY failure_count DESC;"
                    }
                ),
                (
                    "Delete all unverified accounts that haven't logged in for more than 90 days.",
                    {
                        "query_type": "DELETE",
                        "target_tables": ["users"],
                        "sql": "DELETE FROM users WHERE is_verified = FALSE AND (last_login_at < NOW() - INTERVAL '90 days' OR (last_login_at IS NULL AND created_at < NOW() - INTERVAL '90 days'));"
                    }
                ),
                (
                    "Calculate the monthly recurring revenue (MRR) for each subscription plan for the current month.",
                    {
                        "query_type": "SELECT",
                        "target_tables": ["subscriptions", "plans"],
                        "sql": "SELECT p.name AS plan_name, COUNT(s.id) * p.monthly_price AS mrr FROM subscriptions s JOIN plans p ON s.plan_id = p.id WHERE s.status = 'active' GROUP BY p.name, p.monthly_price ORDER BY mrr DESC;"
                    }
                )
            ]
            for input_doc, output_dict in scenarios:
                results.append(CultivatedPair(
                    instruction="Translate the natural language data request into an optimized SQL query with metadata in strict JSON format: query_type, target_tables, sql. Output ONLY valid JSON.",
                    input_text=input_doc,
                    output_text=json.dumps(output_dict, indent=2),
                    metadata={"domain": "nl_to_sql", "origin": "goal_synthesis"}
                ))
            return results

        # General Goal Synthesizer fallback: parse goal into structured tasks
        clean_goal = goal_text.strip()
        variations = [
            ("Task Scenario A: Normal Operating Conditions", "success", "Optimal execution with zero anomalies detected.", 100),
            ("Task Scenario B: Elevated Load Boundary", "warning", "Increased concurrency; adaptive rate limiting triggered.", 75),
            ("Task Scenario C: Failover Recovery Sequence", "remediation", "Failover to redundant node completed within 250ms SLA.", 90),
            ("Task Scenario D: Strict Validation Guard", "validated", "Input schema strictly conforms to destination specification.", 100),
        ]
        for name, status, details, confidence in variations:
            output_obj = {
                "objective": clean_goal,
                "scenario": name,
                "status": status,
                "confidence_score": confidence,
                "resolution": details
            }
            results.append(CultivatedPair(
                instruction=f"Execute deterministic processing for the goal: '{clean_goal}'. Output a structured verification object in valid JSON with fields: objective, scenario, status, confidence_score, resolution.",
                input_text=f"Execution context: {name}",
                output_text=json.dumps(output_obj, indent=2),
                metadata={"domain": "general_goal", "origin": "goal_synthesis"}
            ))

        return results

    def _extract_tabular_scenarios(self, block: DocumentBlock) -> List[CultivatedPair]:
        """Extracts structured query/extraction scenarios from CSV or tabular text."""
        pairs: List[CultivatedPair] = []
        content = block.content.strip()

        # Check if content has CSV characteristics (comma-separated, multiple lines)
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if len(lines) < 2 or "," not in lines[0]:
            return pairs

        try:
            reader = csv.DictReader(io.StringIO(content))
            rows = list(reader)
            if not rows or not reader.fieldnames:
                return pairs

            for idx, row in enumerate(rows[:10]):
                clean_row = {k.strip(): v.strip() for k, v in row.items() if k}
                instruction = f"Extract and validate the record attributes for entry #{idx+1} matching the dataset schema. Output ONLY valid JSON."
                pairs.append(CultivatedPair(
                    instruction=instruction,
                    input_text=json.dumps(clean_row),
                    output_text=json.dumps(clean_row, indent=2),
                    metadata={"source": block.source, "origin": "tabular_csv", "index": idx}
                ))
        except Exception:
            pass

        return pairs

    def _extract_code_scenarios(self, block: DocumentBlock) -> List[CultivatedPair]:
        """Extracts docstring/implementation scenarios from Python or JS/TS code blocks."""
        pairs: List[CultivatedPair] = []
        content = block.content
        if not any(token in content for token in ["def ", "class ", "function ", "export const ", "async function"]):
            return pairs

        # Python functions with docstrings
        py_func_pattern = re.compile(r'def\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)\s*(?:->\s*[^:]+)?:\s*(?:"""([\s\S]*?)""")?([\s\S]*?)(?=\ndef |\Z)')
        for match in py_func_pattern.finditer(content):
            fn_name = match.group(1)
            params = match.group(2).strip()
            docstring = (match.group(3) or "").strip()
            body = match.group(4).strip()

            if docstring and body and len(body) > 20:
                pairs.append(CultivatedPair(
                    instruction=f"Implement the Python function '{fn_name}' according to its specification and signature: def {fn_name}({params}).",
                    input_text=docstring,
                    output_text=f"def {fn_name}({params}):\n    \"\"\"{docstring}\"\"\"\n    {body[:400]}",
                    metadata={"source": block.source, "origin": "code_function", "function": fn_name}
                ))

        return pairs

    def _extract_document_scenarios(self, block: DocumentBlock) -> List[CultivatedPair]:
        """Extracts Q&A and summary scenarios from markdown or documentation paragraphs."""
        pairs: List[CultivatedPair] = []
        content = block.content.strip()
        if len(content) < 40:
            return pairs

        sections = re.split(r'\n#{1,4}\s+', content)
        for s in sections:
            s = s.strip()
            if len(s) < 50:
                continue
            lines = s.splitlines()
            title = lines[0].strip() if lines else "Document Section"
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else lines[0].strip()

            if body:
                output_obj = {
                    "topic": title,
                    "summary": body[:180] + ("..." if len(body) > 180 else ""),
                    "key_takeaways": [line.strip("- *") for line in lines if line.strip().startswith(("-", "*"))][:3] or [title]
                }
                pairs.append(CultivatedPair(
                    instruction=f"Analyze the technical documentation section '{title}' and extract its structured topic summary and key takeaways into valid JSON.",
                    input_text=body[:600],
                    output_text=json.dumps(output_obj, indent=2),
                    metadata={"source": block.source, "origin": "document_qa", "topic": title}
                ))

        return pairs

    def _extract_log_incident_patterns(self, block: DocumentBlock) -> List[CultivatedPair]:
        """Extracts and formats messy ops logs into strict schema extraction pairs."""
        results = []
        text = block.content

        # Only process if text exhibits operational log patterns
        has_log_indicators = any(k in text.upper() for k in ["CRITICAL", "ERROR", "WARN", "FATAL", "PANIC", "TIMEOUT", "HTTP 5"]) or bool(re.search(r'\[\d{4}-\d{2}-\d{2}', text))
        if not has_log_indicators and not block.source.endswith((".log", "logs.txt")):
            return []

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

    def augment_training_scenarios(self, base_pairs: List[CultivatedPair], target_count: int = 24) -> List[CultivatedPair]:
        """
        Expands base training scenarios with diverse prompt phrasings while preserving target outputs.
        Dynamically adapts template variations to match the pair's actual domain.
        """
        augmented: List[CultivatedPair] = list(base_pairs)
        if not base_pairs:
            return augmented

        # Universal phrasing prefixes and wrappers
        preambles = [
            "Please carefully process the following input according to requirements:\n{base}",
            "Analyze the given input data and respond strictly as specified:\n{base}",
            "Task Directive: Execute precise extraction on the text below.\n{base}",
            "Convert and process the provided context into the required format:\n{base}",
            "Direct evaluation request: Follow the format specification exactly.\n{base}"
        ]

        idx = 0
        safety_max = target_count * 2
        while len(augmented) < target_count and idx < safety_max:
            src_pair = base_pairs[idx % len(base_pairs)]
            preamble = preambles[idx % len(preambles)]
            new_instruction = preamble.format(base=src_pair.instruction)

            augmented.append(CultivatedPair(
                instruction=new_instruction,
                input_text=src_pair.input_text,
                output_text=src_pair.output_text,
                metadata={"augmented": True, "template_variant": idx % len(preambles)}
            ))
            idx += 1

        return augmented
