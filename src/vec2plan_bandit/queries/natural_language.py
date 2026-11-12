"""Question-only arXiv field extraction; no gold SQL or requested-k side channel."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import time
from typing import Any
from urllib.request import Request, urlopen

from vec2plan_bandit.execution.path_base import remaining_timeout_ms
from vec2plan_bandit.execution.timing import evaluation_deadline_ns
from vec2plan_bandit.features.feature_pipeline import SUPPORTED_PREDICATE_TYPES
from vec2plan_bandit.queries.loader import QueryRecord


PROMPT_VERSION = "arxiv-compact-fields-chatml-v1"
JSON_PREFIX = '{"s":'
SYSTEM = (
    'Extract retrieval fields as compact JSON only. No SQL or explanations. '
    'Schema: {"s":"search topic only","e":"abstract","k":5,"p":[["predicate_type",["value"]]]}. '
    'e is abstract unless explicitly requesting title similarity; requested output titles do not change e. '
    'k is the requested count, integer 1..10; use 1 for single article, 5 if unspecified. '
    'p is empty without filters. Multiple predicates are AND. '
    'Types: submitter_eq,submitter_in,author_eq,author_in,author_like,category_eq,category_in,license_eq,license_in,comments_eq,comments_like,journal_ref_eq,journal_ref_like,report_no_eq,update_date_eq,update_date_le,update_date_ge,update_date_between,update_date_outside,update_date_range,latest_update. '
    'Use YYYY-MM-DD dates. between includes both endpoints; outside means <= first OR >= second; range means >= first AND < second. '
    'Copy exact metadata strings. LIKE values use % wildcards. For license use its name or URL. '
    'Preserve the full topic, removing metadata, counts, and requested result fields.'
)
EXAMPLES = (
    ('Find the 3 papers about graph learning updated on or before 2020-01-01.',
     '{"s":"graph learning","e":"abstract","k":3,"p":[["update_date_le",["2020-01-01"]]]}'),
    ('Find 2 articles submitted by Example Institute about quantum computing.',
     '{"s":"quantum computing","e":"abstract","k":2,"p":[["submitter_eq",["Example Institute"]]]}'),
)


def license_value(value: str) -> str:
    aliases = {
        "arxiv non-exclusive distribution 1.0": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
        "arxiv nonexclusive distribution 1.0": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
        "public domain": "http://creativecommons.org/licenses/publicdomain/",
        "cc0": "http://creativecommons.org/publicdomain/zero/1.0/",
        "cc0 1.0": "http://creativecommons.org/publicdomain/zero/1.0/",
    }
    for code in ("by", "by-sa", "by-nc-sa", "by-nc-nd"):
        for version in ("3.0", "4.0"):
            aliases[f"cc {code} {version}"] = f"http://creativecommons.org/licenses/{code}/{version}/"
    key = " ".join(value.strip().casefold().split())
    if key in aliases:
        return aliases[key]
    if key.startswith(("creativecommons.org/", "arxiv.org/licenses/")):
        return "http://" + value.strip().rstrip("/") + "/"
    return value


@dataclass(frozen=True)
class StructuredQueryRecord(QueryRecord):
    # These replace SQL-text-derived features while keeping the frozen feature order.
    has_range_condition: bool = False
    has_explicit_order_by: bool = False


def query_from_fields(question: str, fields: dict[str, Any]) -> StructuredQueryRecord:
    if not isinstance(fields, dict) or set(fields) != {"s", "e", "k", "p"}:
        raise ValueError("expected exactly s, e, k, p in structured response")
    semantic, column, k, raw = (fields[key] for key in ("s", "e", "k", "p"))
    if not isinstance(semantic, str) or not semantic.strip() or len(semantic) > 2048:
        raise ValueError("invalid semantic search text")
    if column not in {"abstract", "title"} or type(k) is not int or not 1 <= k <= 10:
        raise ValueError("invalid embedding field or top-k")
    if not isinstance(raw, list) or len(raw) > 16:
        raise ValueError("invalid predicate list")
    predicates = []
    for predicate in raw:
        if not isinstance(predicate, list) or len(predicate) != 2:
            raise ValueError("predicate must be [type, values]")
        kind, values = predicate
        if kind not in SUPPORTED_PREDICATE_TYPES or not isinstance(values, list):
            raise ValueError("unsupported predicate")
        if any(not isinstance(v, str) or len(v) > 4096 for v in values):
            raise ValueError("invalid predicate value")
        count = 0 if kind == "latest_update" else 2 if kind in {
            "update_date_between", "update_date_outside", "update_date_range"
        } else None if kind.endswith("_in") else 1
        if (count is None and not values) or (count is not None and len(values) != count):
            raise ValueError("wrong predicate value count")
        if kind.startswith("update_date_"):
            for value in values:
                if date.fromisoformat(value).isoformat() != value:
                    raise ValueError("dates must be YYYY-MM-DD")
            if len(values) == 2 and values[0] > values[1]:
                raise ValueError("date endpoints are reversed")
        if kind.startswith("license_"):
            values = [license_value(v) for v in values]
        predicates.append({"type": kind, "values": list(dict.fromkeys(values)) if kind.endswith("_in") else list(values)})
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
    return StructuredQueryRecord(
        query_id="N" + digest[:16], group_id="N" + digest[:16], question=question,
        semantic_text=semantic.strip(), embedding_column=column + "_embedding", k=k,
        predicates=tuple(predicates), source_sql="", cosine_sql="", candidate_sql=(),
        source_record_number=0, source_hash=digest,
        has_range_condition=any(p["type"] in {
            "update_date_le", "update_date_ge", "update_date_between",
            "update_date_outside", "update_date_range"
        } for p in predicates),
    )


class NaturalLanguageStructurer:
    """Fresh streamed completion for each request, with compact JSON prefill."""

    def __init__(self, *, url: str, model: str, model_identity: str,
                 model_revision: str, max_tokens: int = 384, timeout_seconds: float = 60):
        self.url, self.model = url, model
        self.model_identity, self.model_revision = model_identity, model_revision
        self.max_tokens, self.timeout_seconds = int(max_tokens), float(timeout_seconds)
        self.last_audit: dict[str, Any] = {}

    def structure(self, question: str) -> StructuredQueryRecord:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("input must be a nonempty question string")
        if "<|im_" in question:
            raise ValueError("reserved model delimiters in input")
        prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        for source, target in EXAMPLES:
            prompt += f"<|im_start|>user\n{source}<|im_end|>\n<|im_start|>assistant\n{target}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{JSON_PREFIX}"
        self.last_audit = {
            "input_contract": "question_string_only", "prompt_version": PROMPT_VERSION,
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "model_identity": self.model_identity, "model_revision": self.model_revision,
            "endpoint": self.url, "requested_model": self.model,
            "raw_output": JSON_PREFIX, "status": "in_progress", "result_cache": False,
            "gold_fields_supplied": [], "max_tokens": self.max_tokens,
        }
        body = json.dumps({"model": self.model, "prompt": prompt, "temperature": 0,
                           "max_tokens": self.max_tokens, "stream": True,
                           "stop": ["<|im_end|>"]}).encode("utf-8")
        budget = remaining_timeout_ms(evaluation_deadline_ns())
        timeout = min(self.timeout_seconds, budget / 1000) if budget is not None else self.timeout_seconds
        request = Request(self.url, data=body, headers={"Content-Type": "application/json"})
        started = time.perf_counter_ns()
        try:
            with urlopen(request, timeout=timeout) as response:
                for line in response:
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        break
                    chunk = json.loads(data)
                    self.last_audit["response_model"] = chunk.get("model")
                    if chunk.get("usage"):
                        self.last_audit["usage"] = chunk["usage"]
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    self.last_audit["raw_output"] += choices[0].get("text", "")
                    try:
                        fields, end = json.JSONDecoder().raw_decode(self.last_audit["raw_output"])
                    except json.JSONDecodeError:
                        continue
                    # Close the stream as soon as the complete object arrives.
                    result = query_from_fields(question, fields)
                    self.last_audit.update(status="ok", json_characters=end,
                        structured_query={"semantic_text": result.semantic_text,
                            "embedding_column": result.embedding_column,
                            "k": result.k, "predicates": list(result.predicates)})
                    return result
            raise ValueError("model did not return a complete structured JSON object")
        except BaseException as error:
            self.last_audit.update(status="failed", error_type=type(error).__name__)
            raise
        finally:
            self.last_audit["structuring_wall_ms"] = (time.perf_counter_ns() - started) / 1_000_000
