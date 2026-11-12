"""UniVectorSQL-compatible online translation with strict arXiv query parsing."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
import re
import time
from urllib.request import Request, urlopen

from vec2plan_bandit.queries.arxiv_sql_adapter import parse_record
from vec2plan_bandit.queries.loader import QueryRecord


_FENCE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE)


class TranslationParseError(ValueError):
    """Retain model output for failed-translation audit, without executing SQL."""

    def __init__(self, message, *, sql, response_model, translation_ms, parse_ms):
        super().__init__(message)
        self.sql = sql
        self.response_model = response_model
        self.translation_ms = translation_ms
        self.parse_ms = parse_ms


@dataclass(frozen=True)
class Translation:
    query: QueryRecord
    sql: str
    translation_ms: float
    parse_ms: float
    response_model: str | None = None


class Text2VectorSQLTranslator:
    """Call a local OpenAI-compatible endpoint serving the released model.

    The generated SQL is treated as data. The strict adapter parses its
    supported subset into predicates; generated SQL is never executed directly.
    """

    def __init__(self, *, url: str, model: str, timeout_seconds: float = 60.0,
                 api_key_env: str | None = None, max_tokens: int = 512,
                 model_identity: str | None = None, model_revision: str | None = None,
                 stream: bool = False):
        if not url.startswith(("http://", "https://")) or not model or timeout_seconds <= 0:
            raise ValueError("translator needs an HTTP endpoint, model, and positive timeout")
        self.url = url
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.api_key_env = api_key_env
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        self.max_tokens = max_tokens
        self.model_identity = model_identity or model
        self.model_revision = model_revision
        self.stream = stream
        self.last_partial_sql = ""

    def translate(self, source: QueryRecord) -> Translation:
        prompt = (
            "Translate the user question to exactly one VectorSQL statement for the arXiv schema. "
            "Output SQL only. Choose a.abstract_embedding or a.title_embedding "
            "from the question, then write <chosen_field> MATCH "
            "lembed('all-MiniLM-L6-v2', \"semantic search text\") in WHERE, "
            "followed by supported metadata predicates and a trailing AND a.k = N. "
            "Use arxiv_exp.articles a; article metadata are license, comments, "
            "journal_ref, report_no, update_date, and submitter_id. "
            "Related tables include arxiv_raw.submitters s (id, name), "
            "arxiv_raw.authors au (id, name), and "
            "arxiv_raw.article_authors aa (article_id, author_id). "
            "Example: SELECT a.id FROM arxiv_exp.articles a WHERE "
            "a.abstract_embedding MATCH lembed('all-MiniLM-L6-v2', \"machine learning\") "
            "AND a.license = 'MIT' AND a.k = 5; "
            "Use the original requested top-k N. "
            "Do not fabricate article IDs.\n"
            f"Question: {source.question}\nRequested k: {source.k}"
        )
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"missing translator credential environment variable: {self.api_key_env}")
            headers["Authorization"] = f"Bearer {key}"
        started = time.perf_counter_ns()
        request = Request(self.url, data=body, headers=headers, method="POST")
        self.last_partial_sql = ""
        with urlopen(request, timeout=self.timeout_seconds) as response:
            if self.stream:
                response_model = None
                for line in response:
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        break
                    chunk = json.loads(data)
                    response_model = chunk.get("model", response_model)
                    choices = chunk.get("choices", [])
                    if choices:
                        self.last_partial_sql += choices[0].get("delta", {}).get("content") or ""
                payload = {"model": response_model, "choices": [{"message": {"content": self.last_partial_sql}}]}
            else:
                payload = json.load(response)
        translation_ms = (time.perf_counter_ns() - started) / 1_000_000
        sql = str(payload["choices"][0]["message"]["content"]).strip()
        blocks = re.findall(r"```sql\s*(.*?)```", sql, re.IGNORECASE | re.DOTALL)
        if len(blocks) == 1:
            sql = blocks[0].strip()
        else:
            sql = _FENCE.sub("", sql).strip()
        parse_started = time.perf_counter_ns()
        bucket = {"L": "low", "H": "high", "M": "medium", "U": "unfiltered",
                  "S": "sigma", "R": "range"}.get(source.query_id[0], "translation")
        try:
            parsed, _ = parse_record({"question": source.question, "sql": sql},
                                     bucket=bucket, row_number=source.source_record_number)
            if parsed.k != source.k:
                raise ValueError("translated k differs from the requested k")
        except ValueError as error:
            raise TranslationParseError(str(error), sql=sql, response_model=payload.get("model"),
                translation_ms=translation_ms,
                parse_ms=(time.perf_counter_ns() - parse_started) / 1_000_000) from error
        parsed = replace(parsed, query_id=source.query_id, group_id=source.group_id)
        parse_ms = (time.perf_counter_ns() - parse_started) / 1_000_000
        return Translation(parsed, sql, translation_ms, parse_ms, payload.get("model"))


def run_text2vectorsql_request(*, query: QueryRecord, collector,
                              translator: Text2VectorSQLTranslator,
                              latency_cap_ms: float) -> dict:
    """Translate online then retrieve IDs; score against the original query truth.

    No original SQL, predicates or semantic text are sent to the translator.
    Endpoint/model/SQL audit fields are stored without credential values.
    """
    from vec2plan_bandit.execution.adaptive_runtime import run_native_baseline_request

    started = time.perf_counter_ns()
    translation_started = time.perf_counter_ns()
    translated = None
    translation_elapsed = 0.0
    row = {}
    failure = None
    try:
        translated = translator.translate(query)
        translation_elapsed = (time.perf_counter_ns() - translation_started) / 1_000_000
        row = run_native_baseline_request(query=translated.query, collector=collector,
                                          latency_cap_ms=latency_cap_ms)
    except Exception as error:
        failure = error
        if translated is None:
            translation_elapsed = (time.perf_counter_ns() - translation_started) / 1_000_000
        row = {"article_ids": [], "error_type": type(error).__name__,
               "error_message": str(error)[:500]}
    elapsed = (time.perf_counter_ns() - started) / 1_000_000
    row.update(query_id=query.query_id, k=query.k, e2e_latency_ms=elapsed,
               deadline_pass=elapsed <= latency_cap_ms,
               result_fill_count=len(row["article_ids"]),
               result_full_k=len(row["article_ids"]) >= query.k,
               translation_audit={"endpoint": translator.url, "requested_model": translator.model,
                                  "model_identity": getattr(translator, "model_identity", translator.model),
                                  "model_revision": getattr(translator, "model_revision", None),
                                  "response_model": translated.response_model if translated else getattr(failure, "response_model", None),
                                  "prompt_version": "arxiv-vectorsql-v1",
                                  "generated_sql": translated.sql if translated else getattr(failure, "sql", None),
                                  "semantic_text": translated.query.semantic_text if translated else None,
                                  "predicates": translated.query.predicates if translated else None},
               input_contract="natural_language_plus_requested_k")
    row.setdefault("spans_ms", {}).update(
        translation_ms=translated.translation_ms if translated else getattr(failure, "translation_ms", translation_elapsed),
        translation_parse_ms=translated.parse_ms if translated else getattr(failure, "parse_ms", 0.0))
    return row
