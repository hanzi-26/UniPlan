"""Read-only adapter for imported arXiv tables; never rewrites source data."""
from __future__ import annotations
import json
import subprocess
from typing import Any


def connection_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    values = dict(parameters)
    container = values.pop("docker_container", None)
    if container and not values.get("password"):
        # The explicitly configured DB container is the credential source.
        # Capture in memory only; never print or persist its password.
        result = subprocess.run(["docker", "inspect", "--format", "{{json .Config.Env}}", str(container)],
                                check=True, capture_output=True, text=True, timeout=10)
        environment = dict(item.split("=", 1) for item in json.loads(result.stdout) if "=" in item)
        if environment.get("POSTGRES_PASSWORD"):
            values["password"] = environment["POSTGRES_PASSWORD"]
    return values


def migrated(config: dict[str, Any]) -> bool:
    return config.get("database", {}).get("layout") == "arxiv_public_migrated"


def adapt_sql(sql: str, config: dict[str, Any]) -> str:
    if not migrated(config):
        return sql
    sql = sql.replace("arxiv_exp.articles", """(SELECT id::bigint AS id, arxiv_id, title, abstract,
        license, comments, journal_ref, report_no,
        NULLIF(submitter_id::text,'')::bigint AS submitter_id, update_date::timestamptz AS update_date,
        title_embedding, abstract_embedding FROM public.articles)""")
    sql = sql.replace("arxiv_raw.versions", "(SELECT article_id, created::timestamptz AS created FROM public.versions)")
    return sql.replace("arxiv_raw.", "public.")
