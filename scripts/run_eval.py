#!/usr/bin/env python3
"""
DOVA Evaluation Framework.

Runs test instances against the DOVA API and scores responses
across multiple dimensions. Supports comparison mode to measure
the impact of DOVA's deep research pipeline vs LLM-only baseline.

Usage:
    # Standard eval
    python scripts/run_eval.py run --subset 10

    # Compare DOVA (tools+synthesis) vs baseline (LLM-only)
    python scripts/run_eval.py run --subset 10 --compare
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import httpx
import structlog

logger = structlog.get_logger(__name__)

# --- Configuration ---

DEFAULT_API_BASE = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 120  # seconds per task
DEFAULT_MAX_RETRIES = 2
DEFAULT_CONCURRENCY = 3

DIMENSIONS = [
    "autonomy_agency",
    "planning_goals",
    "environment_perception",
    "research_synthesis",
    "reasoning",
    "multi_agent",
]

# Labels for comparison modes
MODE_DOVA = "dova"       # Full pipeline: deliberation + tools + synthesis
MODE_BASELINE = "baseline"  # LLM-only: no tools, direct response


# --- Data Models ---

@dataclass
class TestInstance:
    id: str
    query: str
    sources: list[str]
    dimension: str
    expected: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    success: bool
    mode: str = MODE_DOVA
    confidence: float = 0.0
    execution_time: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    response: dict[str, Any] | None = None


@dataclass
class DimensionResult:
    name: str
    scores: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def std(self) -> float:
        if len(self.scores) < 2:
            return 0.0
        m = self.mean
        return (sum((s - m) ** 2 for s in self.scores) / len(self.scores)) ** 0.5

    @property
    def ci_95(self) -> tuple[float, float]:
        if not self.scores:
            return (0.0, 0.0)
        m = self.mean
        margin = 1.96 * self.std / (len(self.scores) ** 0.5) if len(self.scores) > 1 else 0.0
        return (round(m - margin, 3), round(m + margin, 3))


# --- Scoring ---

def score_response(
    response: dict[str, Any],
    instance: TestInstance,
    mode: str = MODE_DOVA,
) -> dict[str, float]:
    """Score a response across dimension-relevant criteria."""
    scores: dict[str, float] = {}
    answer = response.get("answer", "") or response.get("summary", "")
    expected = instance.expected

    # Length / completeness
    min_len = expected.get("min_answer_length", 50)
    if len(answer) >= min_len:
        scores["completeness"] = 1.0
    elif len(answer) >= min_len * 0.5:
        scores["completeness"] = 0.6
    elif len(answer) > 0:
        scores["completeness"] = 0.3
    else:
        scores["completeness"] = 0.0

    # Source evidence (grounding)
    has_papers = bool(response.get("papers"))
    has_repos = bool(response.get("repositories"))
    has_models = bool(response.get("models") or response.get("datasets"))
    has_web = bool(response.get("web_results"))
    source_count = sum([has_papers, has_repos, has_models, has_web])

    if mode == MODE_BASELINE:
        # Baseline has no tools — score sources leniently
        scores["sources"] = 0.5
    elif expected.get("must_contain_sources"):
        scores["sources"] = min(1.0, source_count / max(1, len(instance.sources)))
    else:
        scores["sources"] = 1.0 if source_count > 0 else 0.5

    # Confidence from API
    scores["confidence"] = response.get("confidence", 0.0)

    # Insights quality
    insights = response.get("insights", [])
    scores["insights"] = min(1.0, len(insights) / 3) if insights else 0.3

    # Relevance (keyword overlap with query)
    query_words = {w.lower().strip(".,!?") for w in instance.query.split() if len(w) > 3}
    answer_lower = answer.lower()
    if query_words:
        overlap = sum(1 for w in query_words if w in answer_lower)
        scores["relevance"] = min(1.0, overlap / (len(query_words) * 0.4))
    else:
        scores["relevance"] = 0.5

    # Specificity — does the answer contain concrete details?
    # (numbers, proper nouns, URLs, code snippets)
    specificity_signals = 0
    if any(c.isdigit() for c in answer):
        specificity_signals += 1
    if "http" in answer_lower or "github.com" in answer_lower:
        specificity_signals += 1
    if "```" in answer or "`" in answer:
        specificity_signals += 1
    if any(word[0].isupper() for word in answer.split()[1:10] if len(word) > 2):
        specificity_signals += 1
    scores["specificity"] = min(1.0, specificity_signals / 3)

    return scores


# --- API Client ---

async def call_research_api(
    client: httpx.AsyncClient,
    instance: TestInstance,
    timeout: int,
    mode: str = MODE_DOVA,
) -> dict[str, Any]:
    """Call the DOVA research API for a single test instance.

    In baseline mode, sources=[] forces the orchestrator to respond
    from LLM knowledge only (no tool invocations).
    """
    sources = instance.sources if mode == MODE_DOVA else []

    response = await client.post(
        "/api/v1/research",
        json={
            "query": instance.query,
            "sources": sources,
            "max_results": 10,
            "orchestrator": "thinking",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


# --- Task Runner ---

async def run_task(
    client: httpx.AsyncClient,
    instance: TestInstance,
    timeout: int,
    max_retries: int,
    mode: str = MODE_DOVA,
) -> TaskResult:
    """Run a single eval task with timeout and retry."""
    last_error = None
    tag = f"[{mode}] {instance.id}"

    for attempt in range(1, max_retries + 1):
        start = time.time()
        try:
            resp = await asyncio.wait_for(
                call_research_api(client, instance, timeout, mode),
                timeout=timeout + 5,
            )
            elapsed = time.time() - start
            scores = score_response(resp, instance, mode)
            confidence = sum(scores.values()) / len(scores) if scores else 0.0

            logger.info(
                "task_completed",
                task_id=instance.id,
                mode=mode,
                confidence=round(confidence, 2),
                execution_time=round(elapsed, 2),
            )
            return TaskResult(
                task_id=instance.id,
                success=True,
                mode=mode,
                confidence=confidence,
                execution_time=elapsed,
                scores=scores,
                response=resp,
            )

        except (asyncio.TimeoutError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            elapsed = time.time() - start
            last_error = f"{type(e).__name__}: timeout after {elapsed:.0f}s"
            logger.warning("task_timeout", tag=tag, attempt=attempt, elapsed=round(elapsed, 1))
            # Don't retry timeouts — provider chain already retried internally
            break

        except httpx.HTTPStatusError as e:
            elapsed = time.time() - start
            last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            logger.warning("task_http_error", tag=tag, attempt=attempt, status=e.response.status_code)
            if e.response.status_code < 500:
                break
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)

        except Exception as e:
            elapsed = time.time() - start
            last_error = f"{type(e).__name__}: {e}"
            logger.warning("task_error", tag=tag, attempt=attempt, error=last_error)
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)

    logger.error("task_failed", tag=tag, error=last_error)
    return TaskResult(task_id=instance.id, success=False, mode=mode, error=last_error)


# --- Pipeline ---

def _aggregate_dimensions(
    results: list[TaskResult],
    instances: list[TestInstance],
) -> dict[str, DimensionResult]:
    """Aggregate task results into dimension scores."""
    dim_results = {d: DimensionResult(name=d) for d in DIMENSIONS}
    inst_map = {i.id: i for i in instances}
    for r in results:
        if not r.success:
            continue
        inst = inst_map.get(r.task_id)
        if inst and inst.dimension in dim_results:
            dim_results[inst.dimension].scores.append(r.confidence)
    return dim_results


async def run_pipeline(
    instances: list[TestInstance],
    api_base: str,
    timeout: int,
    max_retries: int,
    concurrency: int,
    compare: bool = False,
) -> dict[str, Any]:
    """Run eval tasks. Returns dict with 'dova' and optionally 'baseline' results."""
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    modes = [MODE_DOVA]
    if compare:
        modes.append(MODE_BASELINE)

    all_results: dict[str, Any] = {"run_id": run_id}

    for mode in modes:
        label = "DOVA (tools + synthesis)" if mode == MODE_DOVA else "Baseline (LLM-only)"
        logger.info("starting_pipeline", run_id=run_id, mode=mode, label=label)
        logger.info("starting_data_collection", tasks=len(instances), mode=mode)

        semaphore = asyncio.Semaphore(concurrency)
        completed = 0
        failed = 0

        async def run_with_sem(inst: TestInstance, _mode: str = mode) -> TaskResult:
            nonlocal completed, failed
            async with semaphore:
                result = await run_task(client, inst, timeout, max_retries, _mode)
                if result.success:
                    completed += 1
                else:
                    failed += 1
                total_done = completed + failed
                logger.info(
                    "progress",
                    mode=_mode,
                    completed=total_done,
                    total=len(instances),
                    success=completed,
                    failed=failed,
                    percent=f"{total_done / len(instances) * 100:.1f}%",
                )
                return result

        async with httpx.AsyncClient(base_url=api_base) as client:
            tasks = [run_with_sem(inst) for inst in instances]
            results = list(await asyncio.gather(*tasks))

        dim_results = _aggregate_dimensions(results, instances)

        success_n = sum(1 for r in results if r.success)
        logger.info("evaluation_complete", mode=mode, total=len(results), success=success_n, failed=len(results) - success_n)

        all_results[mode] = {
            "results": results,
            "dim_results": dim_results,
        }

        # Reset counters for next mode
        completed = 0
        failed = 0

    logger.info("pipeline_complete", run_id=run_id, modes=modes)
    return all_results


# --- Reporting ---

def generate_reports(
    pipeline_output: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Generate JSON and HTML reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = pipeline_output["run_id"]
    compare = MODE_BASELINE in pipeline_output

    report: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "compare": compare,
        "modes": {},
    }

    for mode in [MODE_DOVA, MODE_BASELINE]:
        if mode not in pipeline_output:
            continue
        data = pipeline_output[mode]
        results: list[TaskResult] = data["results"]
        dim_results: dict[str, DimensionResult] = data["dim_results"]

        mode_report: dict[str, Any] = {
            "total": len(results),
            "success": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "dimensions": {},
            "tasks": [],
        }
        for name, dr in dim_results.items():
            mode_report["dimensions"][name] = {
                "mean": round(dr.mean, 3),
                "std": round(dr.std, 3),
                "ci_95": list(dr.ci_95),
                "n": len(dr.scores),
                "pass": dr.mean >= 0.6,
            }
        for r in results:
            mode_report["tasks"].append({
                "task_id": r.task_id,
                "success": r.success,
                "confidence": round(r.confidence, 3),
                "execution_time": round(r.execution_time, 2),
                "scores": {k: round(v, 3) for k, v in r.scores.items()},
                "error": r.error,
            })
        report["modes"][mode] = mode_report

    # Compute deltas if comparing
    if compare:
        report["deltas"] = {}
        dova_dims = report["modes"][MODE_DOVA]["dimensions"]
        base_dims = report["modes"][MODE_BASELINE]["dimensions"]
        for dim in DIMENSIONS:
            d_mean = dova_dims.get(dim, {}).get("mean", 0.0)
            b_mean = base_dims.get(dim, {}).get("mean", 0.0)
            delta = d_mean - b_mean
            pct = (delta / b_mean * 100) if b_mean > 0 else 0.0
            report["deltas"][dim] = {
                "dova": round(d_mean, 3),
                "baseline": round(b_mean, 3),
                "delta": round(delta, 3),
                "pct_change": round(pct, 1),
            }

    json_path = output_dir / f"report_{run_id}.json"
    json_path.write_text(json.dumps(report, indent=2))
    logger.info("json_report_generated", path=str(json_path))

    html = _build_html_dashboard(report)
    html_path = output_dir / f"dashboard_{run_id}.html"
    html_path.write_text(html)
    logger.info("html_dashboard_generated", path=str(html_path))

    return json_path, html_path


def _build_html_dashboard(report: dict) -> str:
    compare = report.get("compare", False)
    run_id = report["run_id"]

    # Build dimension table
    if compare:
        header = "<tr><th>Dimension</th><th>DOVA</th><th>Baseline</th><th>Delta</th><th>Change</th><th>Winner</th></tr>"
        rows = ""
        for dim in DIMENSIONS:
            d = report["deltas"].get(dim, {})
            dova_v = d.get("dova", 0)
            base_v = d.get("baseline", 0)
            delta = d.get("delta", 0)
            pct = d.get("pct_change", 0)
            if delta > 0.01:
                winner = "DOVA"
                color = "#4caf50"
            elif delta < -0.01:
                winner = "Baseline"
                color = "#f44336"
            else:
                winner = "Tie"
                color = "#999"
            sign = "+" if delta >= 0 else ""
            rows += (
                f'<tr><td>{dim}</td><td>{dova_v:.3f}</td><td>{base_v:.3f}</td>'
                f'<td style="color:{color};font-weight:bold">{sign}{delta:.3f}</td>'
                f'<td>{sign}{pct:.1f}%</td>'
                f'<td style="color:{color};font-weight:bold">{winner}</td></tr>\n'
            )
        dova_m = report["modes"][MODE_DOVA]
        base_m = report["modes"][MODE_BASELINE]
        summary = (
            f"<p>DOVA: {dova_m['success']}/{dova_m['total']} succeeded | "
            f"Baseline: {base_m['success']}/{base_m['total']} succeeded</p>"
        )
    else:
        header = "<tr><th>Dimension</th><th>Mean</th><th>95% CI</th><th>Std</th><th>N</th><th>Status</th></tr>"
        rows = ""
        dova_m = report["modes"][MODE_DOVA]
        for dim in DIMENSIONS:
            d = dova_m["dimensions"].get(dim, {})
            mean = d.get("mean", 0)
            ci = d.get("ci_95", [0, 0])
            std = d.get("std", 0)
            n = d.get("n", 0)
            passed = d.get("pass", False)
            status = "PASS" if passed else "FAIL"
            color = "#4caf50" if passed else "#f44336"
            rows += (
                f'<tr><td>{dim}</td><td>{mean:.3f}</td><td>[{ci[0]}, {ci[1]}]</td>'
                f'<td>{std:.3f}</td><td>{n}</td>'
                f'<td style="color:{color};font-weight:bold">{status}</td></tr>\n'
            )
        summary = f"<p>Total: {dova_m['total']} | Success: {dova_m['success']} | Failed: {dova_m['failed']}</p>"

    title = "DOVA vs Baseline Comparison" if compare else "DOVA Evaluation Report"

    return f"""<!DOCTYPE html>
<html><head><title>{title} - {run_id}</title>
<style>
body{{font-family:monospace;margin:2em;background:#fafafa}}
h1{{color:#333}}
table{{border-collapse:collapse;width:100%;margin:1em 0}}
th,td{{border:1px solid #ccc;padding:8px;text-align:left}}
th{{background:#f0f0f0}}
tr:nth-child(even){{background:#f9f9f9}}
.card{{background:white;border:1px solid #ddd;border-radius:4px;padding:1em;margin:1em 0}}
</style></head>
<body>
<h1>{title}</h1>
<div class="card">{summary}</div>
<table>{header}
{rows}</table>
<p style="color:#999;font-size:0.85em">Run: {run_id} | Generated: {report.get("timestamp", "")}</p>
</body></html>"""


# --- Per-task comparison table ---

def _build_task_comparison(
    dova_results: list[TaskResult],
    baseline_results: list[TaskResult],
) -> list[dict[str, Any]]:
    """Build per-task comparison data."""
    base_map = {r.task_id: r for r in baseline_results}
    rows = []
    for dr in dova_results:
        br = base_map.get(dr.task_id)
        delta = (dr.confidence - br.confidence) if br and br.success and dr.success else None
        rows.append({
            "task_id": dr.task_id,
            "dova_score": round(dr.confidence, 3) if dr.success else "FAIL",
            "baseline_score": round(br.confidence, 3) if br and br.success else "FAIL",
            "delta": round(delta, 3) if delta is not None else "N/A",
            "dova_time": round(dr.execution_time, 1) if dr.success else "-",
            "baseline_time": round(br.execution_time, 1) if br and br.success else "-",
        })
    return rows


# --- CLI ---

def load_instances(path: str) -> list[TestInstance]:
    instances = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            instances.append(TestInstance(
                id=data["id"],
                query=data["query"],
                sources=data.get("sources", ["arxiv"]),
                dimension=data.get("dimension", "reasoning"),
                expected=data.get("expected", {}),
            ))
    logger.info("loaded_test_instances", count=len(instances), path=path)
    return instances


@click.group()
def cli():
    """DOVA Evaluation Framework"""
    click.echo("DOVA Evaluation Framework\n")
    for d in DIMENSIONS:
        click.echo(f"  Registered: {d}")
    click.echo()


@cli.command()
@click.option("--test-file", default="data/test_sets/test_instances.jsonl", help="Test instances JSONL file")
@click.option("--subset", default=0, type=int, help="Run only first N instances (0=all)")
@click.option("--api-base", default=DEFAULT_API_BASE, help="DOVA API base URL")
@click.option("--timeout", default=DEFAULT_TIMEOUT, type=int, help="Per-task timeout in seconds")
@click.option("--max-retries", default=DEFAULT_MAX_RETRIES, type=int, help="Max retries per task")
@click.option("--concurrency", default=DEFAULT_CONCURRENCY, type=int, help="Max concurrent tasks")
@click.option("--output-dir", default="data/results", help="Output directory for reports")
@click.option("--compare", is_flag=True, default=False, help="Compare DOVA vs LLM-only baseline")
def run(test_file, subset, api_base, timeout, max_retries, concurrency, output_dir, compare):
    """Run evaluation pipeline."""
    instances = load_instances(test_file)

    if subset > 0:
        instances = instances[:subset]
        click.echo(f"Running subset of {subset} test instances")

    click.echo(f"Loaded {len(instances)} test instances")
    if compare:
        click.echo("Mode: COMPARISON (DOVA vs Baseline)\n")
    else:
        click.echo("Mode: DOVA only\n")

    pipeline_output = asyncio.run(
        run_pipeline(instances, api_base, timeout, max_retries, concurrency, compare)
    )
    run_id = pipeline_output["run_id"]

    click.echo("\nGenerating reports...")
    output_path = Path(output_dir)
    json_path, html_path = generate_reports(pipeline_output, output_path)
    click.echo(f"  JSON report: {json_path}")
    click.echo(f"  HTML dashboard: {html_path}")

    # --- Print summary ---
    dova_data = pipeline_output[MODE_DOVA]
    dova_results: list[TaskResult] = dova_data["results"]
    dova_dims: dict[str, DimensionResult] = dova_data["dim_results"]

    dova_success = sum(1 for r in dova_results if r.success)
    click.echo(f"\nEvaluation Summary\n")
    click.echo(f"{'Run ID':<15} {run_id}")
    click.echo(f"{'Total Tasks':<15} {len(dova_results)}")

    if compare:
        base_data = pipeline_output[MODE_BASELINE]
        base_results: list[TaskResult] = base_data["results"]
        base_dims: dict[str, DimensionResult] = base_data["dim_results"]
        base_success = sum(1 for r in base_results if r.success)

        click.echo(f"{'DOVA Success':<15} {dova_success}/{len(dova_results)}")
        click.echo(f"{'Base Success':<15} {base_success}/{len(base_results)}")
        click.echo()

        # Comparison dimension table
        click.echo(f"{'Dimension':<25} {'DOVA':>7} {'Baseline':>9} {'Delta':>7} {'Change':>8} {'Winner':>8}")
        click.echo("-" * 70)

        dova_wins = 0
        base_wins = 0
        for dim in DIMENSIONS:
            d_mean = dova_dims[dim].mean
            b_mean = base_dims[dim].mean
            delta = d_mean - b_mean
            pct = (delta / b_mean * 100) if b_mean > 0 else 0.0
            sign = "+" if delta >= 0 else ""
            if delta > 0.01:
                winner = "DOVA"
                dova_wins += 1
            elif delta < -0.01:
                winner = "Base"
                base_wins += 1
            else:
                winner = "Tie"
            click.echo(
                f"{dim:<25} {d_mean:>7.3f} {b_mean:>9.3f} {sign}{delta:>6.3f} {sign}{pct:>6.1f}% {winner:>8}"
            )

        click.echo("-" * 70)
        click.echo(f"DOVA wins: {dova_wins} | Baseline wins: {base_wins} | Ties: {len(DIMENSIONS) - dova_wins - base_wins}")

        # Per-task comparison
        task_comp = _build_task_comparison(dova_results, base_results)
        click.echo(f"\n{'Task':<12} {'DOVA':>7} {'Baseline':>9} {'Delta':>7} {'DOVA(s)':>8} {'Base(s)':>8}")
        click.echo("-" * 55)
        for row in task_comp:
            click.echo(
                f"{row['task_id']:<12} {str(row['dova_score']):>7} {str(row['baseline_score']):>9} "
                f"{str(row['delta']):>7} {str(row['dova_time']):>8} {str(row['baseline_time']):>8}"
            )

    else:
        click.echo(f"{'Successful':<15} {dova_success}")
        click.echo(f"{'Failed':<15} {len(dova_results) - dova_success}")
        click.echo(f"{'Success Rate':<15} {dova_success / len(dova_results) * 100:.1f}%")
        click.echo()

        click.echo(f"{'Dimension':<25} {'Mean':>6} {'95% CI':>16} {'Std':>6} {'N':>4} {'Status':>8}")
        click.echo("-" * 70)
        for dim in DIMENSIONS:
            dr = dova_dims[dim]
            ci = dr.ci_95
            status = "PASS" if dr.mean >= 0.6 else "FAIL"
            click.echo(
                f"{dim:<25} {dr.mean:>6.3f} [{ci[0]:.3f}, {ci[1]:.3f}] {dr.std:>6.3f} {len(dr.scores):>4} {status:>8}"
            )


if __name__ == "__main__":
    cli()
