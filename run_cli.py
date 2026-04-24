"""
CLI entry point for the Acquirer Identification Engine.

Usage:
    python run_cli.py                              # Default: HC Services, $200M
    python run_cli.py --sector "Health IT" --size 300

Outputs:
  1. Live progress with rich console formatting
  2. Top 10 acquirers with conviction badges
  3. Full rationale cards per acquirer
  4. Final cost and latency summary
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure src/ is importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from acquirer_engine.agent import identify_acquirers
from acquirer_engine.config import settings
from acquirer_engine.evidence import load_csv
from acquirer_engine.observability import configure_logging
from acquirer_engine.schemas import TargetProfile

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M&A Acquirer Identification Engine")
    p.add_argument("--sector", default="Healthcare Services")
    p.add_argument("--size", type=float, default=200.0, help="Target EV in $M")
    p.add_argument("--geography", default="Regional")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--csv", default=str(settings.csv_path))
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logs")
    return p.parse_args()


def print_header(target: TargetProfile) -> None:
    console.print(
        Panel.fit(
            f"[bold navy_blue]William Blair[/] · Acquirer Identification Engine\n\n"
            f"[bold]Target:[/] {target.sector}  ·  [bold]EV:[/] ${target.size_mm:.0f}M  "
            f"·  [bold]Geo:[/] {target.geography}\n"
            f"[dim]Model: {settings.model} · Temp: {settings.temperature} · "
            f"Concurrency: {settings.max_concurrent_requests}[/dim]",
            border_style="blue",
            padding=(1, 2),
        )
    )


def print_top_10(result) -> None:
    table = Table(
        title=f"\nTop {len(result.rationales)} Acquirers",
        title_style="bold",
        show_lines=False,
        header_style="bold cyan",
    )
    table.add_column("#", justify="right", width=3)
    table.add_column("Acquirer", style="bold")
    table.add_column("Type", width=18)
    table.add_column("Conviction", justify="center", width=12)
    table.add_column("Deals", justify="right", width=6)

    conv_style = {"High": "bold yellow", "Medium": "cyan", "Low": "dim"}

    for r in result.rationales:
        # acquirer_type is not on rationale; recover from the packet (not stored here).
        # For display we can just show rank + name + conviction from rationale itself.
        style = conv_style.get(r.conviction.level, "white")
        table.add_row(
            str(r.rank),
            r.acquirer_name,
            "—",  # Type would come from the packet if we threaded it through
            f"[{style}]{r.conviction.level}[/]",
            "—",
        )

    console.print(table)


def print_rationale(r) -> None:
    """Pretty-print one rationale card."""
    conv_color = {"High": "yellow", "Medium": "cyan", "Low": "white"}[r.conviction.level]

    header = (
        f"[bold]#{r.rank}  {r.acquirer_name}[/]  "
        f"[{conv_color}][{r.conviction.level}][/]"
    )

    body = (
        f"[bold cyan]OVERVIEW[/]\n{r.acquirer_overview}\n\n"
        f"[bold cyan]STRATEGIC FIT[/]\n{r.strategic_fit_thesis}\n\n"
        f"[bold cyan]PRECEDENT ACTIVITY[/]\n{r.precedent_activity}\n\n"
        f"[bold cyan]VALUATION CONTEXT[/]\n{r.valuation_context}\n\n"
        f"[bold cyan]RISK FLAGS[/]\n{r.risk_flags}\n\n"
        f"[bold cyan]CONVICTION RATIONALE[/]\n{r.conviction.rationale}"
    )

    console.print(Panel(body, title=header, border_style=conv_color, padding=(1, 2)))


def print_summary(result) -> None:
    """The production-thinking proof: cost, latency, token breakdown."""
    table = Table(title="\nRun Summary", show_header=False, title_style="bold")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total duration", f"{result.total_duration_seconds:.2f}s")
    table.add_row("Rationales produced", f"{len(result.rationales)}/10")
    table.add_row("Input tokens", f"{result.total_input_tokens:,}")
    table.add_row("Output tokens", f"{result.total_output_tokens:,}")
    table.add_row("[bold yellow]Total cost (USD)[/]", f"[bold yellow]${result.total_cost_usd:.4f}[/]")

    console.print(table)


async def main_async() -> int:
    args = parse_args()

    if args.verbose:
        # Mutate runtime settings for this run
        import os
        os.environ["LOG_LEVEL"] = "DEBUG"

    configure_logging()

    target = TargetProfile(
        sector=args.sector,
        size_mm=args.size,
        geography=args.geography,
    )
    print_header(target)

    # Load CSV
    df = load_csv(args.csv)
    console.print(f"[dim]Loaded {len(df)} transactions from {args.csv}[/dim]\n")

    # Run the agentic pipeline
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Running agentic pipeline…", total=None)
        try:
            result = await identify_acquirers(df, target, top_n=args.top_n)
        except Exception as e:
            console.print(f"[bold red]ERROR:[/] {e}")
            return 1
        finally:
            progress.update(task, completed=1)

    # Render results
    print_top_10(result)
    console.rule("[bold]Full Rationales[/]")
    for r in result.rationales:
        print_rationale(r)

    print_summary(result)
    return 0


def main() -> None:
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
