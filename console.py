"""
Rich console output for PDF to Markdown converter.

Provides visual feedback with progress bars, spinners, and formatted output.
"""

import time
from pathlib import Path
from typing import Optional, Dict
from contextlib import contextmanager

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.live import Live
from rich.text import Text

# Global console instance
console = Console()


def print_header(input_path: str, output_path: str, page_count: Optional[int] = None,
                 file_size: Optional[int] = None, first_page: Optional[int] = None,
                 last_page: Optional[int] = None):
    """Print the conversion header with file info."""
    input_name = Path(input_path).name
    output_name = Path(output_path).name

    # Build info line
    info_parts = []
    if page_count:
        if first_page is not None and last_page is not None:
            # Show page range for non-sequential numbering
            info_parts.append(f"{page_count:,} pages, pp. {first_page}-{last_page}")
        else:
            info_parts.append(f"{page_count:,} pages")
    if file_size:
        size_mb = file_size / (1024 * 1024)
        info_parts.append(f"{size_mb:.1f} MB")
    info_str = f" ({', '.join(info_parts)})" if info_parts else ""

    content = f"[bold]Input:[/bold]  {input_name}{info_str}\n[bold]Output:[/bold] {output_name}"

    panel = Panel(
        content,
        title="[bold blue]PDF to Markdown Converter[/bold blue]",
        border_style="blue",
        padding=(0, 1),
    )
    console.print(panel)
    console.print()


def create_conversion_progress() -> Progress:
    """Create a progress bar for the main conversion phases."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def create_spinner_progress() -> Progress:
    """Create a spinner for indeterminate tasks."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


@contextmanager
def conversion_spinner(description: str):
    """Context manager for showing a spinner during long operations."""
    with create_spinner_progress() as progress:
        task = progress.add_task(description, total=None)
        yield progress, task


def print_step(step_num: int, total_steps: int, description: str, status: str = ""):
    """Print a step indicator."""
    status_text = f"  {status}" if status else ""
    console.print(f"[dim]\\[{step_num}/{total_steps}][/dim] {description}{status_text}")


def print_success(message: str):
    """Print a success message."""
    console.print(f"[green]✓[/green] {message}")


def print_warning(message: str):
    """Print a warning message."""
    console.print(f"[yellow]![/yellow] {message}")


def print_error(message: str):
    """Print an error message."""
    console.print(f"[red]✗[/red] {message}")


def print_conversion_report(report: Dict, show_file_path: bool = True):
    """Print a formatted conversion report."""
    console.print()

    if report['status'] == 'success':
        stats = report['statistics']

        # Create stats table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Label", style="bold")
        table.add_column("Value", justify="right")

        # Format time nicely
        seconds = report['conversion_time']
        if seconds >= 60:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            time_str = f"{minutes}m {secs}s"
        else:
            time_str = f"{seconds:.1f}s"

        table.add_row("Time", time_str)

        # Show page range if PDF has non-sequential numbering
        first_page = stats.get('first_page', 1)
        last_page = stats.get('last_page', stats['pages'])
        if first_page != 1 or last_page != stats['pages']:
            # Non-sequential: show "122 (pp. 41-162)"
            table.add_row("Pages", f"{stats['pages']:,} (pp. {first_page}-{last_page})")
        else:
            table.add_row("Pages", f"{stats['pages']:,}")

        table.add_row("Words", f"{stats['words']:,}")
        table.add_row("Headings", f"{stats['headings']:,}")
        table.add_row("Tables", f"{stats['tables']:,}")

        if 'pages_marked' in stats:
            markers_str = f"{stats['pages_marked']:,} / {stats['pages']:,}"
            # Add note about blank pages if some pages weren't marked
            blank_pages = stats.get('blank_pages', 0)
            unmarked = stats['pages'] - stats['pages_marked']
            if unmarked > 0 and blank_pages > 0:
                if blank_pages >= unmarked:
                    markers_str += f" ({unmarked} blank)"
                else:
                    markers_str += f" ({blank_pages} blank, {unmarked - blank_pages} other)"
            table.add_row("Page Markers", markers_str)

        # Build panel content
        panel = Panel(
            table,
            title="[bold green]Conversion Complete[/bold green]",
            border_style="green",
            padding=(0, 1),
        )
        console.print(panel)

        if show_file_path:
            output_path = Path(report['output_file'])
            console.print(f"\n[dim]Saved to:[/dim] {output_path.name}")
    else:
        error_text = Text()
        error_text.append("Error: ", style="bold red")
        error_text.append(report.get('error', 'Unknown error'))

        panel = Panel(
            error_text,
            title="[bold red]Conversion Failed[/bold red]",
            border_style="red",
        )
        console.print(panel)


def print_batch_summary(success_count: int, error_count: int, total_time: float,
                        total_pages: int, total_words: int):
    """Print a summary for batch conversions."""
    console.print()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Label", style="bold")
    table.add_column("Value", justify="right")

    total_files = success_count + error_count

    table.add_row("Files", f"{success_count} / {total_files} successful")
    if error_count > 0:
        table.add_row("", f"[red]{error_count} failed[/red]")

    # Format time
    if total_time >= 60:
        minutes = int(total_time // 60)
        secs = int(total_time % 60)
        time_str = f"{minutes}m {secs}s"
    else:
        time_str = f"{total_time:.1f}s"

    table.add_row("Total Time", time_str)
    table.add_row("Total Pages", f"{total_pages:,}")
    table.add_row("Total Words", f"{total_words:,}")

    if success_count > 0:
        avg_time = total_time / success_count
        table.add_row("Avg per File", f"{avg_time:.1f}s")

    border_style = "green" if error_count == 0 else "yellow"
    title_style = "bold green" if error_count == 0 else "bold yellow"

    panel = Panel(
        table,
        title=f"[{title_style}]Batch Conversion Complete[/{title_style}]",
        border_style=border_style,
        padding=(0, 1),
    )
    console.print(panel)


class ConversionProgress:
    """
    Context manager for tracking conversion progress with rich output.

    Usage:
        with ConversionProgress(input_path, output_path) as progress:
            with progress.phase("Converting PDF"):
                # ... do conversion
            with progress.phase("Adding page markers", total=1309) as update:
                for i, page in enumerate(pages):
                    update(i + 1)
    """

    def __init__(self, input_path: str, output_path: str,
                 page_count: Optional[int] = None, file_size: Optional[int] = None,
                 first_page: Optional[int] = None, last_page: Optional[int] = None,
                 quiet: bool = False):
        self.input_path = input_path
        self.output_path = output_path
        self.page_count = page_count
        self.file_size = file_size
        self.first_page = first_page
        self.last_page = last_page
        self.quiet = quiet
        self.start_time = None
        self._progress = None
        self._current_task = None

    def __enter__(self):
        if not self.quiet:
            print_header(self.input_path, self.output_path, self.page_count, self.file_size,
                        self.first_page, self.last_page)
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._progress:
            self._progress.stop()
        return False

    @contextmanager
    def phase(self, description: str, total: Optional[int] = None):
        """
        Context manager for a conversion phase.

        Args:
            description: What this phase is doing
            total: If provided, shows a progress bar; otherwise shows a spinner
        """
        if self.quiet:
            yield lambda x: None
            return

        if total is not None:
            # Progress bar mode
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}[/bold]"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            )
            with progress:
                task = progress.add_task(description, total=total)

                def update(completed: int, desc: Optional[str] = None):
                    progress.update(task, completed=completed, description=desc or description)

                yield update
        else:
            # Spinner mode
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}[/bold]"),
                TimeElapsedColumn(),
                console=console,
            )
            with progress:
                task = progress.add_task(description, total=None)
                yield lambda x: None

    def update_status(self, message: str):
        """Update the current status message."""
        if not self.quiet:
            console.print(f"  [dim]{message}[/dim]")


def suppress_docling_logging():
    """Suppress verbose Docling logging output."""
    import logging

    # Suppress Docling's verbose output
    logging.getLogger('docling').setLevel(logging.WARNING)
    logging.getLogger('docling_core').setLevel(logging.WARNING)
    logging.getLogger('docling.pipeline').setLevel(logging.WARNING)

    # Also suppress other noisy loggers
    for logger_name in ['PIL', 'urllib3', 'httpx']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
