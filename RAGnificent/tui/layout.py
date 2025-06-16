"""
Rich-based layout components for the RAGnificent TUI.

This module provides layout components for building terminal user interfaces
using the Rich library.
"""
from typing import Dict, List, Optional, Union

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich.text import Text

# Type aliases
RichRenderable = Union[str, Text, Panel, Table, "Progress"]


def create_help_panel(commands: Dict[str, str]) -> Panel:
    """Create a help panel with available commands.
    
    Args:
        commands: Dictionary mapping command keys to their descriptions.
        
    Returns:
        A Rich Panel containing the help information.
    """
    help_table = Table.grid(padding=(0, 2))
    help_table.add_column("Key", style="cyan", no_wrap=True)
    help_table.add_column("Action", style="white")
    
    for key, description in commands.items():
        help_table.add_row(f"[{key}]", description)
    
    return Panel(
        help_table,
        title="Commands",
        border_style="blue",
        padding=(1, 2),
    )


def create_status_panel(status: str, details: str = "") -> Panel:
    """Create a status panel with the current operation status.
    
    Args:
        status: The main status message.
        details: Optional additional details to display.
        
    Returns:
        A Rich Panel containing the status information.
    """
    content = Text()
    content.append(status, style="bold green")
    if details:
        content.append("\n\n" + details, style="dim")
    
    return Panel(
        content,
        title="Status",
        border_style="green",
        padding=(1, 2),
    )


def create_progress_display() -> Progress:
    """Create a progress display with multiple columns.
    
    Returns:
        A configured Rich Progress instance.
    """
    return Progress(
        SpinnerColumn(),
        "•",
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        "•",
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        "•",
        TimeRemainingColumn(),
        "•",
        TransferSpeedColumn(),
        console=Console(),
        refresh_per_second=10,
    )


def create_log_panel(log_entries: List[str], max_lines: int = 10) -> Panel:
    """Create a panel displaying log entries.
    
    Args:
        log_entries: List of log messages to display.
        max_lines: Maximum number of log lines to show.
        
    Returns:
        A Rich Panel containing the log entries.
    """
    if not log_entries:
        content = Text("No log entries yet", style="dim")
    else:
        content = Text("\n".join(log_entries[-max_lines:]))
    
    return Panel(
        content,
        title="Log",
        border_style="yellow",
        padding=(1, 2),
    )


def create_main_layout(
    help_commands: Dict[str, str],
    status: str = "Ready",
    status_details: str = "",
    log_entries: Optional[List[str]] = None,
) -> Layout:
    """Create the main application layout.
    
    Args:
        help_commands: Dictionary of command keys and descriptions for help panel.
        status: Initial status message.
        status_details: Additional status details.
        log_entries: Optional list of log entries to display.
        
    Returns:
        A configured Rich Layout instance.
    """
    if log_entries is None:
        log_entries = []
    
    layout = Layout()
    
    # Split into top (status and help) and bottom (progress and logs)
    layout.split_column(
        Layout(name="header", ratio=1),
        Layout(name="main", ratio=4),
    )
    
    # Header contains status and help side by side
    header_layout = layout["header"]
    header_layout.split_row(
        Layout(
            create_status_panel(status, status_details),
            name="status",
            ratio=2,
        ),
        Layout(
            create_help_panel(help_commands),
            name="help",
            ratio=1,
        ),
    )
    
    # Main area contains progress and logs
    main_layout = layout["main"]
    main_layout.split_row(
        Layout(
            name="progress",
            ratio=1,
        ),
        Layout(
            create_log_panel(log_entries),
            name="logs",
            ratio=1,
        ),
    )
    
    return layout
