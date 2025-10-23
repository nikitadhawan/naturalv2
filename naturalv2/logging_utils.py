"""Utilities for logging Rich tables.

This module centralises helpers for rendering key-value data in Rich tables and
forwarding the formatted output to both the active logger and, optionally, the
terminal.
"""

import logging
from io import StringIO
from typing import Any, Iterable, Mapping, Sequence

from rich.console import Console
from rich.pretty import Pretty
from rich.table import Table


def build_kv_table(
    title: str,
    rows: Sequence[tuple[str, Any]] | Mapping[str, Any],
) -> Table:
    """Create a two-column Rich table from key-value pairs.

    Parameters
    ----------
    title : str
        Title shown above the rendered table.
    rows : sequence of tuple or mapping
        Key-value data to populate the table. Any values that are not strings or
        :class:`rich.pretty.Pretty` instances are wrapped in ``Pretty`` with
        ``expand_all=True`` for readability.

    Returns
    -------
    rich.table.Table
        Configured Rich table ready for printing or logging. An empty input
        results in a table with a single ``"--"`` placeholder row.

    See Also
    --------
    emit_table : Render a table and log the textual representation.
    """

    table = Table(title=title)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="magenta")

    for key, value in _iter_rows(rows):
        if isinstance(value, (str, Pretty)):
            rendered_value = value
        else:
            rendered_value = Pretty(value, expand_all=True)
        table.add_row(str(key), rendered_value)

    if table.row_count == 0:
        table.add_row("--", "")

    return table


def emit_table(
    table: Table,
    *,
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
    render_console: bool = False,
    console: Console | None = None,
) -> str:
    """Render a table to text and log it.

    Parameters
    ----------
    table : rich.table.Table
        Table to render.
    logger : logging.Logger, optional
        Logger used to emit the table output. Defaults to
        ``logging.getLogger(__name__)`` when omitted.
    level : int, optional
        Logging level used for the emitted record. Defaults to ``logging.INFO``.
    render_console : bool, optional
        When ``True`` the table is also printed directly to ``console``; by
        default the table is only logged.
    console : rich.console.Console, optional
        Console instance used for optional on-screen rendering. If not
        provided, a new ``Console`` is instantiated.

    Returns
    -------
    str
        The textual representation of ``table`` generated with
        :class:`rich.console.Console.export_text`.

    Notes
    -----
    A separate in-memory console is used for capture so that logging handlers
    control where the text is written. The ``render_console`` flag allows the
    caller to mirror the same table directly to stdout/stderr when desired.
    """
    logger = logger or logging.getLogger(__name__)
    console = console or Console()

    capture_buffer = StringIO()
    capture = Console(
        record=True,
        width=console.width,
        file=capture_buffer,
        force_terminal=False,
        color_system=None,
    )
    capture.print(table)
    table_text = capture.export_text().rstrip()

    if render_console:
        console.print(table)

    logger.log(level, "\n%s", table_text)
    return table_text


def _iter_rows(
    rows: Sequence[tuple[str, Any]] | Mapping[str, Any],
) -> Iterable[tuple[str, Any]]:
    """Yield ``(key, value)`` pairs from a mapping or sequence.

    Parameters
    ----------
    rows : sequence of tuple or mapping
        Either a sequence of ``(key, value)`` tuples or a mapping that can be
        iterated over. Non-mapping inputs are returned unchanged.

    Returns
    -------
    iterable of tuple
        Iterator yielding ``(key, value)`` pairs suitable for table
        construction.
    """

    if isinstance(rows, Mapping):
        return rows.items()
    return rows
