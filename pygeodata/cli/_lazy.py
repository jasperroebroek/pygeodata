import importlib
from typing import Any

import click


class LazyGroup(click.Group):
    """A group that imports each subcommand's module only when that subcommand runs.

    ``lazy_commands`` maps a command name to ``('module:attribute', short_help)``. The short help
    is listed by ``--help`` without importing the command; a test keeps it equal to the
    command's own docstring summary.
    """

    def __init__(self, *args: Any, lazy_commands: dict[str, tuple[str, str]], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lazy_commands = lazy_commands

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted([*super().list_commands(ctx), *self.lazy_commands])

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if cmd_name not in self.lazy_commands:
            return super().get_command(ctx, cmd_name)
        module_name, attribute = self.lazy_commands[cmd_name][0].split(':')
        return getattr(importlib.import_module(module_name), attribute)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        rows = [
            (name, self.lazy_commands[name][1])
            if name in self.lazy_commands
            else (name, self.commands[name].get_short_help_str(formatter.width))
            for name in self.list_commands(ctx)
        ]
        if rows:
            with formatter.section('Commands'):
                formatter.write_dl(rows)
