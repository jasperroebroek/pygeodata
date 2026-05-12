import ast
import hashlib
import inspect
import textwrap


def define_hash_from_class(cls: type) -> str:
    """Formatting-agnostic hash of a class using AST dump."""
    try:
        source = inspect.getsource(cls)
        tree = ast.parse(textwrap.dedent(source))
    except (TypeError, OSError) as err:
        raise OSError(
            'AST Parsing failed. Caching is disabled. You are likely in a REPL/Notebook environment. Use standard .py files.',
        ) from err
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()
