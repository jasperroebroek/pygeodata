# tests/test_tracked_object_dependencies.py
import ast
import importlib.util
import sys
import textwrap
from collections.abc import Generator
from pathlib import Path

import pytest

from pygeodata.tracked_object import TrackedObject


def import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse(code: str) -> ast.AST:
    return ast.parse(textwrap.dedent(code))


def make_tracked_class(
    name: str,
    code: str,
    bases: tuple[type[TrackedObject], ...] = (TrackedObject,),
) -> type[TrackedObject]:
    namespace = {}

    @classmethod
    def get_source_ast_tree(cls) -> ast.AST:
        return parse(code)

    namespace['get_source_ast_tree'] = get_source_ast_tree
    return type(name, bases, namespace)


@pytest.fixture(autouse=True)
def reset_registry() -> Generator[None, None, None]:
    saved = dict(TrackedObject._registry)
    TrackedObject._registry = {}
    TrackedObject.clear_function_caches()
    yield
    TrackedObject._registry = saved
    TrackedObject.clear_function_caches()


def test_direct_instantiation_dependency() -> None:
    loader = make_tracked_class('Loader', 'class Loader: pass')
    user = make_tracked_class(
        'User',
        """
        class User:
            def build(self):
                return Loader()
        """,
    )

    assert user.get_call_dependencies() == {loader}


def test_bare_class_reference_counts_as_dependency() -> None:
    loader = make_tracked_class('Loader', 'class Loader: pass')
    user = make_tracked_class(
        'User',
        """
        class User:
            dependency = Loader
        """,
    )

    assert user.get_call_dependencies() == {loader}


def test_class_method_access_counts_as_dependency() -> None:
    loader = make_tracked_class('Loader', 'class Loader: pass')
    user = make_tracked_class(
        'User',
        """
        class User:
            def name(self):
                return Loader.get_class_name()
        """,
    )

    assert user.get_call_dependencies() == {loader}


def test_import_alias_resolves_to_original_class_name(tmp_path: Path) -> None:
    package_dir = tmp_path / 'dummy_pkg'
    package_dir.mkdir()

    (package_dir / '__init__.py').write_text('', encoding='utf-8')

    a_py = package_dir / 'a.py'
    a_py.write_text(
        textwrap.dedent(
            """
            from pygeodata.data import Data


            class Loader(Data):
                pass
            """,
        ),
        encoding='utf-8',
    )

    user_py = package_dir / 'user_module.py'
    user_py.write_text(
        textwrap.dedent(
            """
            from pygeodata.data import Data
            from dummy_pkg.a import Loader as L


            class User(Data):
                def build(self):
                    return L()
            """,
        ),
        encoding='utf-8',
    )

    a_module = import_module_from_path('dummy_pkg.a', a_py)
    user_module = import_module_from_path('dummy_pkg.user_module', user_py)

    loader = a_module.Loader
    user = user_module.User

    loader.clear_function_caches()
    user.clear_function_caches()

    assert user.get_call_dependencies() == {loader}


def test_module_alias_attribute_access_resolves_class_name(tmp_path: Path) -> None:
    package_dir = tmp_path / 'pkg'
    package_dir.mkdir()

    (package_dir / '__init__.py').write_text('', encoding='utf-8')

    mod_dir = package_dir / 'mod'
    mod_dir.mkdir()

    (mod_dir / '__init__.py').write_text(
        textwrap.dedent(
            """
            from pygeodata.data import Data


            class Loader(Data):
                pass
            """,
        ),
        encoding='utf-8',
    )

    user_py = package_dir / 'user_module.py'
    user_py.write_text(
        textwrap.dedent(
            """
            from pygeodata.data import Data
            import pkg.mod as m


            class User(Data):
                def build(self):
                    return m.Loader.get_class_name()
            """,
        ),
        encoding='utf-8',
    )

    sys.path.insert(0, str(tmp_path))
    try:
        importlib.invalidate_caches()

        loader_module = importlib.import_module('pkg.mod')
        user_module = importlib.import_module('pkg.user_module')

        loader = loader_module.Loader
        user = user_module.User

        loader.clear_function_caches()
        user.clear_function_caches()

        assert user.get_call_dependencies() == {loader}
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop('pkg.user_module', None)
        sys.modules.pop('pkg.mod', None)
        sys.modules.pop('pkg', None)


def test_mixed_reference_forms_are_deduplicated() -> None:
    loader = make_tracked_class('Loader', 'class Loader: pass')
    user = make_tracked_class(
        'User',
        """
        from a import Loader as L
        import pkg.mod as m

        class User:
            x = Loader

            def build(self):
                L()
                m.Loader.get_class_name()
                return Loader()
        """,
    )

    assert user.get_call_dependencies() == {loader}


def test_self_reference_is_excluded() -> None:
    user = make_tracked_class(
        'User',
        """
        class User:
            def build(self):
                return User()
        """,
    )

    assert user.get_call_dependencies() == set()


def test_unknown_names_are_ignored() -> None:
    loader = make_tracked_class('Loader', 'class Loader: pass')
    user = make_tracked_class(
        'User',
        """
        class User:
            def build(self):
                UnknownThing()
                maybe = SomethingElse
                return Loader()
        """,
    )

    assert user.get_call_dependencies() == {loader}


def test_multiple_dependencies_are_collected() -> None:
    loader = make_tracked_class('Loader', 'class Loader: pass')
    writer = make_tracked_class('Writer', 'class Writer: pass')
    user = make_tracked_class(
        'User',
        """
        class User:
            def build(self):
                Loader()
                Writer.get_class_name()
                return Writer
        """,
    )

    assert user.get_call_dependencies() == {loader, writer}


def test_inheritance_dependencies_are_combined_with_source_dependencies() -> None:
    base = make_tracked_class('Base', 'class Base: pass')
    helper = make_tracked_class('Helper', 'class Helper: pass')
    child = make_tracked_class(
        'Child',
        """
        class Child(Base):
            def build(self):
                return Helper()
        """,
        bases=(base,),
    )

    assert child.get_inheritance_dependencies() == {base}
    assert child.get_call_dependencies() == {helper}
    assert child.get_all_dependencies() == {base, helper}


def test_duplicate_class_name_is_rejected() -> None:
    make_tracked_class('Loader', 'class Loader: pass')

    with pytest.raises(ValueError, match='Duplicate TrackedObject class name'):
        make_tracked_class('Loader', 'class Loader: pass')
