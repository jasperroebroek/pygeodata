import lazy_loader as lazy

# Public API resolved on first access, declared in __init__.pyi: `import pygeodata.cli` must stay light.
__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
