try:
    from pygeodata.registry_browser.serve import open_registry_browser
except ImportError as _err:
    _missing = str(_err)

    def open_registry_browser() -> None:
        raise ImportError(
            f'The dashboard extra is required: pip install pygeodata[dashboard]. Original error: {_missing}',
        )


__all__ = ['open_registry_browser']
