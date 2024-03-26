from typing import Optional, Dict

from data_framework.types import RasterLoader


class LoaderRegistry:
    _instance: Optional['LoaderRegistry'] = None
    _loaders: Dict[str, RasterLoader] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def register_loader(self, name: str, loader: RasterLoader) -> None:
        if name in self._loaders:
            raise ValueError(f"Name already registered {name=}")
        if loader in self._loaders.values():
            raise ValueError(f"Loader already registered {loader=}")
        self._loaders[name] = loader

    def get_loader(self, name: str):
        if name not in self._loaders:
            raise ValueError(f"No such loader: {name=}")
        return self._loaders.get(name)
