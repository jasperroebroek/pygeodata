import re
from pathlib import Path
from typing import Any, Generic

from pygeodata.config import get_config
from pygeodata.paths import generate_path
from pygeodata.types import Driver, Processor, SpatialSpec, T


class DataLoader(Generic[T]):
    def resolve_spec(self, spec: SpatialSpec) -> SpatialSpec:
        if spec.is_fully_defined:
            return spec
        resolver = getattr(self.processor, 'resolve_spec', None)
        return resolver(spec) if resolver is not None else spec

    @property
    def processor(self) -> Processor | None:
        return None

    @property
    def driver(self) -> Driver:
        processor = self.processor
        if processor is None:
            raise NotImplementedError(f'{self}: Either processor or driver must be implemented')

        driver = getattr(processor, 'default_driver')
        if driver is None:
            raise AttributeError(f'Processor {processor} lacks default_driver and no driver is set')

        return driver

    @property
    def class_name(self) -> str:
        return self.__class__.__name__.split('.')[-1].replace('Loader', '')

    @property
    def name(self) -> str:
        # Handle acronym → word transitions (e.g. XMLHTTPRequest → XML_Http_Request)
        s1 = re.sub('([A-Z]+)([A-Z][a-z])', r'\1_\2', self.class_name)
        # Handle normal camelCase → camel_Case
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        return s2.lower()

    @property
    def ext(self) -> str:
        ext = getattr(self.processor, 'ext', None)
        if ext is None:
            ext = self.driver.default_ext
        return ext

    def get_params(self) -> dict[str, Any]:
        params = {}
        for key in vars(self):
            if key in ('name', 'class_name', 'processor', 'driver', 'process', 'load'):
                continue
            if key.startswith('_'):
                continue
            params.update({key: vars(self)[key]})
        return params

    def __repr__(self) -> str:
        params = self.get_params()
        parts = [f'{k}={v!r}' for k, v in sorted(params.items())]
        return f'{self.class_name}({", ".join(parts)})'

    def get_src_path(self) -> Path:
        processor = self.processor
        if processor is None:
            raise NotImplementedError('Processor must be implemented to get src_path')

        if not hasattr(processor, 'src_path'):
            raise NotImplementedError(f'Processor {processor} lacks src_path')

        return Path(getattr(processor, 'src_path'))

    def get_processed_path(self, spec: SpatialSpec, ext: str | None = None) -> Path:
        spec = self.resolve_spec(spec)
        ext = ext or self.ext

        path = generate_path(
            spec=spec,
            name=self.class_name,
            filename=self.name,
            base_dir=get_config().path_data_processed,
            ext=ext,
            **self.get_params(),
        )
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

    def is_processed(self, spec: SpatialSpec) -> bool:
        p = self.get_processed_path(spec)
        return p.exists()

    def process(self, spec: SpatialSpec) -> None:
        spec = self.resolve_spec(spec)
        processor = self.processor
        if processor is None:
            raise NotImplementedError('Either load, processor or process must be implemented')
        processor(self.get_processed_path(spec), spec)

    def load(self, spec: SpatialSpec) -> T:
        spec = self.resolve_spec(spec)
        return self.driver(self.get_processed_path(spec))

    def __call__(self, spec: SpatialSpec) -> T:
        spec = self.resolve_spec(spec)
        if not self.is_processed(spec):
            self.process(spec)
        return self.load(spec)
