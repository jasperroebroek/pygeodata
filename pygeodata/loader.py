import re
from pathlib import Path
from typing import Any

from pygeodata.config import get_config
from pygeodata.paths import generate_path
from pygeodata.types import Driver, Processor, SpatialSpec


class DataLoader:
    @property
    def processor(self) -> Processor:
        raise NotImplementedError('Either processor or process must be implemented')

    @property
    def driver(self) -> Driver:
        try:
            processor = self.processor
        except NotImplementedError:
            raise NotImplementedError('Either processor or driver must be implemented')

        if not hasattr(processor, 'default_driver'):
            raise AttributeError(f'Processor {processor} lacks default_driver and no driver is set')
        return getattr(processor, 'default_driver')

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

    def get_params(self) -> dict[str, Any]:
        params = {}
        for key in self.__dict__:
            if key in ('name', 'class_name', 'processor', 'driver'):
                continue
            if key.startswith('_'):
                continue
            params.update({key: self.__dict__[key]})
        return params

    def __repr__(self) -> str:
        params = self.get_params()
        parts = [f'{k}={v!r}' for k, v in sorted(params.items())]
        return f'{self.class_name}({", ".join(parts)})'

    def get_processed_path(self, spec: SpatialSpec) -> Path:
        path = generate_path(
            spec=spec,
            name=self.class_name,
            filename=self.name,
            base_dir=get_config().path_data_processed,
            **self.get_params(),
        )
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

    def is_processed(self, spec: SpatialSpec) -> bool:
        p = self.get_processed_path(spec)
        return p.exists()

    def process(self, spec: SpatialSpec) -> None:
        if not self.is_processed(spec):
            self.processor(self.get_processed_path(spec), spec)

    def load(self, spec: SpatialSpec) -> Any:
        self.process(spec)
        return self.driver.load(self.get_processed_path(spec))
