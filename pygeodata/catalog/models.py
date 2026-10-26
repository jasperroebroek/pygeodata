"""SQLAlchemy models — the single source of truth for the ``.source/catalog.db`` schema."""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, MetaData, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            'ix': 'ix_%(column_0_label)s',
            'uq': 'uq_%(table_name)s_%(column_0_name)s',
            'ck': 'ck_%(table_name)s_%(constraint_name)s',
            'fk': 'fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s',
            'pk': 'pk_%(table_name)s',
        },
    )


class CatalogMeta(Base):
    __tablename__ = 'meta'

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(String)


class Crs(Base):
    __tablename__ = 'crs'

    crs_hash: Mapped[str] = mapped_column(String, primary_key=True)
    crs: Mapped[str] = mapped_column(Text)


class Resolution(Base):
    __tablename__ = 'resolutions'

    transform_hash: Mapped[str] = mapped_column(String, primary_key=True)
    res_x: Mapped[float | None] = mapped_column(Float)
    res_y: Mapped[float | None] = mapped_column(Float)


class Bounds(Base):
    __tablename__ = 'bounds'

    bounds_hash: Mapped[str] = mapped_column(String, primary_key=True)
    left: Mapped[float | None] = mapped_column(Float)
    bottom: Mapped[float | None] = mapped_column(Float)
    right: Mapped[float | None] = mapped_column(Float)
    top: Mapped[float | None] = mapped_column(Float)


class Shape(Base):
    __tablename__ = 'shapes'

    shape_hash: Mapped[str] = mapped_column(String, primary_key=True)
    shape: Mapped[str] = mapped_column(Text)


class Entry(Base):
    __tablename__ = 'entries'

    state_hash: Mapped[str] = mapped_column(String, primary_key=True)
    class_name: Mapped[str] = mapped_column(Text, nullable=False)
    object_type: Mapped[str | None] = mapped_column(Text)
    source_hash: Mapped[str | None] = mapped_column(String)
    dependency_tree_hash: Mapped[str | None] = mapped_column(String)
    instance_hash: Mapped[str | None] = mapped_column(String)
    params_hash: Mapped[str | None] = mapped_column(String)
    spec_hash: Mapped[str | None] = mapped_column(String)
    crs_hash: Mapped[str | None] = mapped_column(ForeignKey('crs.crs_hash'))
    transform_hash: Mapped[str | None] = mapped_column(ForeignKey('resolutions.transform_hash'))
    bounds_hash: Mapped[str | None] = mapped_column(ForeignKey('bounds.bounds_hash'))
    shape_hash: Mapped[str | None] = mapped_column(ForeignKey('shapes.shape_hash'))
    co_output_hashes: Mapped[str | None] = mapped_column(Text)
    dir_relpath: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[float | None] = mapped_column(Float)
    updated_at: Mapped[float | None] = mapped_column(Float)
