"""Just enough of Immich's response shapes to be useful, and no more.

Deliberately tolerant: every field is pulled with ``.get`` and unknown keys are
ignored, because Immich's API moves and a photo picker has no business breaking
when a new field shows up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Album:
    id: str
    name: str
    asset_count: int = 0

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Album:
        return cls(
            id=str(data["id"]),
            name=str(data.get("albumName") or data.get("name") or "(untitled)"),
            asset_count=int(data.get("assetCount") or 0),
        )


@dataclass(frozen=True)
class Person:
    id: str
    name: str
    hidden: bool = False

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Person:
        return cls(
            id=str(data["id"]),
            name=str(data.get("name") or ""),
            hidden=bool(data.get("isHidden") or False),
        )


@dataclass(frozen=True)
class Asset:
    id: str
    type: str = "IMAGE"
    original_file_name: str = ""
    width: int = 0
    height: int = 0
    is_favorite: bool = False
    local_date_time: str = ""
    people: tuple[str, ...] = ()
    exif: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Asset:
        exif = data.get("exifInfo") or {}
        return cls(
            id=str(data["id"]),
            type=str(data.get("type") or "IMAGE"),
            original_file_name=str(data.get("originalFileName") or ""),
            # Newer servers report dimensions on the asset; older ones only in EXIF.
            width=int(data.get("width") or exif.get("exifImageWidth") or 0),
            height=int(data.get("height") or exif.get("exifImageHeight") or 0),
            is_favorite=bool(data.get("isFavorite") or False),
            local_date_time=str(data.get("localDateTime") or data.get("fileCreatedAt") or ""),
            people=tuple(
                str(p.get("name") or "")
                for p in (data.get("people") or [])
                if p.get("name")
            ),
            exif=exif,
        )

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000 if self.width and self.height else 0.0

    def describe(self) -> str:
        bits = [self.original_file_name or self.id]
        if self.local_date_time:
            bits.append(self.local_date_time[:10])
        if self.width and self.height:
            bits.append(f"{self.width}x{self.height}")
        if self.people:
            bits.append("with " + ", ".join(self.people))
        return " | ".join(bits)
