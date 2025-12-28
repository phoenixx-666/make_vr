from datetime import datetime
import json
import os
from pathlib import Path
from pydantic import Field, PrivateAttr, ValidationError, field_serializer, field_validator
from typing import Optional

from .shell import CLIArgs, warning
from .tools import NEStr, SingletonModel, ValidatedModel


__all__ = ['Cache']


_cache_path = str(Path(os.getcwd()) / 'make_vr.cache.json')
SyncDict = dict[tuple[NEStr, NEStr], float]
SyncDictSerialized = list[tuple[str, str, float]]


class Cache(ValidatedModel, metaclass=SingletonModel):
    modified: dict[NEStr, datetime] = Field(default_factory=dict, alias='m')
    created: dict[NEStr, Optional[datetime]] = Field(default_factory=dict, alias='c')
    sync: SyncDict = Field(default_factory=dict, alias='s')

    _updated: bool = PrivateAttr(default=False)
    _ignore: Optional[tuple[str, ...]] = PrivateAttr()

    def __init__(self):
        try:
            json_data: dict = {}
            if os.path.isfile(_cache_path):
                with open(_cache_path, 'rb') as f:
                    json_data = json.load(f)
            super().__init__(**json_data)
            self._cleanup()

        except (IOError, OSError, json.JSONDecodeError, ValidationError) as e:
            warning(f'Unable to read cache: {e}')
            super().__init__()

        self._ignore = CLIArgs().ignore_cache

    def get_modified(self, file_name: str) -> datetime | None:
        return self.modified.get(file_name)

    def set_match(self, file_name: str, modified_date: datetime, creation_date: datetime | None):
        if (file_name not in self.created) or (self.modified.get(file_name) != modified_date):
            self.created[file_name] = creation_date
            self._set_modified(file_name, modified_date)
            self._updated = True

    def get_match(self, file_name: str) -> datetime | None:
        return self.created.get(file_name)

    def set_sync(self, file_name1: str, modified_date1: datetime, file_name2: str, modified_date2: datetime, offset: float):
        if (self.modified.get(file_name1) != modified_date1) or \
            (self.modified.get(file_name2) != modified_date2) or \
            (self._find_offset(file_name1, file_name2) != offset):
            self.sync[(file_name1, file_name2)] = offset
            self.sync.pop((file_name2, file_name1), None)
            self._set_modified(file_name1, modified_date1)
            self._set_modified(file_name2, modified_date2)
            self._updated = True

    def get_sync(self, file_name1: str, file_name2: str):
        if (sync := self._find_offset(file_name1, file_name2)) is not None:
            return sync

    def save_if_updated(self):
        if self.is_ignoring('save') or not self._updated:
            return

        try:
            encoded_json = self.model_dump_json(by_alias=True, ensure_ascii=True, indent=4)
            with open(_cache_path, 'w') as f:
                f.write(encoded_json)
            self._updated = False
        except (IOError, OSError, UnicodeEncodeError) as e:
            warning(f'There was an error while saving cache: {e}')

    def is_ignoring(self, name: str) -> bool:
        if self._ignore is None:
            return False
        return (not self._ignore) or (name in self._ignore)

    def _set_modified(self, file_name: str, modified_date: datetime):
        self.modified[file_name] = modified_date

    def _find_offset(self, file_name1: str, file_name2: str) -> float | None:
        if (sync := self.sync.get((file_name1, file_name2))) is not None:
            return sync
        elif (sync := self.sync.get((file_name2, file_name1))) is not None:
            return -sync
        return None

    def _cleanup(self):
        if nonexistent := {fn for fn in self.modified.keys() if not os.path.isfile(fn)}:
            remove_sync: set[tuple[str, str]] = set()
            for file_name in nonexistent:
                del self.modified[file_name]
                self.created.pop(file_name, None)
                remove_sync |= (file_names for file_names in self.sync.keys() if file_name in file_names)
            for file_names in remove_sync:
                del self.sync[file_names]
            self._updated = True

    @field_serializer('sync')
    @staticmethod
    def _serialize_sync(value: SyncDict) -> SyncDictSerialized:
        return [(fn1, fn2, offset) for (fn1, fn2), offset in value.items()]

    @field_validator('sync', mode='before')
    @staticmethod
    def _validate_sync(value: SyncDictSerialized) -> SyncDict:
        # TODO: validate structure
        return {(fn1, fn2): offset for fn1, fn2, offset in value}
