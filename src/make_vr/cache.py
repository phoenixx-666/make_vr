from datetime import datetime
import json
import os
from pathlib import Path
from pydantic import BaseModel, Field, RootModel
from typing import Optional

from .tools import NEStr, SingletonModel


__all__ = ['Cache']


class FileData(BaseModel):
    modified_date: datetime = Field(alias='m')
    creation_date: Optional[datetime] = Field(alias='c', default=None)


class Cache(SingletonModel, RootModel[dict[NEStr, FileData]]):
    def __init__(self):
        self._cache_path = cache_path = str(Path(os.getcwd()) / 'make_vr.cache.json')
        self._checked: set[str] = set()
        self._updated = False

        json_data: dict = {}
        if not os.path.isfile(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    json_data = json.load(f)

            except (IOError, OSError, json.JSONDecodeError) as e:
                print(f'Unable to read cache: {e}')

        super().__init__(root=json_data)

    def set_match(self, file_name: str, modified_date: datetime, creation_date: datetime | None):
        if (not (file_data := self.root.get('file_name'))) or file_data.modified_date != modified_date:
            self.root[file_name] = FileData(m=modified_date, c=creation_date)
            self._checked.add(file_name)
            self._updated = True

    def get_match(self, file_name: str) -> FileData | None:
        if result := self.root.get(file_name):
            self._checked.add(file_name)
            return result.model_copy()
        return None

    def cleanup(self):
        if nonexistent := {fn for fn in self.root.keys() if fn not in self._checked and not os.path.isfile(fn)}:
            for file_name in nonexistent:
                del self.root[file_name]
            self._updated = True

    def save_if_updated(self):
        if not self._updated:
            return

        try:
            encoded_json = self.model_dump_json(ensure_ascii=True, indent=4)
            with open(self._cache_path, 'w') as f:
                f.write(encoded_json)
        except (IOError, OSError, UnicodeEncodeError) as e:
            print(f'There was an error while saving cache: {e}')
