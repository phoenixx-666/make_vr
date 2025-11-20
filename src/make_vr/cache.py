from dataclasses import dataclass
from datetime import datetime
import json
import jsonschema
import os
from pathlib import Path
from typing import ClassVar

from .tools import make_object, prop_nestr


__all__ = ['Cache']


class Cache:
    @dataclass
    class FileData:
        modified_date: datetime
        creation_date: datetime | None

        def as_json(self) -> dict:
            result = {'m': self.modified_date.isoformat()}
            if self.creation_date:
                result['c'] = self.creation_date.isoformat()
            return result

    def __init__(self):
        self._cache_path = cache_path = str(Path(os.getcwd()) / 'make_vr.cache.json')
        self._data: dict[str, Cache.FileData] = {}
        self._checked: set[str] = set()
        self._updated = False

        if not os.path.isfile(cache_path):
            return

        try:
            with open(cache_path, 'rb') as f:
                json_data: dict[str, dict] = json.load(f)
                jsonschema.validate(json_data, self._schema)

            for file_name, file_data in json_data.items():
                if creation_date := file_data.get('c'):
                    creation_date = datetime.fromisoformat(creation_date)
                self._data[file_name] = Cache.FileData(datetime.fromisoformat(file_data['m']), creation_date)

        except (IOError, OSError, json.JSONDecodeError) as e:
            print(f'Unable to read cache: {e}')
            self._data = {}
            return

    def set(self, file_name: str, modified_date: datetime, creation_date: datetime | None):
        if (not (file_data := self._data.get('file_name'))) or file_data.modified_date != modified_date:
            self._data[file_name] = Cache.FileData(modified_date, creation_date)
            self._checked.add(file_name)
            self._updated = True

    def get(self, file_name: str) -> Cache.FileData | None:
        if result := self._data.get(file_name):
            self._checked.add(file_name)
            return Cache.FileData(result.modified_date, result.creation_date)
        return None

    def cleanup(self):
        if nonexistent := {fn for fn in self._data.keys() if fn not in self._checked and not os.path.isfile(fn)}:
            for file_name in nonexistent:
                del self._data[file_name]
            self._updated = True

    def save_if_updated(self):
        if not self._updated:
            return

        json_data = {file_name: file_data.as_json() for file_name, file_data in self._data.items()}

        try:
            encoded_json = json.dumps(json_data, ensure_ascii=True, indent=4)
            with open(self._cache_path, 'w') as f:
                f.write(encoded_json)
        except (IOError, OSError, UnicodeEncodeError) as e:
            print(f'There was an error while saving cache: {e}')

    _schema: ClassVar[dict] = make_object(pattern_properties={
        r'.*' : make_object({ 'm': prop_nestr, 'c': prop_nestr,}, required=['m'])
    })
