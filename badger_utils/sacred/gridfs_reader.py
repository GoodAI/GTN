from typing import List

from gridfs import GridFS, GridOut

from badger_utils.sacred.natural_sort import NaturalSort


class GridFSReader:

    def __init__(self, fs: GridFS):
        self.fs = fs

    @staticmethod
    def _artifact_prefix(run_id: int) -> str:
        return f'artifact://runs/{run_id}/'

    def list_artifacts(self, run_id: int) -> List[str]:
        prefix = self._artifact_prefix(run_id)
        files = self.fs.find({'filename': {'$regex': prefix}})
        return [self.strip_artifact_prefix(run_id, f.name) for f in files]

    def list_files(self, run_id: int, sort: bool = False) -> List[GridOut]:
        prefix = self._artifact_prefix(run_id)
        files = list(self.fs.find({'filename': {'$regex': prefix}}))
        return sorted(files, key=lambda x: NaturalSort.natural_keys(x.name)) if sort else files

    def strip_artifact_prefix(self, run_id: int, filename: str) -> str:
        prefix = self._artifact_prefix(run_id)
        return filename[len(prefix):]

    def read_artifact(self, run_id: int, filename: str) -> bytes:
        file = self.fs.find({'filename': f'{self._artifact_prefix(run_id)}{filename}'})[0]
        return file.read()

# import io
# import torch
# from badger_utils.sacred import SacredConfigFactory
#
# observer = SacredConfigFactory.shared().create_mongo_observer()
#
# reader = GridFSReader(observer.fs)
# run_id = 946
#
# files = reader.list_artifacts(run_id)
# print(f'Files: \n{files}')
# item = torch.load(io.BytesIO(reader.read_artifact(run_id, 'agent_ep_1000.model')))
# print(type(item))
