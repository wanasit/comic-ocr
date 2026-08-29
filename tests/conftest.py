import shutil
import pytest
import torch
from comic_ocr.utils import files


@pytest.fixture(autouse=True)
def mock_torch_hub_download(monkeypatch):
    """Mocks torch.hub.download_url_to_file to use local trained_models weights during tests."""
    orig_download = torch.hub.download_url_to_file

    def _mock_download(url, dst, hash_prefix=None, progress=True):
        if 'localization.pth' in url or 'localization.pth' in str(dst):
            src = files.get_path_project_dir('trained_models/localization.pth')
            shutil.copyfile(src, dst)
        elif 'recognition.pth' in url or 'recognition.pth' in str(dst):
            src = files.get_path_project_dir('trained_models/recognition.pth')
            shutil.copyfile(src, dst)
        else:
            orig_download(url, dst, hash_prefix=hash_prefix, progress=progress)

    monkeypatch.setattr(torch.hub, 'download_url_to_file', _mock_download)
