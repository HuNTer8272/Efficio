"""Storage in a Hugging Face Hub model repo (private by default). Needs the optional
huggingface_hub package: pip install "efficio[hub]"."""

import os
import shutil
import tempfile

from .base import Storage, StorageError, redact
from .hf_token import HOW_TO_ADD, find_hf_token

INSTALL = 'Saving to the Hugging Face Hub needs huggingface_hub: pip install "efficio[hub]"'


def _hub():
    try:
        import huggingface_hub
    except ImportError:
        raise ImportError(INSTALL) from None
    return huggingface_hub


class HuggingFaceHub(Storage):
    """Checkpoints in the Hugging Face repo `repo_id` ("user/name"), optionally in a folder
    of it (`path`). The repo is created as private if it doesn't exist. The token comes from
    `token`, the HF_TOKEN environment variable, or Colab or Kaggle secrets named HF_TOKEN;
    it is never printed, logged or written to any file."""

    def __init__(self, repo_id, token=None, path="", private=True):
        parts = repo_id.strip("/").split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError(f'repo_id must look like "user/repo-name", not {repo_id!r}')
        self.repo_id = "/".join(parts)
        self.path = path.strip("/")
        self.private = private
        self._token, self.token_source = find_hf_token(token)
        self._api = None

    def describe(self):
        return f"hf://{self.repo_id}" + (f"/{self.path}" if self.path else "")

    def redact(self, text):
        return redact(text, [self._token])

    def __getstate__(self):                  # never pickle the token or the client
        return {**self.__dict__, "_token": None, "_api": None}

    def _client(self):
        if self._api is None:
            self._api = _hub().HfApi(token=self._token)
        return self._api

    def _full(self, name):
        return f"{self.path}/{name}" if self.path else name

    def connect(self, log=print):
        api = self._client()
        try:
            api.create_repo(self.repo_id, repo_type="model", private=self.private, exist_ok=True)
        except Exception as e:
            hint = "" if self._token else "\n" + HOW_TO_ADD
            raise StorageError(f"Could not open or create the Hugging Face repo {self.repo_id} "
                               f"({type(e).__name__}: {self.redact(e)}).{hint}") from None
        try:
            private = api.repo_info(self.repo_id, repo_type="model").private
        except Exception:
            private = None
        log(f"[efficio] Hugging Face token: from {self.token_source}")
        if private is False:
            log(f"[efficio] WARNING: {self.repo_id} is a PUBLIC repo; anyone can download "
                "these checkpoints. Make it private in the repo settings.")

    def upload(self, local_path, name):
        self._client().upload_file(path_or_fileobj=local_path, path_in_repo=self._full(name),
                                   repo_id=self.repo_id, repo_type="model",
                                   commit_message=f"efficio: add {name}")

    def download(self, name, local_path):
        folder = os.path.dirname(os.path.abspath(local_path))
        tmp = tempfile.mkdtemp(prefix=".efficio-download-", dir=folder)
        try:
            got = _hub().hf_hub_download(repo_id=self.repo_id, filename=self._full(name),
                                         repo_type="model", token=self._token, local_dir=tmp,
                                         force_download=True)
            shutil.move(got, local_path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def list(self):
        files = self._client().list_repo_files(self.repo_id, repo_type="model")
        prefix = f"{self.path}/" if self.path else ""
        return sorted(f[len(prefix):] for f in files
                      if f.startswith(prefix) and "/" not in f[len(prefix):])

    def delete(self, name):
        self._client().delete_file(self._full(name), repo_id=self.repo_id, repo_type="model",
                                   commit_message=f"efficio: remove {name}")
