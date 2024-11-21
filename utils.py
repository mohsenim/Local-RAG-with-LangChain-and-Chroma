import tempfile
from pathlib import Path


def copy_files_to_dir(path, docs):
    for doc in docs:
        suffix = Path(doc.name).suffix
        with tempfile.NamedTemporaryFile(
            dir=path.as_posix(), suffix=suffix, delete=False
        ) as tmp_file:
            tmp_file.write(doc.read())


def empty_dir(path):
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            item.rmdir()
