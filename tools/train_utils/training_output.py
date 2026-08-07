import getpass
import os
import re
import socket
from pathlib import Path


SHARED_OUTPUT_ENV = 'HR4D_TRAINING_OUTPUT_ROOT'
DEFAULT_SHARED_OUTPUT_ROOT = Path('/mnt/nvme-ai-data/4DRadar/output')


def _resolve_path(path, project_root):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (Path(project_root) / path).resolve()


def _path_component(value, fallback):
    value = re.sub(r'[^A-Za-z0-9_.-]+', '-', str(value).strip()).strip('-.')
    return value or fallback


def get_training_source():
    source_user = (
        os.environ.get('HR4D_TRAINING_USER')
        or os.environ.get('USER')
        or os.environ.get('LOGNAME')
        or getpass.getuser()
    )
    source_container = (
        os.environ.get('HR4D_TRAINING_CONTAINER')
        or os.environ.get('HOSTNAME')
        or socket.gethostname()
    )
    source_host = os.environ.get('HR4D_TRAINING_HOST') or socket.gethostname()
    return {
        'source_user': source_user,
        'source_container': source_container,
        'source_host': source_host,
    }


def resolve_training_output_dir(configured_root, project_root, exp_group_path, tag, extra_tag):
    """Resolve one training run directory, preferring the server-wide log root."""

    configured_shared_root = os.environ.get(SHARED_OUTPUT_ENV, '').strip()
    if configured_shared_root:
        output_root = _resolve_path(configured_shared_root, project_root)
        shared = True
    elif DEFAULT_SHARED_OUTPUT_ROOT.is_dir():
        output_root = DEFAULT_SHARED_OUTPUT_ROOT
        shared = True
    else:
        output_root = _resolve_path(configured_root, project_root)
        shared = False

    shared_output_root = output_root
    source = get_training_source()
    if shared:
        output_root = (
            output_root
            / _path_component(source['source_user'], 'unknown-user')
            / _path_component(source['source_container'], 'unknown-container')
        )

    output_dir = output_root / exp_group_path / tag / extra_tag
    metadata = {
        **source,
        'shared_output': shared,
        'training_output_root': str(shared_output_root),
    }
    return output_dir, metadata
