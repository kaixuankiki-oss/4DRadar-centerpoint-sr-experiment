import io
import os
import pickle
import re
from pathlib import Path
from typing import Any, Optional, Tuple


_OBJECT_URI_RE = re.compile(r'^(?P<scheme>tos|s3)://(?P<bucket>[^/]+)(?:/(?P<key>.*))?$')


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, 'get'):
        return config.get(key, default)
    return getattr(config, key, default)


def _cfg_bool(config: Any, key: str, default: bool = False) -> bool:
    value = _cfg_get(config, key, default)
    if isinstance(value, str):
        return value.lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)


def _normalize_key(value: Any) -> str:
    return str(value).replace('\\', '/').strip('/')


def _join_key(*parts: Any) -> str:
    return '/'.join(part for part in (_normalize_key(part) for part in parts) if part)


def parse_object_uri(path: Any) -> Optional[Tuple[str, str]]:
    match = _OBJECT_URI_RE.match(str(path).replace('\\', '/'))
    if not match:
        return None
    return match.group('bucket'), match.group('key') or ''


def storage_type_from_config(dataset_cfg: Any) -> str:
    storage_cfg = _cfg_get(dataset_cfg, 'STORAGE', {})
    storage_type = _cfg_get(storage_cfg, 'TYPE', None)
    if storage_type is None and _cfg_get(dataset_cfg, 'USE_TOS', False):
        storage_type = 'tos'
    return str(storage_type or 'file').lower()


class FileStorage:
    type = 'file'

    def __init__(self, root_path: Any, project_root: Optional[Any] = None):
        root = Path(root_path).expanduser()
        if not root.is_absolute():
            project_root = Path(project_root) if project_root is not None else Path.cwd()
            root = project_root / root
        self.root_path = root.resolve()

    def resolve_path(self, path: Any) -> Path:
        path = Path(path).expanduser()
        return path if path.is_absolute() else self.root_path / path

    def describe(self, path: Any) -> str:
        return str(self.resolve_path(path))

    def exists(self, path: Any) -> bool:
        return self.resolve_path(path).exists()

    def read_bytes(self, path: Any) -> bytes:
        with self.resolve_path(path).open('rb') as stream:
            return stream.read()

    def load_pickle(self, path: Any) -> Any:
        with self.resolve_path(path).open('rb') as stream:
            return pickle.load(stream)

    def fromfile(self, path: Any, dtype: Any):
        import numpy as np

        return np.fromfile(str(self.resolve_path(path)), dtype=dtype)

    def np_load(self, path: Any):
        import numpy as np

        return np.load(self.resolve_path(path))


class TOSStorage:
    type = 'tos'

    def __init__(
        self,
        root_path: Any,
        storage_cfg: Optional[Any] = None,
        logger: Optional[Any] = None,
        client: Optional[Any] = None,
    ):
        self.logger = logger
        self.storage_cfg = storage_cfg or {}
        self.tos_cfg = _cfg_get(self.storage_cfg, 'TOS', self.storage_cfg) or {}
        self._client = client
        self._tos_module = None

        uri_ref = parse_object_uri(root_path)
        cfg_bucket = (
            _cfg_get(self.tos_cfg, 'BUCKET', None)
            or _cfg_get(self.storage_cfg, 'BUCKET', None)
            or _cfg_get(self.storage_cfg, 'TOS_BUCKET', None)
            or os.getenv('TOS_BUCKET')
        )
        cfg_prefix = _cfg_get(self.tos_cfg, 'PREFIX', '')

        if uri_ref is not None:
            uri_bucket, uri_key = uri_ref
            self.bucket = uri_bucket
            self.root_path = _join_key(cfg_prefix, uri_key)
        else:
            self.bucket = cfg_bucket
            self.root_path = _join_key(cfg_prefix, root_path)
        self.remap_hr4d_original_paths = _cfg_bool(
            self.tos_cfg, 'REMAP_HR4D_ORIGINAL_PATHS', True
        )

        if not self.bucket:
            raise ValueError(
                'TOS storage requires DATA_CONFIG.STORAGE.TOS.BUCKET, '
                'DATA_CONFIG.DATA_PATH as tos://bucket/key, or TOS_BUCKET.'
            )

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_client'] = None
        state['_tos_module'] = None
        state['logger'] = None
        return state

    def _env_or_cfg(self, cfg_key: str, env_key: str, default: Any = None) -> Any:
        return _cfg_get(self.tos_cfg, cfg_key, None) or os.getenv(env_key, default)

    def _int_env_or_cfg(self, cfg_key: str, env_key: str, default: int) -> int:
        return int(self._env_or_cfg(cfg_key, env_key, default))

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            import tos
        except ImportError as exc:
            raise RuntimeError(
                'TOS storage is selected but the optional "tos" Python package is not installed.'
            ) from exc

        self._tos_module = tos
        ak = self._env_or_cfg('AK', 'TOS_AK') or os.getenv('TOS_ACCESS_KEY')
        sk = self._env_or_cfg('SK', 'TOS_SK') or os.getenv('TOS_SECRET_KEY')
        if not ak or not sk:
            raise RuntimeError('TOS storage requires TOS_AK and TOS_SK environment variables.')

        endpoint = self._env_or_cfg('ENDPOINT', 'TOS_ENDPOINT', 'tos-cn-shanghai.ivolces.com')
        region = self._env_or_cfg('REGION', 'TOS_REGION', 'cn-shanghai')
        options = {
            'request_timeout': self._int_env_or_cfg('REQUEST_TIMEOUT', 'TOS_REQUEST_TIMEOUT', 600),
            'max_connections': self._int_env_or_cfg('MAX_CONNECTIONS', 'TOS_MAX_CONNECTIONS', 1024),
            'connection_time': self._int_env_or_cfg('CONNECTION_TIME', 'TOS_CONNECTION_TIME', 60),
            'socket_timeout': self._int_env_or_cfg('SOCKET_TIMEOUT', 'TOS_SOCKET_TIMEOUT', 600),
            'max_retry_count': self._int_env_or_cfg('MAX_RETRY_COUNT', 'TOS_MAX_RETRY_COUNT', 5),
        }

        try:
            self._client = tos.TosClientV2(ak, sk, endpoint, region, **options)
        except TypeError:
            self._client = tos.TosClientV2(ak, sk, endpoint, region)
        return self._client

    def _object_ref(self, path: Any) -> Tuple[str, str]:
        uri_ref = parse_object_uri(path)
        if uri_ref is not None:
            return uri_ref
        if self.remap_hr4d_original_paths:
            hr4d_key = self._map_hr4d_original_data_key(path)
            if hr4d_key is not None:
                return self.bucket, hr4d_key
        return self.bucket, _join_key(self.root_path, path)

    def _map_to_object_key(self, path: Any) -> str:
        if self.remap_hr4d_original_paths:
            hr4d_key = self._map_hr4d_original_data_key(path)
            if hr4d_key is not None:
                return hr4d_key
        return _join_key(self.root_path, path)

    def _map_hr4d_original_data_key(self, path: Any) -> Optional[str]:
        """Map HR-4D source paths embedded in pkl files to TOS object keys.

        Example:
            /mnt/nas_02/obs02/his3userrw/parse-process-data-2/202601/F772Y8/
            20260120103500/parsed/radar_front_bottom/a.pcd

        maps to:
            datasets/4d_data/original_data/202601/F772Y8/20260120103500/
            parsed/radar_front_bottom/a.pcd
        """

        normalized = str(path).replace('\\', '/')
        if parse_object_uri(normalized) is not None:
            return None

        parts = [part for part in normalized.split('/') if part]
        try:
            marker_index = parts.index('parse-process-data-2')
        except ValueError:
            return None

        tail = parts[marker_index + 1:]
        if len(tail) < 4:
            return None

        if not re.fullmatch(r'20\d{4}', tail[0]):
            return None

        return _join_key(self.root_path, *tail)

    def describe(self, path: Any) -> str:
        bucket, key = self._object_ref(path)
        return 'tos://%s/%s' % (bucket, key)

    @staticmethod
    def _is_not_found(error: Exception) -> bool:
        status = getattr(error, 'status_code', None) or getattr(error, 'status', None)
        code = str(getattr(error, 'code', '') or '')
        message = str(getattr(error, 'message', '') or error)
        return (
            status == 404
            or code in {'NoSuchKey', 'NoSuchBucket', 'NotFound'}
            or 'NoSuchKey' in message
            or 'not found' in message.lower()
        )

    def exists(self, path: Any) -> bool:
        bucket, key = self._object_ref(path)
        client = self._get_client()
        try:
            if hasattr(client, 'head_object'):
                client.head_object(bucket, key)
            else:
                client.get_object(bucket, key)
            return True
        except Exception as exc:
            if self._is_not_found(exc):
                return False
            raise

    def read_bytes(self, path: Any) -> bytes:
        bucket, key = self._object_ref(path)
        response = self._get_client().get_object(bucket, key)
        payload = response.read()
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, str):
            return payload.encode('utf-8')
        return bytes(payload)

    def load_pickle(self, path: Any) -> Any:
        return pickle.loads(self.read_bytes(path))

    def fromfile(self, path: Any, dtype: Any):
        import numpy as np

        return np.frombuffer(self.read_bytes(path), dtype=dtype)

    def np_load(self, path: Any):
        import numpy as np

        return np.load(io.BytesIO(self.read_bytes(path)))


def build_storage(
    dataset_cfg: Any,
    root_path: Optional[Any] = None,
    project_root: Optional[Any] = None,
    logger: Optional[Any] = None,
):
    root_path = root_path if root_path is not None else _cfg_get(dataset_cfg, 'DATA_PATH', '.')
    storage_cfg = _cfg_get(dataset_cfg, 'STORAGE', {})
    storage_type = storage_type_from_config(dataset_cfg)

    if storage_type == 'file':
        return FileStorage(root_path, project_root=project_root)
    if storage_type == 'tos':
        return TOSStorage(root_path, storage_cfg=storage_cfg, logger=logger)
    raise ValueError('Unsupported storage type: %s' % storage_type)
