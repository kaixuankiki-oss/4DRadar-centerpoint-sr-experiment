import pickle
import tempfile
import unittest
from pathlib import Path

from pcdet.utils.storage import FileStorage, TOSStorage, parse_object_uri


class _ObjectResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return self.payload


class _NotFoundError(Exception):
    status_code = 404
    code = 'NoSuchKey'


class _FakeTOSClient:
    def __init__(self, objects):
        self.objects = objects
        self.calls = []

    def head_object(self, bucket, key):
        self.calls.append(('head', bucket, key))
        if (bucket, key) not in self.objects:
            raise _NotFoundError(key)

    def get_object(self, bucket, key):
        self.calls.append(('get', bucket, key))
        if (bucket, key) not in self.objects:
            raise _NotFoundError(key)
        return _ObjectResponse(self.objects[(bucket, key)])


class TestStorage(unittest.TestCase):
    def test_parse_object_uri_accepts_tos_and_s3(self):
        self.assertEqual(parse_object_uri('tos://bucket/a/b.pkl'), ('bucket', 'a/b.pkl'))
        self.assertEqual(parse_object_uri('s3://bucket/a/b.pkl'), ('bucket', 'a/b.pkl'))
        self.assertIsNone(parse_object_uri('data/1000_original_data'))

    def test_file_storage_resolves_relative_paths_from_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'splits').mkdir()
            (root / 'splits' / 'info.pkl').write_bytes(pickle.dumps({'ok': True}))

            storage = FileStorage(root)

            self.assertTrue(storage.exists('splits/info.pkl'))
            self.assertEqual(storage.load_pickle('splits/info.pkl'), {'ok': True})

    def test_tos_storage_joins_prefix_data_path_and_relative_key(self):
        payload = pickle.dumps({'samples': 3})
        client = _FakeTOSClient({
            ('hr4d-bucket', 'datasets/1000_original_data/splits/info.pkl'): payload,
        })
        storage = TOSStorage(
            root_path='1000_original_data',
            storage_cfg={
                'TOS': {
                    'BUCKET': 'hr4d-bucket',
                    'PREFIX': 'datasets',
                },
            },
            client=client,
        )

        self.assertTrue(storage.exists('splits/info.pkl'))
        self.assertEqual(storage.load_pickle('splits/info.pkl'), {'samples': 3})

    def test_tos_storage_uri_overrides_config_bucket(self):
        client = _FakeTOSClient({
            ('uri-bucket', 'root/splits/info.pkl'): b'abc',
        })
        storage = TOSStorage(
            root_path='tos://uri-bucket/root',
            storage_cfg={'TOS': {'BUCKET': 'config-bucket'}},
            client=client,
        )

        self.assertEqual(storage.read_bytes('splits/info.pkl'), b'abc')
        self.assertIn(('get', 'uri-bucket', 'root/splits/info.pkl'), client.calls)

    def test_tos_storage_maps_hr4d_relative_source_path(self):
        source_path = (
            'obs02/his3userrw/parse-process-data-2/202601/F772Y8/20260120103500/'
            'parsed/radar_front_bottom/sample.pcd'
        )
        mapped_key = (
            'datasets/4d_data/original_data/202601/F772Y8/20260120103500/'
            'parsed/radar_front_bottom/sample.pcd'
        )
        client = _FakeTOSClient({
            ('perception-result', mapped_key): b'pcd',
        })
        storage = TOSStorage(
            root_path='datasets/4d_data/original_data',
            storage_cfg={
                'TOS': {
                    'BUCKET': 'perception-result',
                },
            },
            client=client,
        )

        self.assertTrue(storage.exists(source_path))
        self.assertEqual(storage.read_bytes(source_path), b'pcd')
        self.assertIn(('get', 'perception-result', mapped_key), client.calls)

    def test_tos_storage_maps_hr4d_absolute_nas_source_path(self):
        source_path = (
            '/mnt/nas_02/obs02/his3usercw/parse-process-data-2/202601/F772Y8/'
            '20260131170000/parsed/lidar_front/sample.pcd'
        )
        mapped_key = (
            'datasets/4d_data/original_data/202601/F772Y8/20260131170000/'
            'parsed/lidar_front/sample.pcd'
        )
        client = _FakeTOSClient({
            ('perception-result', mapped_key): b'lidar',
        })
        storage = TOSStorage(
            root_path='datasets/4d_data/original_data',
            storage_cfg={
                'TOS': {
                    'BUCKET': 'perception-result',
                },
            },
            client=client,
        )

        self.assertEqual(storage.read_bytes(source_path), b'lidar')
        self.assertIn(('get', 'perception-result', mapped_key), client.calls)

if __name__ == '__main__':
    unittest.main()
