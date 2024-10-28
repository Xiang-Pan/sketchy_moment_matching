from minio import Minio
from minio.error import S3Error
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import os
import boto3
from botocore.client import Config

# # upload a file from local file system '/home/john/piano.mp3' to bucket 'songs' with 'piano.mp3' as the object name.
# s3.Bucket('songs').upload_file('/home/john/piano.mp3','piano.mp3')

# # download the object 'piano.mp3' from the bucket 'songs' and save it to local FS as /tmp/classical.mp3
# s3.Bucket('songs').download_file('piano.mp3', '/tmp/classical.mp3')

# print "Downloaded 'piano.mp3' as  'classical.mp3'. "

def get_s3_client():
    return boto3.client('s3',
                    endpoint_url='http://minioapi.xiangpan.site',
                    aws_access_key_id='HmBR4sWSV5ukyVIZfmwA',
                    aws_secret_access_key='gRI4CqFF6XCDdjXpyxJcwUMFx3bnsC3LKIeE8sl9',
                    config=Config(signature_version='s3v4'),
                    region_name='us-east-1')

def get_minio_client():
    return Minio(
        "minioapi.xiangpan.site",
        access_key="HmBR4sWSV5ukyVIZfmwA",
        secret_key="gRI4CqFF6XCDdjXpyxJcwUMFx3bnsC3LKIeE8sl9",
        secure=False
    )

def download_directory(bucket_name, remote_prefix, local_path, minio_client=None):
    """
    Downloads an entire directory structure from MinIO to a local path with a progress bar.
    """
    if minio_client is None:
        minio_client = get_minio_client()
    objects = list(minio_client.list_objects(bucket_name, prefix=remote_prefix, recursive=True))
    total = len(objects)
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    
    with tqdm(total=total, desc="Downloading files") as pbar:
        for obj in objects:
            local_file_path = os.path.join(local_path, obj.object_name[len(remote_prefix)+1:])
            local_dir = os.path.dirname(local_file_path)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            try:
                minio_client.fget_object(bucket_name, obj.object_name, local_file_path)
            except S3Error as e:
                print(f"Failed to download {obj.object_name}: {e}")
            pbar.update(1)


def exists_on_minio(minio_client, bucket_name, object_name):
    """
    Check if an object or directory exists on MinIO.
    """
    return False
    try:
        minio_client.stat_object(bucket_name, object_name)
        return True
    except S3Error:
        return False
    
def upload_file(local_file_path, bucket_path, pbar, bucket_name, minio_client):
    try:
        stat = minio_client.stat_object(bucket_name, bucket_path)
        remote_md5 = stat.etag.replace('"', '')
        with open(local_file_path, "rb") as f:
            local_md5 = hashlib.md5(f.read()).hexdigest()
        if remote_md5 == local_md5:
            logger.info(f"File {bucket_path} already exists and has the same hash. Skipping...")
        else:
            logger.info(f"File {bucket_path} already exists but has a different hash. Uploading...")
            minio_client.fput_object(bucket_name, bucket_path, local_file_path)
    except S3Error:
        logger.info(f"File {bucket_path} does not exist. Uploading...")
        minio_client.fput_object(bucket_name, bucket_path, local_file_path)
    pbar.update(1)



def upload_directory(local_path, bucket_name, base_prefix, minio_client=None):
    """
    Uploads a directory and its subdirectories to a MinIO bucket with a progress bar.
    """
    if minio_client is None:
        minio_client = get_minio_client()
    # list buckets
    buckets = minio_client.list_buckets()
    logger.debug(f"Found buckets: {buckets}")

    files = [os.path.join(root, filename) for root, _, files in os.walk(local_path) for filename in files]
    total_files = len(files)
    for local_file_path in files:
        relative_path = os.path.relpath(local_file_path, start=local_path)
        bucket_path = os.path.join(base_prefix, relative_path)
        upload_file(local_file_path, bucket_path, tqdm(total=total_files, desc="Uploading files"), bucket_name, minio_client)
    # with ThreadPoolExecutor() as executor, tqdm(total=total_files, desc="Uploading files") as pbar:
    #     for local_file_path in files:
    #         relative_path = os.path.relpath(local_file_path, start=local_path)
    #         bucket_path = os.path.join(base_prefix, relative_path)
    #         executor.submit(upload_file, local_file_path, bucket_path, pbar, bucket_name, minio_client)

if __name__ == "__main__":
    s3 = boto3.resource('s3',
                    endpoint_url='http://minioapi.xiangpan.site',
                    aws_access_key_id='HmBR4sWSV5ukyVIZfmwA',
                    aws_secret_access_key='gRI4CqFF6XCDdjXpyxJcwUMFx3bnsC3LKIeE8sl9',
                    config=Config(signature_version='s3v4'),
                    region_name='us-east-1')
    s3.Bucket('labs').upload_file('viz.py', 'data_pruning/viz.py')

    # minio_client = get_minio_client()
    # upload_directory("outputs", "labs", "data_pruning/outputs", minio_client)
    # upload_directory("cached_grads", "labs", "data_pruning/cached_grads", minio_client)
    # download_directory("labs", "data_pruning/outputs", "outputs", minio_client)
