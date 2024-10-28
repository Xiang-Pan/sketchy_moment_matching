import os
import io
import boto3
os.environ['AWS_ACCESS_KEY_ID'] = 'HmBR4sWSV5ukyVIZfmwA'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'gRI4CqFF6XCDdjXpyxJcwUMFx3bnsC3LKIeE8sl9'
os.environ['AWS_ENDPOINT_URL'] = 'http://minioapi.xiangpan.site'
client = boto3.client('s3')

#TODO
def load_obj(path):
    if os.path.exists(path):
        return torch.load(path)
    else:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            client.download_file('labs', path, path)
            return torch.load(path)
        except:
            return None
    # # return torch.load(path)
    # if os.path.exists(path):
    #     return torch.load(path)
    # else:
    # #     return None
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     os.system(f"rsync -avzrP ai:LABROOTs/data_pruning/{path} {os.path.dirname(path)}")
    #     return torch.load(path)
def test_server():
    print("Testing server connection")
    print(client.list_buckets())

def save_obj(obj, path):
    # save locally
    # torch.save(obj, path)
    if type(obj) == str:
        obj = obj.encode()
    if type(obj) == torch.Tensor:
        buffer = io.BytesIO()
        torch.save(obj, buffer)
    
    client.put_object(Bucket='labs', 
                        Key=path, 
                        Body=buffer)


def upload_file(file_path, bucket_name="labs"):
    # find the path after labs
    full_path = os.path.abspath(file_path)
    print(full_path)
    path_after_labs = full_path.split('labs/')[1]
    target_path = path_after_labs
    # check if the file exists
    try:
        client.head_object(Bucket=bucket_name, Key=target_path)
        print(f"{target_path} already exists")
        return
    except:
        client.upload_file(full_path, bucket_name, target_path)
        
def download_file(file_path, bucket_name="labs"):
    # find the path after labs
    full_path = os.path.abspath(file_path)
    path_after_labs = full_path.split('labs/')[1]
    target_path = path_after_labs
    # check if the file exists
    try:
        client.head_object(Bucket=bucket_name, Key=target_path)
        print(f"{target_path} exists, downloading")
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        client.download_file(bucket_name, target_path, full_path)
    except:
        print(f"{target_path} does not exist")
        return

def upload_folder(folder_path, bucket_name="labs"):
    print("Uploading folder")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # check if is a link    
            if not os.path.islink(file_path) and "back" not in file_path:
                upload_file(file_path, bucket_name=bucket_name)
            else:
                print(f"{file_path} is a link, skipping")


def download_folder(folder_path, bucket_name="labs"):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # check file exist 
            if not os.path.exists(file_path):
                download_file(file_path, bucket_name=bucket_name)

if __name__ == '__main__':
    test_server()
    upload_folder("cached_datasets/")
    # download_folder("cached_datasets/")
