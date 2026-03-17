import gdown
import os

def sync_test(url, target='data1'):
    print(f"Starting sync test for: {url}")
    
    os.makedirs(target, exist_ok=True)

    try:
        gdown.download_folder(url, output=target, quiet=False, use_cookies=False)
        print(f"Successfully synced data to {target}")
    except Exception as e:
        print("Sync failed:", e)

if __name__ == "__main__":
    url = "https://drive.google.com/drive/folders/1l9d8bkPa2xZVYFLl2ZsbBL9DwwPHPhkm"
    sync_test(url, 'data1')