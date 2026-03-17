import os
import gdown

class PublicDriveManager:
    """
    Simpler manager that uses public links to download data.
    No API Key or credentials.json required!
    """
    def __init__(self, data1_path='data1', data2_path='data2'):
        self.data1_path = data1_path
        self.data2_path = data2_path
        
        # Ensure local directories exist
        os.makedirs(self.data1_path, exist_ok=True)
        os.makedirs(self.data2_path, exist_ok=True)

    def sync_public_folder(self, folder_url, target_dir):
        """
        Downloads all contents of a public Google Drive folder link.
        """
        print(f"Syncing folder from: {folder_url} to {target_dir}")
        try:
            # gdown will handle the folder download
            # use_cookies=False is safer for public links
            gdown.download_folder(folder_url, output=target_dir, quiet=False, use_cookies=False)
            return True
        except Exception as e:
            print(f"Sync error: {e}")
            return False

if __name__ == "__main__":
    # Example:
    # manager = PublicDriveManager()
    # manager.sync_public_folder('https://drive.google.com/drive/folders/YOUR_ID', 'data1')
    pass
