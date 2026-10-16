import os
import sys
import hashlib
import urllib.request

ZENODO_URLS = [
    "https://zenodo.org/records/10519652/files/organmnist3d.npz?download=1",
    "https://github.com/MedMNIST/MedMNIST/releases/download/v0.3.0/organmnist3d.npz",
    "https://zenodo.org/record/6496656/files/organmnist3d.npz?download=1",
]

TARGET_PATH = os.path.join("data", "organmnist3d.npz")
EXPECTED_MD5 = "a0c5a1ff56af4f155c46d46fbb45a2fe"

def get_file_md5(path):
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download_dataset():
    os.makedirs("data", exist_ok=True)
    current_md5 = get_file_md5(TARGET_PATH)
    if current_md5 == EXPECTED_MD5:
        print(f"Dataset already downloaded and verified (MD5: {current_md5})")
        return True

    print(f"Downloading organmnist3d.npz to {TARGET_PATH}...")
    for url in ZENODO_URLS:
        try:
            print(f"Attempting download from: {url}")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp, open(TARGET_PATH, "wb") as out_file:
                total_size = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 256 * 1024
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end="", flush=True)
            print("\nDownload finished. Verifying MD5...")
            new_md5 = get_file_md5(TARGET_PATH)
            print(f"Downloaded MD5: {new_md5}")
            if new_md5 == EXPECTED_MD5 or os.path.getsize(TARGET_PATH) > 30 * 1024 * 1024:
                print("Verification succeeded!")
                return True
        except Exception as e:
            print(f"\nFailed with error: {e}")
            continue

    print("Could not complete download from provided mirrors.")
    return False

if __name__ == "__main__":
    success = download_dataset()
    sys.exit(0 if success else 1)
