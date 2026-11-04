"""Download Quick Draw numpy bitmaps for the categories in config.py."""
import urllib.request

from config import CATEGORIES, DATA_DIR

URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for name in CATEGORIES:
        dest = DATA_DIR / f"{name}.npy"
        if dest.exists():
            print(f"skip {name} (already downloaded)")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(URL.format(name), dest)
    print("All categories downloaded to", DATA_DIR)


if __name__ == "__main__":
    main()
