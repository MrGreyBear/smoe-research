"""Download the full research universe into data/raw and print the manifest."""

from smoe.data.loaders import download_all

if __name__ == "__main__":
    man = download_all()
    print(man.to_string(index=False))
