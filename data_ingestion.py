import os
import requests
from pathlib import Path
from utils import check_gpu_temp

class DataIngestor:
    def __init__(self, target_dir="data"):
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(exist_ok=True)
        check_gpu_temp()

    def download_arxiv(self, url="https://arxiv.org/dataset", size_limit_gb=50):
        # Download arXiv subset (~50GB)
        arxiv_path = self.target_dir / "arxiv_subset.tar.gz"
        print(f"Downloading arXiv subset (~{size_limit_gb}GB) to {arxiv_path}...")
        # Placeholder: Simulate download
        with open(arxiv_path, "wb") as f:
            # Mock download for now; replace with actual URL and streaming
            f.write(b"Mock arXiv data (50GB placeholder)")
        print("Download complete. Extracting... (simulated)")
        (self.target_dir / "arxiv").mkdir(exist_ok=True)

    def download_the_stack(self, url="https://huggingface.co/datasets/bigcode/the-stack-dedup", size_limit_gb=50):
        # Download The Stack deduplicated code (~50GB)
        stack_path = self.target_dir / "the_stack_dedup.tar.gz"
        print(f"Downloading The Stack (~{size_limit_gb}GB) to {stack_path}...")
        # Placeholder: Simulate download
        with open(stack_path, "wb") as f:
            f.write(b"Mock The Stack data (50GB placeholder)")
        print("Download complete. Extracting... (simulated)")
        (self.target_dir / "the_stack").mkdir(exist_ok=True)

    def download_bookcorpus(self, url="https://huggingface.co/datasets/bookcorpus", size_limit_gb=30):
        # Download BookCorpus/OpenWebText (~30GB)
        book_path = self.target_dir / "bookcorpus.tar.gz"
        print(f"Downloading BookCorpus (~{size_limit_gb}GB) to {book_path}...")
        # Placeholder: Simulate download
        with open(book_path, "wb") as f:
            f.write(b"Mock BookCorpus data (30GB placeholder)")
        print("Download complete. Extracting... (simulated)")
        (self.target_dir / "bookcorpus").mkdir(exist_ok=True)

    def download_commoncrawl(self, url="https://commoncrawl.org", size_limit_gb=50):
        # Download CommonCrawl filtered snippets (~50GB)
        ccrawl_path = self.target_dir / "commoncrawl.tar.gz"
        print(f"Downloading CommonCrawl (~{size_limit_gb}GB) to {ccrawl_path}...")
        # Placeholder: Simulate download
        with open(ccrawl_path, "wb") as f:
            f.write(b"Mock CommonCrawl data (50GB placeholder)")
        print("Download complete. Extracting... (simulated)")
        (self.target_dir / "commoncrawl").mkdir(exist_ok=True)

    def chunk_data(self, max_size_gb=15):
        # Chunk data to fit <15GB manifold target
        total_size = sum(f.stat().st_size for f in self.target_dir.glob("**/*") if f.is_file()) / (1024 ** 3)
        print(f"Total data size: {total_size:.2f}GB")
        if total_size > max_size_gb:
            print(f"Chunking data to fit {max_size_gb}GB...")
            # Placeholder: Simulate chunking (select subset)
            for subdir in ["arxiv", "the_stack", "bookcorpus", "commoncrawl"]:
                files = list((self.target_dir / subdir).glob("*"))
                keep_count = int((max_size_gb / total_size) * len(files))
                for f in files[keep_count:]:
                    f.unlink()
            print("Chunking complete.")

if __name__ == "__main__":
    ingestor = DataIngestor()
    ingestor.download_arxiv()
    ingestor.download_the_stack()
    ingestor.download_bookcorpus()
    ingestor.download_commoncrawl()
    ingestor.chunk_data()