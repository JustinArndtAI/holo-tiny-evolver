import os
import subprocess
import requests
from pathlib import Path
from utils import check_gpu_temp

class DataIngestor:
    def __init__(self, target_dir="data"):
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(exist_ok=True)
        check_gpu_temp()

    def download_arxiv(self, size_limit_gb=50):
        # Download arXiv metadata dataset from Kaggle/HF (~2GB, with links to full sources subset)
        print(f"Downloading arXiv metadata subset (~{size_limit_gb}GB effective) from Hugging Face...")
        repo_id = "Cornell-University/arxiv"
        subprocess.run([
            "huggingface-cli", "download", repo_id, "--repo-type", "dataset",
            "--local-dir", str(self.target_dir / "arxiv"),
            "--include", "arxiv-metadata-oai-snapshot.json"  # Core metadata file
        ], check=True)
        # Download sample source files (subset to ~50GB via first 100 files from manifest)
        print("Downloading sample arXiv sources...")
        manifest_url = "https://arxiv.org/help/bulk_data_s3/arXiv_src_manifest.xml"
        response = requests.get(manifest_url)
        response.raise_for_status()
        # Parse XML for first 100 file URLs (simplified; in production, use xml.etree)
        sample_urls = ["https://arxiv.org/ftp/arxiv/papers/2409/2409.00001.tar" for _ in range(100)]  # Mock sample; replace with parsed
        for i, url in enumerate(sample_urls[:10]):  # Limit to 10 for ~50GB simulation
            local_path = self.target_dir / "arxiv" / f"arxiv_sample_{i}.tar"
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        print("Download complete. (Extraction simulated)")
        (self.target_dir / "arxiv").mkdir(exist_ok=True)

    def download_the_stack(self, size_limit_gb=50):
        # Download The Stack dedup subset from Hugging Face (~50GB)
        print(f"Downloading The Stack dedup subset (~{size_limit_gb}GB) from Hugging Face...")
        repo_id = "bigcode/the-stack-dedup"
        subprocess.run([
            "huggingface-cli", "download", repo_id, "--repo-type", "dataset",
            "--local-dir", str(self.target_dir / "the_stack"),
            "--include", "data/train-00000-of-00001.parquet"  # First shard for subset
        ], check=True)
        print("Download complete. (Extraction simulated)")
        (self.target_dir / "the_stack").mkdir(exist_ok=True)

    def download_bookcorpus(self, size_limit_gb=30):
        # Download BookCorpus from Hugging Face (~5GB full, under limit)
        print(f"Downloading BookCorpus (~{size_limit_gb}GB) from Hugging Face...")
        repo_id = "bookcorpus/bookcorpus"
        subprocess.run([
            "huggingface-cli", "download", repo_id, "--repo-type", "dataset",
            "--local-dir", str(self.target_dir / "bookcorpus")
        ], check=True)
        print("Download complete. (Extraction simulated)")
        (self.target_dir / "bookcorpus").mkdir(exist_ok=True)

    def download_commoncrawl(self, size_limit_gb=50):
        # Download CommonCrawl WARC subset via direct HTTP (~1GB single file as 50GB proxy; expand for full)
        print(f"Downloading CommonCrawl subset (~{size_limit_gb}GB effective) via direct HTTP...")
        url = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2023-10/segments/1695831076008.13/warc/CC-MAIN-20231015153250-20231015183250-00000.warc.gz"
        ccrawl_path = self.target_dir / "commoncrawl.tar.gz"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(ccrawl_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete. (Extraction and filtering simulated)")
        (self.target_dir / "commoncrawl").mkdir(exist_ok=True)

    def chunk_data(self, max_size_gb=15):
        # Chunk data to fit <15GB manifold target
        total_size = sum(f.stat().st_size for f in self.target_dir.glob("**/*") if f.is_file()) / (1024 ** 3)
        print(f"Total data size: {total_size:.2f}GB")
        if total_size > max_size_gb:
            print(f"Chunking data to fit {max_size_gb}GB...")
            # Simple subset selection for each subdir
            for subdir in ["arxiv", "the_stack", "bookcorpus", "commoncrawl"]:
                files = list((self.target_dir / subdir).glob("*"))
                if len(files) > 1:
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