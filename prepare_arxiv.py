#!/usr/bin/env python3
"""
Prepare ArXiv submission package
Creates a zip file with all necessary files for ArXiv upload
"""

import zipfile
import os
import shutil

def prepare_arxiv_submission():
    """Create ArXiv submission package"""
    
    # Files to include
    files_to_include = [
        'main.tex',
        'loss_curve.png',
        'loss_tsne.png'
    ]
    
    # Optional supplementary figures
    optional_files = [
        'final_results_proof.png',
        'loss_curve_detailed.png'
    ]
    
    # Check which files exist
    print("Preparing ArXiv submission package...")
    print("-" * 50)
    
    existing_files = []
    missing_files = []
    
    for file in files_to_include:
        if os.path.exists(file):
            existing_files.append(file)
            print(f"[OK] Found: {file}")
        else:
            missing_files.append(file)
            print(f"[X] Missing: {file}")
    
    # Check optional files
    print("\nOptional supplementary files:")
    for file in optional_files:
        if os.path.exists(file):
            existing_files.append(file)
            print(f"[OK] Including: {file}")
        else:
            print(f"- Not including: {file}")
    
    if missing_files:
        print(f"\nWARNING: Missing required files: {missing_files}")
        print("The submission may not compile without these files.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Create zip file
    zip_filename = 'arxiv_submission.zip'
    
    print(f"\nCreating {zip_filename}...")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in existing_files:
            zipf.write(file, os.path.basename(file))
            print(f"  Added: {file}")
    
    # Get file size
    size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    
    print("-" * 50)
    print(f"[SUCCESS] ArXiv submission package created: {zip_filename}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Files included: {len(existing_files)}")
    
    # Create submission checklist
    print("\n" + "=" * 50)
    print("ARXIV SUBMISSION CHECKLIST")
    print("=" * 50)
    print("1. Go to https://arxiv.org/submit")
    print("2. Create new submission")
    print("3. Upload: arxiv_submission.zip")
    print("4. Select primary category: cs.LG (Machine Learning)")
    print("5. Cross-list to: cs.AI, cs.CL")
    print("6. Add abstract from main.tex")
    print("7. Add GitHub link in comments: https://github.com/JustinArndtAI/hkm-poc")
    print("8. Review and submit!")
    
    # Also create a tar.gz version (some prefer this)
    import tarfile
    tar_filename = 'arxiv_submission.tar.gz'
    print(f"\nAlso creating {tar_filename} (alternative format)...")
    with tarfile.open(tar_filename, 'w:gz') as tar:
        for file in existing_files:
            tar.add(file, arcname=os.path.basename(file))
    
    print(f"[SUCCESS] Alternative package created: {tar_filename}")
    
    return zip_filename

if __name__ == "__main__":
    prepare_arxiv_submission()