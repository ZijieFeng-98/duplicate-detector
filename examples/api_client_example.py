"""
Example: Using the REST API

Demonstrates how to use the REST API for duplicate detection.
"""

import requests
import time
from pathlib import Path


def analyze_pdf_via_api(pdf_path: Path, api_url: str = "http://localhost:8000", preset: str = "balanced"):
    """
    Analyze PDF using REST API.
    
    Args:
        pdf_path: Path to PDF file
        api_url: Base URL of API server
        preset: Configuration preset
    
    Returns:
        Analysis results dictionary
    """
    # Step 1: Upload PDF and start analysis
    print(f"Uploading {pdf_path.name}...")
    with open(pdf_path, "rb") as f:
        response = requests.post(
            f"{api_url}/analyze",
            files={"file": f},
            data={"preset": preset}
        )
    
    if response.status_code != 200:
        raise Exception(f"Failed to start analysis: {response.text}")
    
    job_data = response.json()
    job_id = job_data["job_id"]
    print(f"Analysis started. Job ID: {job_id}")
    
    # Step 2: Poll for completion
    print("Waiting for analysis to complete...")
    while True:
        status_response = requests.get(f"{api_url}/status/{job_id}")
        status = status_response.json()
        
        print(f"Status: {status['status']}, Progress: {status.get('progress', 0):.1%}")
        
        if status["status"] == "completed":
            break
        elif status["status"] == "failed":
            raise Exception(f"Analysis failed: {status.get('error', 'Unknown error')}")
        
        time.sleep(2)  # Poll every 2 seconds
    
    # Step 3: Get results
    print("Retrieving results...")
    results_response = requests.get(f"{api_url}/results/{job_id}/json")
    results = results_response.json()
    
    print(f"\nResults:")
    print(f"  Total pairs: {results['total_pairs']}")
    print(f"  Tier A: {results['tier_a_count']}")
    print(f"  Tier B: {results['tier_b_count']}")
    
    # Step 4: Download TSV file
    tsv_response = requests.get(f"{api_url}/results/{job_id}/download")
    output_path = Path(f"results_{job_id}.tsv")
    with open(output_path, "wb") as f:
        f.write(tsv_response.content)
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python api_example.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    try:
        results = analyze_pdf_via_api(pdf_path)
        print("\n✅ Analysis complete!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

