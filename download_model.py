import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def resolve_direct_url(url, token):
    """
    Manually follow redirects to get the final download URL and correct headers.
    Strips the Authorization header if redirected to a non-Hugging Face host.
    """
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    current_url = url
    max_redirects = 10
    redirects = 0
    
    while redirects < max_redirects:
        # Use stream=True so we don't download the file body during resolution
        r = requests.get(current_url, headers=headers, allow_redirects=False, stream=True)
        if r.status_code in [301, 302, 303, 307, 308]:
            next_url = r.headers["Location"]
            # If domain changes, strip Authorization
            if "huggingface.co" not in next_url:
                headers = {}
            current_url = next_url
            redirects += 1
            r.close()
        else:
            r.close()
            return current_url, headers
            
    raise Exception("Too many redirects")

def download_part(part_num, start_byte, end_byte, url, token, part_file):
    """
    Downloads a single byte range of the file. Supports resuming within the part itself.
    """
    max_retries = 15
    retry_delay = 2
    
    part_size = end_byte - start_byte + 1
    
    for attempt in range(1, max_retries + 1):
        try:
            # Check current size of the part file
            current_size = 0
            if os.path.exists(part_file):
                current_size = os.path.getsize(part_file)
                
            if current_size == part_size:
                # Already fully downloaded
                return True
                
            if current_size > part_size:
                # Corrupted/oversized, delete and restart
                try:
                    os.remove(part_file)
                except:
                    pass
                current_size = 0
                
            # Determine start byte for this range request
            req_start = start_byte + current_size
            
            # Resolve the final URL and headers for this request
            direct_url, headers = resolve_direct_url(url, token)
            headers["Range"] = f"bytes={req_start}-{end_byte}"
            
            resp = requests.get(direct_url, headers=headers, stream=True, timeout=15)
            
            if resp.status_code not in [200, 206]:
                raise Exception(f"Server returned status code {resp.status_code}")
                
            # Append to file if we have existing progress, otherwise write fresh
            mode = "ab" if current_size > 0 else "wb"
            with open(part_file, mode) as f:
                for chunk in resp.iter_content(chunk_size=256*1024):  # 256KB chunks
                    if chunk:
                        f.write(chunk)
                        
            # Verify size
            if os.path.getsize(part_file) == part_size:
                return True
            else:
                raise Exception("Part download ended prematurely")
                
        except Exception as e:
            if attempt == max_retries:
                print(f"\nPart {part_num} failed after {max_retries} attempts: {e}")
                return False
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 20)

def main():
    url = "https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/pytorch_model.bin"
    output_file = "c:\\Users\\akand\\Desktop\\peter (project)\\model_cache\\local_nllb\\pytorch_model.bin"
    token = "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"
    num_threads = 8
    part_size_mb = 10  # Size of each part in MB (smaller parts are faster and less prone to timeout)
    
    # 1. Get remote file size
    print("Determining remote file size...")
    try:
        direct_url, headers = resolve_direct_url(url, token)
        r = requests.head(direct_url, headers=headers, timeout=20)
        total_size = int(r.headers.get('content-length', 0))
        if not total_size:
            r = requests.get(direct_url, headers=headers, stream=True, timeout=20)
            total_size = int(r.headers.get('content-length', 0))
            r.close()
    except Exception as e:
        print(f"Failed to get remote size: {e}")
        total_size = 2460457927
        
    print(f"Total size: {total_size:,} bytes (~{total_size/(1024**3):.2f} GB)")
    
    # Ensure local directory exists
    local_dir = os.path.dirname(output_file)
    os.makedirs(local_dir, exist_ok=True)
    
    # 2. Divide file into parts
    chunk_bytes = part_size_mb * 1024 * 1024
    parts = []
    start = 0
    part_idx = 0
    while start < total_size:
        end = min(start + chunk_bytes - 1, total_size - 1)
        part_file = os.path.join(local_dir, f"pytorch_model.bin.part_{part_idx}")
        parts.append({
            "num": part_idx,
            "start": start,
            "end": end,
            "file": part_file
        })
        start = end + 1
        part_idx += 1
        
    print(f"Divided file into {len(parts)} parts of {part_size_mb} MB each.")
    
    # Count how many parts are already fully downloaded
    completed_parts = 0
    for part in parts:
        expected_size = part["end"] - part["start"] + 1
        if os.path.exists(part["file"]) and os.path.getsize(part["file"]) == expected_size:
            completed_parts += 1
            
    print(f"Already completed parts: {completed_parts}/{len(parts)}")
    
    # 3. Download remaining parts in parallel
    start_time = time.time()
    futures = {}
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for part in parts:
            expected_size = part["end"] - part["start"] + 1
            if os.path.exists(part["file"]) and os.path.getsize(part["file"]) == expected_size:
                continue
            
            future = executor.submit(
                download_part, 
                part["num"], 
                part["start"], 
                part["end"], 
                url, 
                token, 
                part["file"]
            )
            futures[future] = part
            
        completed = completed_parts
        total_parts = len(parts)
        
        for future in as_completed(futures):
            part = futures[future]
            success = future.result()
            if success:
                completed += 1
                elapsed = time.time() - start_time
                percent = (completed / total_parts) * 100
                print(f"Part {part['num']} finished. Overall progress: {completed}/{total_parts} ({percent:.1f}%) | Elapsed: {elapsed:.1f}s")
            else:
                print(f"Error downloading Part {part['num']}!")
                sys.exit(1)
                
    # 4. Merge parts into target file
    print("\nAll parts downloaded successfully. Merging files...")
    merge_start = time.time()
    
    with open(output_file, "wb") as outfile:
        for part in parts:
            part_file = part["file"]
            with open(part_file, "rb") as infile:
                while True:
                    data = infile.read(1024*1024)
                    if not data:
                        break
                    outfile.write(data)
                    
    print(f"Merged successfully in {time.time() - merge_start:.1f}s.")
    
    # 5. Clean up part files
    print("Cleaning up temporary part files...")
    for part in parts:
        try:
            os.remove(part["file"])
        except Exception as e:
            print(f"Failed to remove {part['file']}: {e}")
            
    print("Download script completed successfully!")

if __name__ == "__main__":
    main()
