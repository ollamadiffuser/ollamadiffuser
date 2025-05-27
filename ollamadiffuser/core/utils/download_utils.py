#!/usr/bin/env python3
"""
Download utilities for robust model downloading with detailed progress tracking
"""

import os
import time
import logging
from typing import Optional, Callable, Any, Dict
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download, HfApi
from tqdm import tqdm
import threading

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Track download progress across multiple files"""
    
    def __init__(self, total_files: int = 0, progress_callback: Optional[Callable] = None):
        self.total_files = total_files
        self.completed_files = 0
        self.current_file = ""
        self.file_progress = {}
        self.progress_callback = progress_callback
        self.lock = threading.Lock()
        
    def update_file_progress(self, filename: str, downloaded: int, total: int):
        """Update progress for a specific file"""
        with self.lock:
            self.file_progress[filename] = (downloaded, total)
            self._report_progress()
    
    def complete_file(self, filename: str):
        """Mark a file as completed"""
        with self.lock:
            self.completed_files += 1
            if filename in self.file_progress:
                downloaded, total = self.file_progress[filename]
                self.file_progress[filename] = (total, total)
            self._report_progress()
    
    def set_current_file(self, filename: str):
        """Set the currently downloading file"""
        with self.lock:
            self.current_file = filename
            self._report_progress()
    
    def _report_progress(self):
        """Report current progress"""
        if self.progress_callback:
            # Calculate overall progress
            total_downloaded = 0
            total_size = 0
            
            for downloaded, size in self.file_progress.values():
                total_downloaded += downloaded
                total_size += size
            
            progress_msg = f"Files: {self.completed_files}/{self.total_files}"
            if total_size > 0:
                percent = (total_downloaded / total_size) * 100
                progress_msg += f" | Overall: {percent:.1f}%"
            
            if self.current_file:
                progress_msg += f" | Current: {self.current_file}"
            
            self.progress_callback(progress_msg)

def configure_hf_environment():
    """Configure HuggingFace Hub environment for better downloads"""
    # Set reasonable timeouts
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '600')  # 10 minutes
    os.environ.setdefault('HF_HUB_CONNECTION_TIMEOUT', '120')  # 2 minutes
    
    # Disable symlinks for better compatibility
    os.environ.setdefault('HF_HUB_LOCAL_DIR_USE_SYMLINKS', 'False')
    
    # Enable resume downloads
    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', 'False')  # Disable for better compatibility

def get_repo_file_list(repo_id: str) -> Dict[str, int]:
    """Get list of files in repository with their sizes"""
    try:
        api = HfApi()
        repo_info = api.repo_info(repo_id=repo_id)
        
        file_sizes = {}
        for sibling in repo_info.siblings:
            # Include all files, use 0 as default size if not available
            size = sibling.size if sibling.size is not None else 0
            file_sizes[sibling.rfilename] = size
        
        return file_sizes
    except Exception as e:
        logger.warning(f"Could not get file list for {repo_id}: {e}")
        return {}

def format_size(size_bytes: int) -> str:
    """Format size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def robust_snapshot_download(
    repo_id: str,
    local_dir: str,
    cache_dir: Optional[str] = None,
    max_retries: int = 3,
    initial_workers: int = 2,
    force_download: bool = False,
    progress_callback: Optional[Callable] = None
) -> str:
    """
    Download repository snapshot with robust error handling and detailed progress tracking
    
    Args:
        repo_id: Repository ID on HuggingFace Hub
        local_dir: Local directory to download to
        cache_dir: Cache directory
        max_retries: Maximum number of retry attempts
        initial_workers: Initial number of workers (reduced on retries)
        force_download: Force re-download
        progress_callback: Optional progress callback function
    
    Returns:
        Path to downloaded repository
    """
    configure_hf_environment()
    
    # Get file list and sizes for progress tracking
    if progress_callback:
        progress_callback("📋 Getting repository information...")
    
    file_sizes = get_repo_file_list(repo_id)
    total_size = sum(file_sizes.values())
    
    if progress_callback and file_sizes:
        progress_callback(f"📦 Repository: {len(file_sizes)} files, {format_size(total_size)} total")
    
    # Check what's already downloaded
    local_path = Path(local_dir)
    if local_path.exists() and not force_download:
        existing_files = []
        existing_size = 0
        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                rel_path = file_path.relative_to(local_path)
                existing_files.append(str(rel_path))
                existing_size += file_path.stat().st_size
        
        if progress_callback and existing_files:
            progress_callback(f"📁 Found {len(existing_files)} existing files ({format_size(existing_size)})")
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            # Reduce workers on retry attempts to avoid overwhelming connections
            workers = 1 if attempt > 0 else initial_workers
            
            if progress_callback:
                progress_callback(f"🔄 Download attempt {attempt + 1}/{max_retries} (workers: {workers})")
            
            logger.info(f"Download attempt {attempt + 1}/{max_retries} with {workers} workers")
            
            # Create a custom progress callback for tqdm
            def tqdm_callback(t):
                def inner(chunk_size):
                    t.update(chunk_size)
                return inner
            
            result = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                cache_dir=cache_dir,
                max_workers=workers,
                resume_download=True,  # Enable resume
                etag_timeout=300 + (attempt * 60),  # Increase timeout on retries
                force_download=force_download,
                tqdm_class=tqdm if progress_callback else None
            )
            
            if progress_callback:
                progress_callback(f"✅ Successfully downloaded {repo_id}")
            
            logger.info(f"Successfully downloaded {repo_id}")
            return result
            
        except Exception as e:
            last_exception = e
            error_msg = str(e)
            
            # Log the specific error
            logger.warning(f"Download attempt {attempt + 1} failed: {error_msg}")
            
            if attempt < max_retries - 1:
                # Determine wait time based on error type
                if "timeout" in error_msg.lower():
                    wait_time = 30 + (attempt * 15)  # Longer wait for timeouts
                elif "connection" in error_msg.lower():
                    wait_time = 20 + (attempt * 10)  # Medium wait for connection errors
                else:
                    wait_time = 10 + (attempt * 5)   # Shorter wait for other errors
                
                logger.info(f"Waiting {wait_time} seconds before retry...")
                
                if progress_callback:
                    progress_callback(f"⚠️ Download failed, retrying in {wait_time}s... (Error: {error_msg[:100]})")
                
                time.sleep(wait_time)
            else:
                logger.error(f"All download attempts failed. Final error: {error_msg}")
                if progress_callback:
                    progress_callback(f"❌ All download attempts failed: {error_msg}")
    
    # If we get here, all retries failed
    raise last_exception

def robust_file_download(
    repo_id: str,
    filename: str,
    local_dir: str,
    cache_dir: Optional[str] = None,
    max_retries: int = 3,
    progress_callback: Optional[Callable] = None
) -> str:
    """
    Download single file with robust error handling and progress tracking
    
    Args:
        repo_id: Repository ID on HuggingFace Hub
        filename: File to download
        local_dir: Local directory to download to
        cache_dir: Cache directory
        max_retries: Maximum number of retry attempts
        progress_callback: Optional progress callback function
    
    Returns:
        Path to downloaded file
    """
    configure_hf_environment()
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            if progress_callback:
                progress_callback(f"📥 Downloading {filename} (attempt {attempt + 1}/{max_retries})")
            
            logger.info(f"File download attempt {attempt + 1}/{max_retries}: {filename}")
            
            result = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                cache_dir=cache_dir,
                resume_download=True,  # Enable resume
                etag_timeout=180 + (attempt * 30)
            )
            
            if progress_callback:
                progress_callback(f"✅ Downloaded {filename}")
            
            logger.info(f"Successfully downloaded {filename}")
            return result
            
        except Exception as e:
            last_exception = e
            error_msg = str(e)
            
            logger.warning(f"File download attempt {attempt + 1} failed: {error_msg}")
            
            if attempt < max_retries - 1:
                wait_time = 5 + (attempt * 3)  # Progressive backoff
                
                if progress_callback:
                    progress_callback(f"⚠️ Retrying {filename} in {wait_time}s...")
                
                time.sleep(wait_time)
            else:
                logger.error(f"All file download attempts failed. Final error: {error_msg}")
                if progress_callback:
                    progress_callback(f"❌ Failed to download {filename}: {error_msg}")
    
    # If we get here, all retries failed
    raise last_exception

def check_download_integrity(local_dir: str, repo_id: str) -> bool:
    """Check if downloaded files are complete and valid"""
    try:
        local_path = Path(local_dir)
        if not local_path.exists():
            return False
        
        # Check for essential files
        essential_files = ['model_index.json']
        for essential_file in essential_files:
            if not (local_path / essential_file).exists():
                logger.warning(f"Missing essential file: {essential_file}")
                return False
        
        # Files to ignore during integrity check
        ignore_patterns = [
            '.lock',           # HuggingFace lock files
            '.metadata',       # HuggingFace metadata files
            '.incomplete',     # Incomplete download files
            '.cache',          # Cache directory
            '.git',            # Git files
            '.gitattributes',  # Git attributes
            'README.md',       # Documentation files
            'LICENSE.md',      # License files
            'dev_grid.jpg'     # Sample images
        ]
        
        # Check for empty files (excluding ignored patterns)
        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                # Skip files that match ignore patterns
                should_ignore = any(pattern in str(file_path) for pattern in ignore_patterns)
                if should_ignore:
                    continue
                
                # Check if file is empty
                if file_path.stat().st_size == 0:
                    logger.warning(f"Empty file detected: {file_path}")
                    return False
        
        # Check for critical model files
        critical_dirs = ['transformer', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2']
        for critical_dir in critical_dirs:
            dir_path = local_path / critical_dir
            if dir_path.exists():
                # Check if directory has any non-empty files
                has_content = False
                for file_path in dir_path.rglob('*'):
                    if file_path.is_file() and file_path.stat().st_size > 0:
                        # Skip ignored files
                        should_ignore = any(pattern in str(file_path) for pattern in ignore_patterns)
                        if not should_ignore:
                            has_content = True
                            break
                
                if not has_content:
                    logger.warning(f"Critical directory {critical_dir} appears to be empty or incomplete")
                    return False
        
        logger.info("Download integrity check passed")
        return True
        
    except Exception as e:
        logger.error(f"Error checking download integrity: {e}")
        return False 