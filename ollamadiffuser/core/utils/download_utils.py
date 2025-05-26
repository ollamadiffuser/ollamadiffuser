#!/usr/bin/env python3
"""
Download utilities for robust model downloading
"""

import os
import time
import logging
from typing import Optional, Callable, Any
from huggingface_hub import snapshot_download, hf_hub_download

logger = logging.getLogger(__name__)

def configure_hf_environment():
    """Configure HuggingFace Hub environment for better downloads"""
    # Set reasonable timeouts
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '300')  # 5 minutes
    os.environ.setdefault('HF_HUB_CONNECTION_TIMEOUT', '60')  # 1 minute
    
    # Disable symlinks for better compatibility
    os.environ.setdefault('HF_HUB_LOCAL_DIR_USE_SYMLINKS', 'False')
    
    # Configure requests session
    os.environ.setdefault('REQUESTS_CA_BUNDLE', '')  # Use system certificates

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
    Download repository snapshot with robust error handling and retry logic
    
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
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            # Reduce workers on retry attempts to avoid overwhelming connections
            workers = 1 if attempt > 0 else initial_workers
            
            if progress_callback:
                progress_callback(f"Attempt {attempt + 1}/{max_retries}")
            
            logger.info(f"Download attempt {attempt + 1}/{max_retries} with {workers} workers")
            
            result = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                cache_dir=cache_dir,
                max_workers=workers,
                resume_download=True,
                etag_timeout=120 + (attempt * 30),  # Increase timeout on retries
                force_download=force_download
            )
            
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
                    wait_time = 10 + (attempt * 5)  # Longer wait for timeouts
                else:
                    wait_time = 5 + (attempt * 2)   # Shorter wait for other errors
                
                logger.info(f"Waiting {wait_time} seconds before retry...")
                
                if progress_callback:
                    progress_callback(f"Download failed, retrying in {wait_time}s...")
                
                time.sleep(wait_time)
            else:
                logger.error(f"All download attempts failed. Final error: {error_msg}")
    
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
    Download single file with robust error handling
    
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
                progress_callback(f"Downloading {filename} (attempt {attempt + 1}/{max_retries})")
            
            logger.info(f"File download attempt {attempt + 1}/{max_retries}: {filename}")
            
            result = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                cache_dir=cache_dir,
                resume_download=True
            )
            
            logger.info(f"Successfully downloaded {filename}")
            return result
            
        except Exception as e:
            last_exception = e
            error_msg = str(e)
            
            logger.warning(f"File download attempt {attempt + 1} failed: {error_msg}")
            
            if attempt < max_retries - 1:
                wait_time = 3 + attempt  # Shorter wait for individual files
                
                if progress_callback:
                    progress_callback(f"Download failed, retrying {filename} in {wait_time}s...")
                
                time.sleep(wait_time)
            else:
                logger.error(f"All file download attempts failed. Final error: {error_msg}")
    
    # If we get here, all retries failed
    raise last_exception 