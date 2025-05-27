#!/usr/bin/env python3
"""
Real-time download progress monitor for FLUX.1-dev
"""

import os
import sys
import time
from pathlib import Path
import subprocess
from datetime import datetime, timedelta

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ollamadiffuser.core.config.settings import settings
from ollamadiffuser.core.utils.download_utils import format_size

class DownloadMonitor:
    def __init__(self, model_name="flux.1-dev"):
        self.model_name = model_name
        self.model_path = settings.get_model_path(model_name)
        self.cache_path = self.model_path / ".cache" / "huggingface" / "download"
        self.start_time = datetime.now()
        self.last_size = 0
        self.last_check_time = datetime.now()
        self.size_history = []
        
    def get_incomplete_files(self):
        """Get list of incomplete download files"""
        incomplete_files = []
        if self.cache_path.exists():
            for file_path in self.cache_path.rglob("*.incomplete"):
                incomplete_files.append(file_path)
        return incomplete_files
    
    def get_total_downloaded_size(self):
        """Get total size of downloaded files"""
        total_size = 0
        if self.model_path.exists():
            for file_path in self.model_path.rglob('*'):
                if file_path.is_file() and not file_path.name.endswith('.lock'):
                    total_size += file_path.stat().st_size
        return total_size
    
    def calculate_speed(self, current_size):
        """Calculate download speed"""
        now = datetime.now()
        time_diff = (now - self.last_check_time).total_seconds()
        
        if time_diff > 0 and self.last_size > 0:
            size_diff = current_size - self.last_size
            speed = size_diff / time_diff
            return speed
        return 0
    
    def format_time(self, seconds):
        """Format time duration"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m {seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def estimate_remaining_time(self, current_size, speed, target_size=None):
        """Estimate remaining download time"""
        if speed <= 0:
            return "Unknown"
        
        if target_size is None:
            # Estimate based on FLUX.1-dev typical size (around 23GB)
            target_size = 23 * 1024 * 1024 * 1024  # 23GB
        
        remaining_size = max(0, target_size - current_size)
        remaining_seconds = remaining_size / speed
        return self.format_time(remaining_seconds)
    
    def check_download_processes(self):
        """Check if download processes are running"""
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            processes = []
            
            for line in result.stdout.split('\n'):
                if any(keyword in line.lower() for keyword in ['ollamadiffuser', 'huggingface', 'download']):
                    if 'grep' not in line and line.strip():
                        processes.append(line.strip())
            
            return processes
        except Exception as e:
            return [f"Error checking processes: {e}"]
    
    def display_progress(self):
        """Display current progress"""
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🚀 FLUX.1-dev Download Progress Monitor")
        print("=" * 60)
        print(f"📦 Model: {self.model_name}")
        print(f"📁 Path: {self.model_path}")
        print(f"⏰ Started: {self.start_time.strftime('%H:%M:%S')}")
        
        elapsed = datetime.now() - self.start_time
        print(f"⌛ Elapsed: {self.format_time(elapsed.total_seconds())}")
        print()
        
        # Check incomplete files
        incomplete_files = self.get_incomplete_files()
        if incomplete_files:
            print("📥 Active Downloads:")
            for file_path in incomplete_files:
                file_size = file_path.stat().st_size
                file_name = file_path.name.replace('.incomplete', '')
                print(f"   📄 {file_name}: {format_size(file_size)}")
            print()
        
        # Total progress
        current_size = self.get_total_downloaded_size()
        speed = self.calculate_speed(current_size)
        
        print("📊 Overall Progress:")
        print(f"   💾 Downloaded: {format_size(current_size)}")
        
        if speed > 0:
            print(f"   🚄 Speed: {format_size(speed)}/s")
            eta = self.estimate_remaining_time(current_size, speed)
            print(f"   ⏳ ETA: {eta}")
        
        # Progress bar (estimated)
        estimated_total = 23 * 1024 * 1024 * 1024  # 23GB estimate
        if current_size > 0:
            progress = min(100, (current_size / estimated_total) * 100)
            bar_length = 40
            filled_length = int(bar_length * progress / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"   📈 Progress: [{bar}] {progress:.1f}%")
        
        print()
        
        # Check processes
        processes = self.check_download_processes()
        if processes:
            print("🔄 Active Processes:")
            for process in processes[:3]:  # Show first 3
                # Truncate long process lines
                if len(process) > 80:
                    process = process[:77] + "..."
                print(f"   {process}")
            if len(processes) > 3:
                print(f"   ... and {len(processes) - 3} more")
        else:
            print("💤 No active download processes detected")
        
        print()
        print("💡 Press Ctrl+C to stop monitoring")
        print("💡 Run 'ollamadiffuser pull flux.1-dev' to resume download")
        
        # Update for next iteration
        self.last_size = current_size
        self.last_check_time = datetime.now()
        
        # Store size history for trend analysis
        self.size_history.append((datetime.now(), current_size))
        # Keep only last 10 measurements
        if len(self.size_history) > 10:
            self.size_history.pop(0)
    
    def run(self, interval=5):
        """Run the monitor"""
        print("🔍 Starting download progress monitor...")
        print(f"📊 Updating every {interval} seconds")
        print("💡 Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.display_progress()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n👋 Monitor stopped by user")
            print("📊 Final status:")
            
            current_size = self.get_total_downloaded_size()
            elapsed = datetime.now() - self.start_time
            
            print(f"   💾 Total downloaded: {format_size(current_size)}")
            print(f"   ⌛ Total time: {self.format_time(elapsed.total_seconds())}")
            
            if elapsed.total_seconds() > 0:
                avg_speed = current_size / elapsed.total_seconds()
                print(f"   📊 Average speed: {format_size(avg_speed)}/s")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor FLUX.1-dev download progress")
    parser.add_argument("--interval", "-i", type=int, default=5, 
                       help="Update interval in seconds (default: 5)")
    parser.add_argument("--model", "-m", type=str, default="flux.1-dev",
                       help="Model name to monitor (default: flux.1-dev)")
    
    args = parser.parse_args()
    
    monitor = DownloadMonitor(args.model)
    monitor.run(args.interval)

if __name__ == "__main__":
    main() 