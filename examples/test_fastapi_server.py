#!/usr/bin/env python3
"""
Test script for the new FastAPI-only server
"""

import sys
import time
import requests
import threading
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_server():
    """Test the FastAPI server"""
    print("🔍 Testing FastAPI server...")
    
    try:
        from ollamadiffuser.api.server import create_app
        import uvicorn
        
        # Create the app
        app = create_app()
        print("  ✅ FastAPI app created successfully")
        
        # Test server startup in a separate thread
        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        time.sleep(3)
        
        # Test health endpoint
        try:
            response = requests.get("http://127.0.0.1:8000/api/health", timeout=5)
            if response.status_code == 200:
                print("  ✅ Health endpoint working")
                print(f"     Response: {response.json()}")
            else:
                print(f"  ❌ Health endpoint failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Failed to connect to server: {e}")
            return False
        
        # Test models endpoint
        try:
            response = requests.get("http://127.0.0.1:8000/api/models", timeout=5)
            if response.status_code == 200:
                print("  ✅ Models endpoint working")
                models_data = response.json()
                print(f"     Available models: {len(models_data.get('available', []))}")
                print(f"     Installed models: {len(models_data.get('installed', []))}")
            else:
                print(f"  ❌ Models endpoint failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Failed to connect to models endpoint: {e}")
            return False
        
        print("  ✅ FastAPI server test completed successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ Server test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing OllamaDiffuser FastAPI Server")
    print("=" * 50)
    
    success = test_server()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! FastAPI server is working correctly.")
        print("\nYou can now start the server with:")
        print("  python -m ollamadiffuser.api.server")
        print("  or")
        print("  ollamadiffuser serve")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 