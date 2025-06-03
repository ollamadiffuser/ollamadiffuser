#!/usr/bin/env python3
"""
Test script for the new model registry system.
This script tests the new dynamic model management features while ensuring
backward compatibility with existing functionality.
"""

import sys
import json
import tempfile
from pathlib import Path

def test_model_registry():
    """Test the model registry functionality"""
    print("🔍 Testing Model Registry System...")
    
    try:
        from ollamadiffuser.core.config.model_registry import model_registry
        from ollamadiffuser.core.models.manager import model_manager
        
        # Test 1: Verify default models are loaded
        print("\n1. Testing default model loading...")
        default_models = model_registry.get_all_models()
        print(f"   ✅ Loaded {len(default_models)} default models")
        
        # Verify some expected default models
        expected_models = ['flux.1-dev', 'flux.1-schnell', 'stable-diffusion-xl-base']
        for model in expected_models:
            if model in default_models:
                print(f"   ✅ Found expected model: {model}")
            else:
                print(f"   ❌ Missing expected model: {model}")
                return False
        
        # Test 2: Test adding a model at runtime
        print("\n2. Testing runtime model addition...")
        test_model_config = {
            "repo_id": "test/test-model",
            "model_type": "test",
            "variant": "fp16"
        }
        
        if model_registry.add_model("test-model", test_model_config):
            print("   ✅ Successfully added test model at runtime")
            
            # Verify it appears in the registry
            if "test-model" in model_registry.get_model_names():
                print("   ✅ Test model appears in registry")
            else:
                print("   ❌ Test model not found in registry")
                return False
        else:
            print("   ❌ Failed to add test model")
            return False
        
        # Test 3: Test model manager integration
        print("\n3. Testing model manager integration...")
        available_models = model_manager.list_available_models()
        if "test-model" in available_models:
            print("   ✅ Test model appears in manager's available models")
        else:
            print("   ❌ Test model not found in manager's available models")
            return False
        
        # Test 4: Test model info retrieval
        print("\n4. Testing model info retrieval...")
        model_info = model_manager.get_model_info("test-model")
        if model_info and model_info.get("repo_id") == "test/test-model":
            print("   ✅ Model info retrieved correctly")
        else:
            print("   ❌ Model info retrieval failed")
            return False
        
        # Test 5: Test configuration file loading
        print("\n5. Testing configuration file loading...")
        test_config = {
            "models": {
                "config-test-model": {
                    "repo_id": "test/config-model",
                    "model_type": "sd15",
                    "variant": "fp16",
                    "license_info": {
                        "type": "MIT",
                        "commercial_use": True
                    }
                }
            }
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            temp_config_path = f.name
        
        try:
            # Load the config file
            model_registry._load_config_file(Path(temp_config_path))
            
            if "config-test-model" in model_registry.get_model_names():
                print("   ✅ Configuration file loaded successfully")
                
                # Verify model details
                config_model_info = model_registry.get_model("config-test-model")
                if (config_model_info and 
                    config_model_info.get("repo_id") == "test/config-model" and
                    config_model_info.get("license_info", {}).get("type") == "MIT"):
                    print("   ✅ Configuration model details are correct")
                else:
                    print("   ❌ Configuration model details are incorrect")
                    return False
            else:
                print("   ❌ Configuration file loading failed")
                return False
        finally:
            # Clean up temp file
            Path(temp_config_path).unlink()
        
        # Test 6: Test model removal
        print("\n6. Testing model removal...")
        if model_registry.remove_model("test-model"):
            print("   ✅ Successfully removed test model")
            
            if "test-model" not in model_registry.get_model_names():
                print("   ✅ Test model no longer in registry")
            else:
                print("   ❌ Test model still in registry after removal")
                return False
        else:
            print("   ❌ Failed to remove test model")
            return False
        
        # Test 7: Test backward compatibility
        print("\n7. Testing backward compatibility...")
        # Verify the old model_registry property still works
        old_style_registry = model_manager.model_registry
        if isinstance(old_style_registry, dict) and len(old_style_registry) > 0:
            print("   ✅ Backward compatibility maintained")
        else:
            print("   ❌ Backward compatibility broken")
            return False
        
        print("\n✅ All model registry tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Model registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_compatibility():
    """Test that CLI commands still work with the new system"""
    print("\n🔍 Testing CLI Compatibility...")
    
    try:
        from ollamadiffuser.core.models.manager import model_manager
        
        # Test that existing CLI-style operations work
        available_models = model_manager.list_available_models()
        installed_models = model_manager.list_installed_models()
        
        print(f"   ✅ Available models: {len(available_models)}")
        print(f"   ✅ Installed models: {len(installed_models)}")
        
        # Test model info for a default model
        if available_models:
            first_model = available_models[0]
            model_info = model_manager.get_model_info(first_model)
            if model_info:
                print(f"   ✅ Model info retrieved for {first_model}")
            else:
                print(f"   ❌ Failed to get model info for {first_model}")
                return False
        
        print("   ✅ CLI compatibility verified!")
        return True
        
    except Exception as e:
        print(f"   ❌ CLI compatibility test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Testing OllamaDiffuser Model Management System")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 2
    
    # Test model registry
    if test_model_registry():
        tests_passed += 1
    
    # Test CLI compatibility
    if test_cli_compatibility():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The model management system is working correctly.")
        print("\nYou can now:")
        print("  • Use all existing CLI commands unchanged")
        print("  • Add custom models via configuration files")
        print("  • Use the new 'ollamadiffuser registry' commands")
        print("  • Share model configurations with your team")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 