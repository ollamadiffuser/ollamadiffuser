try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

import numpy as np
from PIL import Image
import logging
from typing import Optional, Union, Tuple, Dict, Any
import torch

logger = logging.getLogger(__name__)

class ControlNetPreprocessorManager:
    """Lazy-loading manager for ControlNet preprocessors"""
    
    def __init__(self):
        self.processors = {}
        self._initialized = False
        self._initialization_attempted = False
        self._available_types = []
        
        # Check dependencies on initialization
        if not CV2_AVAILABLE:
            logger.warning(
                "OpenCV (cv2) is not installed. "
                "Install it with: pip install opencv-python>=4.8.0"
            )
    
    def is_initialized(self) -> bool:
        """Check if preprocessors are initialized"""
        return self._initialized
    
    def is_available(self) -> bool:
        """Check if ControlNet preprocessors are available"""
        if not CV2_AVAILABLE:
            return False
            
        if not self._initialization_attempted:
            # Try a lightweight check without full initialization
            try:
                import controlnet_aux
                return True
            except ImportError:
                return False
        return self._initialized
    
    def initialize(self, force: bool = False) -> bool:
        """
        Initialize ControlNet preprocessors
        
        Args:
            force: Force re-initialization even if already initialized
            
        Returns:
            True if initialization successful, False otherwise
        """
        if not CV2_AVAILABLE:
            raise ImportError(
                "OpenCV (cv2) is required for ControlNet preprocessors. "
                "Install it with: pip install opencv-python>=4.8.0"
            )
            
        if self._initialized and not force:
            return True
            
        if self._initialization_attempted and not force:
            return self._initialized
            
        self._initialization_attempted = True
        logger.info("Initializing ControlNet preprocessors...")
        
        try:
            from controlnet_aux import (
                CannyDetector,
                MidasDetector,
                OpenposeDetector,
                HEDdetector,
                MLSDdetector,
                NormalBaeDetector,
                LineartDetector,
                LineartAnimeDetector,
                ContentShuffleDetector,
                ZoeDetector
            )
            
            # Initialize processors with proper error handling
            self.processors = {}
            
            # Canny detector (no model download needed)
            try:
                self.processors['canny'] = CannyDetector()
                logger.info("Canny detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Canny detector: {e}")
            
            # Depth detectors with fallback
            try:
                # Try MiDaS first with default model
                self.processors['depth'] = MidasDetector.from_pretrained('lllyasviel/Annotators')
                logger.info("MiDaS depth detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MiDaS detector: {e}")
                try:
                    # Try ZoeDepth as fallback
                    self.processors['depth_zoe'] = ZoeDetector.from_pretrained('lllyasviel/Annotators')
                    self.processors['depth'] = self.processors['depth_zoe']  # Use as main depth
                    logger.info("ZoeDepth detector initialized as fallback")
                except Exception as e2:
                    logger.warning(f"Failed to initialize ZoeDepth detector: {e2}")
            
            # OpenPose detector
            try:
                self.processors['openpose'] = OpenposeDetector.from_pretrained('lllyasviel/Annotators')
                logger.info("OpenPose detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenPose detector: {e}")
            
            # HED detector
            try:
                self.processors['hed'] = HEDdetector.from_pretrained('lllyasviel/Annotators')
                logger.info("HED detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize HED detector: {e}")
            
            # MLSD detector
            try:
                self.processors['mlsd'] = MLSDdetector.from_pretrained('lllyasviel/Annotators')
                logger.info("MLSD detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MLSD detector: {e}")
            
            # Normal detector
            try:
                self.processors['normal'] = NormalBaeDetector.from_pretrained('lllyasviel/Annotators')
                logger.info("Normal detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Normal detector: {e}")
            
            # Lineart detectors
            try:
                self.processors['lineart'] = LineartDetector.from_pretrained('lllyasviel/Annotators')
                logger.info("Lineart detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Lineart detector: {e}")
            
            try:
                self.processors['lineart_anime'] = LineartAnimeDetector.from_pretrained('lllyasviel/Annotators')
                logger.info("Lineart Anime detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Lineart Anime detector: {e}")
            
            # Content shuffle (no model download needed)
            try:
                self.processors['shuffle'] = ContentShuffleDetector()
                logger.info("Content Shuffle detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Content Shuffle detector: {e}")
            
            # Add scribble as alias for HED
            if 'hed' in self.processors:
                self.processors['scribble'] = self.processors['hed']
            
            if self.processors:
                self._initialized = True
                self._available_types = list(self.processors.keys())
                logger.info(f"ControlNet preprocessors initialized: {self._available_types}")
                return True
            else:
                logger.warning("No ControlNet preprocessors could be initialized, falling back to basic processors")
                self._init_basic_processors()
                return True
            
        except ImportError as e:
            logger.warning(f"controlnet-aux not available: {e}")
            # Fallback to basic OpenCV-based processors
            self._init_basic_processors()
            return True
        except Exception as e:
            logger.error(f"Error initializing ControlNet preprocessors: {e}")
            # Fallback to basic OpenCV-based processors
            self._init_basic_processors()
            return True
    
    def _init_basic_processors(self):
        """Initialize basic OpenCV-based processors as fallback"""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, using minimal fallback processors")
            self.processors = {
                'canny': self._simple_edge_fallback,
                'depth': self._simple_depth_fallback,
                'scribble': self._simple_edge_fallback,
            }
        else:
            logger.info("Using basic OpenCV-based preprocessors")
            self.processors = {
                'canny': self._canny_opencv,
                'depth': self._depth_basic,
                'scribble': self._scribble_basic,
            }
        self._initialized = True
        self._available_types = list(self.processors.keys())
    
    def _canny_opencv(self, image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
        """Basic Canny edge detection using OpenCV"""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required for Canny edge detection")
            
        # Convert PIL to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply Canny edge detection
        edges = cv2.Canny(image_cv, low_threshold, high_threshold)
        
        # Convert back to PIL
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)
    
    def _depth_basic(self, image: Image.Image) -> Image.Image:
        """Basic depth estimation using simple gradients"""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required for depth estimation")
            
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Create a simple depth map using gradients
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine gradients
        depth = np.sqrt(grad_x**2 + grad_y**2)
        depth = np.uint8(255 * depth / np.max(depth))
        
        # Convert to RGB
        depth_rgb = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(depth_rgb)
    
    def _scribble_basic(self, image: Image.Image) -> Image.Image:
        """Basic scribble detection using edge detection"""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required for scribble detection")
            
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to make lines thicker
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Convert to RGB
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)
    
    def _simple_edge_fallback(self, image: Image.Image, **kwargs) -> Image.Image:
        """Simple edge detection fallback using PIL/numpy only"""
        # Convert to numpy array
        img_array = np.array(image.convert('L'))  # Convert to grayscale
        
        # Simple gradient-based edge detection
        grad_x = np.abs(np.diff(img_array, axis=1))
        grad_y = np.abs(np.diff(img_array, axis=0))
        
        # Pad to maintain original size
        grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
        grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')
        
        # Combine gradients
        edges = np.sqrt(grad_x**2 + grad_y**2)
        edges = np.uint8(255 * edges / np.max(edges) if np.max(edges) > 0 else edges)
        
        # Convert to RGB
        edges_rgb = np.stack([edges, edges, edges], axis=2)
        return Image.fromarray(edges_rgb)
    
    def _simple_depth_fallback(self, image: Image.Image, **kwargs) -> Image.Image:
        """Simple depth estimation fallback using PIL/numpy only"""
        # Convert to grayscale
        gray = np.array(image.convert('L'))
        
        # Simple depth based on intensity (brighter = closer)
        depth = 255 - gray  # Invert so darker areas are "further"
        
        # Convert to RGB
        depth_rgb = np.stack([depth, depth, depth], axis=2)
        return Image.fromarray(depth_rgb)
    
    def preprocess(self, 
                   image: Union[Image.Image, str], 
                   control_type: str,
                   **kwargs) -> Image.Image:
        """
        Preprocess image for ControlNet
        
        Args:
            image: Input image (PIL Image or path)
            control_type: Type of control (canny, depth, openpose, etc.)
            **kwargs: Additional parameters for specific processors
            
        Returns:
            Preprocessed control image
        """
        # Initialize if not already done
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize ControlNet preprocessors")
        
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL Image or file path")
        
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get processor
        if control_type not in self.processors:
            available = list(self.processors.keys())
            raise ValueError(f"Control type '{control_type}' not available. Available: {available}")
        
        processor = self.processors[control_type]
        
        try:
            if callable(processor):
                # Basic OpenCV processor
                return processor(image, **kwargs)
            else:
                # ControlNet-aux processor
                return processor(image, **kwargs)
                
        except Exception as e:
            logger.error(f"Failed to preprocess image with {control_type}: {e}")
            # Return original image as fallback
            return image
    
    def get_available_types(self) -> list:
        """Get list of available control types"""
        if not self._initialized:
            # Return cached types if available, otherwise return basic types
            if self._available_types:
                return self._available_types
            elif self.is_available():
                return ['canny', 'depth', 'openpose', 'hed', 'mlsd', 'normal', 'lineart', 'lineart_anime', 'shuffle', 'scribble']
            else:
                return ['canny', 'depth', 'scribble']  # Basic OpenCV types
        return list(self.processors.keys())
    
    def resize_for_controlnet(self, 
                             image: Image.Image, 
                             width: int = 512, 
                             height: int = 512) -> Image.Image:
        """Resize image for ControlNet while maintaining aspect ratio"""
        # Calculate aspect ratio
        aspect_ratio = image.width / image.height
        
        if aspect_ratio > 1:
            # Landscape
            new_width = width
            new_height = int(width / aspect_ratio)
        else:
            # Portrait
            new_height = height
            new_width = int(height * aspect_ratio)
        
        # Resize image
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with target size and paste resized image
        result = Image.new('RGB', (width, height), (0, 0, 0))
        
        # Calculate position to center the image
        x = (width - new_width) // 2
        y = (height - new_height) // 2
        
        result.paste(resized, (x, y))
        return result

# Global manager instance - no initialization at import time
controlnet_preprocessor = ControlNetPreprocessorManager() 