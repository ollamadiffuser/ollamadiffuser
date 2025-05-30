#!/usr/bin/env python3
"""
Generate sample control images for ControlNet demonstration
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
import os

def create_canny_samples():
    """Create sample images good for canny edge detection"""
    
    # 1. Simple geometric shapes
    img = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(img)
    
    # Rectangle
    draw.rectangle([50, 50, 200, 150], outline='black', width=3)
    # Circle
    draw.ellipse([300, 50, 450, 200], outline='black', width=3)
    # Triangle
    draw.polygon([(100, 300), (200, 200), (300, 300)], outline='black', width=3)
    # Diamond
    draw.polygon([(400, 250), (450, 300), (400, 350), (350, 300)], outline='black', width=3)
    
    img.save('ollamadiffuser/ui/samples/canny/geometric_shapes.png')
    
    # 2. Simple house outline
    img = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(img)
    
    # House base
    draw.rectangle([150, 250, 350, 400], outline='black', width=3)
    # Roof
    draw.polygon([(130, 250), (250, 150), (370, 250)], outline='black', width=3)
    # Door
    draw.rectangle([220, 320, 280, 400], outline='black', width=2)
    # Windows
    draw.rectangle([170, 280, 210, 320], outline='black', width=2)
    draw.rectangle([290, 280, 330, 320], outline='black', width=2)
    # Chimney
    draw.rectangle([300, 170, 330, 220], outline='black', width=2)
    
    img.save('ollamadiffuser/ui/samples/canny/house_outline.png')
    
    # 3. Portrait silhouette
    img = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(img)
    
    # Head outline
    draw.ellipse([180, 100, 330, 280], outline='black', width=3)
    # Neck
    draw.rectangle([235, 280, 275, 320], outline='black', width=3)
    # Shoulders
    draw.arc([150, 300, 360, 450], start=0, end=180, fill='black', width=3)
    
    img.save('ollamadiffuser/ui/samples/canny/portrait_outline.png')

def create_depth_samples():
    """Create sample depth maps"""
    
    # 1. Radial gradient (good for centered subjects)
    img = Image.new('RGB', (512, 512), 'white')
    pixels = np.zeros((512, 512, 3), dtype=np.uint8)
    
    center_x, center_y = 256, 256
    max_distance = 200
    
    for y in range(512):
        for x in range(512):
            distance = min(np.sqrt((x - center_x)**2 + (y - center_y)**2), max_distance)
            intensity = int(255 * (1 - distance / max_distance))
            pixels[y, x] = [intensity, intensity, intensity]
    
    Image.fromarray(pixels).save('ollamadiffuser/ui/samples/depth/radial_gradient.png')
    
    # 2. Linear perspective (good for landscapes, roads)
    img = Image.new('RGB', (512, 512), 'white')
    pixels = np.zeros((512, 512, 3), dtype=np.uint8)
    
    for y in range(512):
        # Create perspective effect - closer at bottom, farther at top
        intensity = int(255 * (y / 512))
        pixels[y, :] = [intensity, intensity, intensity]
    
    Image.fromarray(pixels).save('ollamadiffuser/ui/samples/depth/linear_perspective.png')
    
    # 3. Simple 3D sphere
    img = Image.new('RGB', (512, 512), 'black')
    pixels = np.zeros((512, 512, 3), dtype=np.uint8)
    
    center_x, center_y = 256, 256
    radius = 150
    
    for y in range(512):
        for x in range(512):
            dx = x - center_x
            dy = y - center_y
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance <= radius:
                # Calculate sphere depth using sphere equation
                z = np.sqrt(radius**2 - distance**2)
                intensity = int(255 * (z / radius))
                pixels[y, x] = [intensity, intensity, intensity]
    
    Image.fromarray(pixels).save('ollamadiffuser/ui/samples/depth/sphere_3d.png')

def create_openpose_samples():
    """Create sample pose images (simplified stick figures)"""
    
    # 1. Standing pose
    img = Image.new('RGB', (512, 512), 'black')
    draw = ImageDraw.Draw(img)
    
    # Head
    draw.ellipse([240, 80, 270, 110], fill='white')
    # Body
    draw.line([255, 110, 255, 250], fill='white', width=4)
    # Arms
    draw.line([255, 150, 200, 200], fill='white', width=4)  # Left arm
    draw.line([255, 150, 310, 200], fill='white', width=4)  # Right arm
    # Legs
    draw.line([255, 250, 220, 350], fill='white', width=4)  # Left leg
    draw.line([255, 250, 290, 350], fill='white', width=4)  # Right leg
    
    img.save('ollamadiffuser/ui/samples/openpose/standing_pose.png')
    
    # 2. Action pose (running)
    img = Image.new('RGB', (512, 512), 'black')
    draw = ImageDraw.Draw(img)
    
    # Head
    draw.ellipse([240, 80, 270, 110], fill='white')
    # Body (slightly tilted)
    draw.line([255, 110, 270, 250], fill='white', width=4)
    # Arms (running motion)
    draw.line([255, 150, 180, 180], fill='white', width=4)  # Left arm back
    draw.line([255, 150, 320, 120], fill='white', width=4)  # Right arm forward
    # Legs (running motion)
    draw.line([270, 250, 240, 350], fill='white', width=4)  # Left leg
    draw.line([270, 250, 320, 320], fill='white', width=4)  # Right leg forward
    
    img.save('ollamadiffuser/ui/samples/openpose/running_pose.png')
    
    # 3. Sitting pose
    img = Image.new('RGB', (512, 512), 'black')
    draw = ImageDraw.Draw(img)
    
    # Head
    draw.ellipse([240, 100, 270, 130], fill='white')
    # Body
    draw.line([255, 130, 255, 220], fill='white', width=4)
    # Arms
    draw.line([255, 170, 200, 220], fill='white', width=4)  # Left arm
    draw.line([255, 170, 310, 220], fill='white', width=4)  # Right arm
    # Legs (bent for sitting)
    draw.line([255, 220, 220, 280], fill='white', width=4)  # Left thigh
    draw.line([220, 280, 200, 350], fill='white', width=4)  # Left shin
    draw.line([255, 220, 290, 280], fill='white', width=4)  # Right thigh
    draw.line([290, 280, 310, 350], fill='white', width=4)  # Right shin
    
    img.save('ollamadiffuser/ui/samples/openpose/sitting_pose.png')

def create_scribble_samples():
    """Create sample scribble/sketch images"""
    
    # 1. Simple tree sketch
    img = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(img)
    
    # Tree trunk
    draw.line([256, 400, 256, 250], fill='black', width=8)
    # Tree crown (rough circle)
    points = []
    for i in range(20):
        angle = (i / 20) * 2 * math.pi
        radius = 80 + 20 * math.sin(i * 3)  # Irregular circle
        x = 256 + radius * math.cos(angle)
        y = 200 + radius * math.sin(angle) * 0.8
        points.append((x, y))
    
    for i in range(len(points)):
        next_i = (i + 1) % len(points)
        draw.line([points[i], points[next_i]], fill='black', width=3)
    
    img.save('ollamadiffuser/ui/samples/scribble/tree_sketch.png')
    
    # 2. Simple face sketch
    img = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(img)
    
    # Face outline
    draw.ellipse([180, 150, 330, 320], outline='black', width=3)
    # Eyes
    draw.ellipse([210, 200, 230, 220], outline='black', width=2)
    draw.ellipse([280, 200, 300, 220], outline='black', width=2)
    # Nose
    draw.line([255, 230, 255, 250], fill='black', width=2)
    draw.line([255, 250, 245, 260], fill='black', width=2)
    # Mouth
    draw.arc([230, 270, 280, 300], start=0, end=180, fill='black', width=2)
    
    img.save('ollamadiffuser/ui/samples/scribble/face_sketch.png')
    
    # 3. Simple car sketch
    img = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(img)
    
    # Car body
    draw.rectangle([100, 250, 400, 320], outline='black', width=3)
    # Car roof
    draw.rectangle([150, 200, 350, 250], outline='black', width=3)
    # Wheels
    draw.ellipse([130, 320, 170, 360], outline='black', width=3)
    draw.ellipse([330, 320, 370, 360], outline='black', width=3)
    # Windows
    draw.rectangle([170, 210, 220, 240], outline='black', width=2)
    draw.rectangle([280, 210, 330, 240], outline='black', width=2)
    
    img.save('ollamadiffuser/ui/samples/scribble/car_sketch.png')

def create_sample_metadata():
    """Create metadata file describing each sample"""
    metadata = {
        "canny": {
            "geometric_shapes.png": {
                "title": "Geometric Shapes",
                "description": "Perfect for generating architectural elements, logos, or geometric art",
                "good_for": ["architecture", "logos", "geometric patterns", "modern art"]
            },
            "house_outline.png": {
                "title": "House Outline", 
                "description": "Great for generating buildings, houses, or architectural scenes",
                "good_for": ["buildings", "houses", "architecture", "real estate"]
            },
            "portrait_outline.png": {
                "title": "Portrait Silhouette",
                "description": "Ideal for generating portraits, characters, or people",
                "good_for": ["portraits", "characters", "people", "headshots"]
            }
        },
        "depth": {
            "radial_gradient.png": {
                "title": "Radial Depth",
                "description": "Perfect for centered subjects with depth, like portraits or objects",
                "good_for": ["portraits", "centered objects", "product photography", "focus effects"]
            },
            "linear_perspective.png": {
                "title": "Linear Perspective", 
                "description": "Great for landscapes, roads, or scenes with distance",
                "good_for": ["landscapes", "roads", "horizons", "perspective scenes"]
            },
            "sphere_3d.png": {
                "title": "3D Sphere",
                "description": "Ideal for round objects, balls, or 3D elements",
                "good_for": ["spheres", "balls", "3D objects", "rounded elements"]
            }
        },
        "openpose": {
            "standing_pose.png": {
                "title": "Standing Pose",
                "description": "Basic standing position, great for portraits and character art",
                "good_for": ["standing portraits", "character design", "fashion", "formal poses"]
            },
            "running_pose.png": {
                "title": "Running Pose",
                "description": "Dynamic action pose, perfect for sports or movement scenes",
                "good_for": ["sports", "action scenes", "dynamic poses", "movement"]
            },
            "sitting_pose.png": {
                "title": "Sitting Pose",
                "description": "Relaxed sitting position, ideal for casual or indoor scenes",
                "good_for": ["casual portraits", "indoor scenes", "relaxed poses", "sitting figures"]
            }
        },
        "scribble": {
            "tree_sketch.png": {
                "title": "Tree Sketch",
                "description": "Simple tree drawing, great for nature and landscape scenes",
                "good_for": ["nature", "landscapes", "trees", "outdoor scenes"]
            },
            "face_sketch.png": {
                "title": "Face Sketch",
                "description": "Basic face outline, perfect for portrait generation",
                "good_for": ["portraits", "faces", "character art", "headshots"]
            },
            "car_sketch.png": {
                "title": "Car Sketch",
                "description": "Simple vehicle outline, ideal for automotive or transportation themes",
                "good_for": ["cars", "vehicles", "transportation", "automotive"]
            }
        }
    }
    
    import json
    with open('ollamadiffuser/ui/samples/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    print("🎨 Creating ControlNet sample images...")
    
    print("📐 Creating Canny edge samples...")
    create_canny_samples()
    
    print("🏔️ Creating depth map samples...")
    create_depth_samples()
    
    print("🕺 Creating OpenPose samples...")
    create_openpose_samples()
    
    print("✏️ Creating scribble samples...")
    create_scribble_samples()
    
    print("📋 Creating metadata...")
    create_sample_metadata()
    
    print("\n✅ Sample images created successfully!")
    print("📁 Samples saved to: ollamadiffuser/ui/samples/")
    print("\n🎛️ Sample types created:")
    print("  • Canny: 3 edge detection samples")
    print("  • Depth: 3 depth map samples") 
    print("  • OpenPose: 3 pose samples")
    print("  • Scribble: 3 sketch samples")
    print("\n💡 These samples will appear in the Web UI for easy ControlNet testing!") 