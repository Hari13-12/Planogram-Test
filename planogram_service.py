import numpy as np
import cv2
import time
import logging
import os
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
from PIL import ImageDraw
import onnxruntime as ort
from yolo_onnx.yolov8_onnx import YOLOv8
from collections import Counter
from inference_sdk import InferenceHTTPClient 
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
 
class PlanogramAnalyzer:
    """
    A comprehensive planogram analysis system that combines YOLO detection
    with ResNet classification for retail product identification.
    """
   
    def __init__(self, resnet_model_path: str,
                 detection_size: int = 640, conf_threshold: float = 0.4,
                 iou_threshold: float = 0.5, classification_threshold: float = 0.50):
        self.detection_size = detection_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classification_threshold = classification_threshold
        # self.yolo_detector = None
        self.resnet_session = None
        # self.yolo_model_path = yolo_model_path
        self.resnet_model_path = resnet_model_path
        self.client = InferenceHTTPClient(
            api_url=os.getenv("ROBOFLOW_API_URL"),
            api_key=os.getenv("ROBOFLOW_API_KEY")
        )
       

 
        self.class_names = ['Coke','Fanta','RedBull','Sprite', 'ThumbsUP', 'Competitor']
        # Create index mapping for class names
        self.class_name_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
       
        # Load models
        # self._load_models(yolo_model_path, resnet_model_path)
        logger.info("PlanogramAnalyzer initialized successfully")
   
    async def initialize(self):
        if self.resnet_model_path is None:
            logger.error("Model paths not provided")
            raise ValueError("Model paths not provided")
           
        if not os.path.exists(self.resnet_model_path):
            logger.error(f"ResNet model not found at: {self.resnet_model_path}")
            raise FileNotFoundError(f"ResNet model not found at: {self.resnet_model_path}")
           
        # Load models
        await self._load_models()
 
    async def _load_models(self) -> None:
        """Load YOLO and ResNet models."""
        try:
            # Load ResNet model with simplified settings
            try:
                providers = ['CPUExecutionProvider']
               
                # Use default session options for better compatibility
                self.resnet_session = ort.InferenceSession(
                    self.resnet_model_path,
                    providers=providers
                )
            except Exception as resnet_error:
                logger.error(f"Failed to load ResNet model: {resnet_error}")
                raise RuntimeError(f"Failed to load ResNet model: {resnet_error}") from resnet_error
           
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise RuntimeError("Failed to load models") from e
   
 
   
    async def load_and_resize_image(self, original_image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Load and resize image for processing.
       
        Args:
            image_path: Path to the image file
           
        Returns:
            Tuple of (original_image, resized_image)
        """
        try:
            # original_image = Image.open(image_path)
           
            if original_image.mode != 'RGB':
                print("Different channels")
                original_image = original_image.convert('RGB')
                print("Image got converted to RGB")
            else:
                print("3 channels")
            resized_image = original_image.resize(
                (self.detection_size, self.detection_size),
                Image.Resampling.LANCZOS
            )
            logger.info(f"Image loaded and resized: {original_image.size} -> {resized_image.size}")
            return original_image, resized_image
            # return original_image, original_image  # Return original image for detection to avoid resizing issues
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise RuntimeError("Failed to load image") from e
   
    async def detect_objects(self, image: Image.Image) -> List[Dict]:
        """
        Detect objects using Roboflow segmentation workflow.
        
        Args:
            image: PIL Image for detection
            
        Returns:
            List of detection results (each dict has x, y, width, height, confidence, class, points)
        """
        try:
            # Convert PIL Image to numpy array (cv2 format)
            image_np = np.array(image)
            image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            result = self.client.run_workflow(
                workspace_name="college-k8kps",
                workflow_id="general-segmentation-api",
                images={"image": image_cv2},   # pass numpy array directly
                parameters={"classes": "Big_Coke-Drinks"},
                use_cache=True
            )
            predictions = result[0]["predictions"]["predictions"]
            logger.info(f"Roboflow detections completed: {len(predictions)} found")
            return predictions
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            raise

    async def crop_detected_objects(self, original_image: Image.Image,
                                detections: List[Dict],
                                min_size: int = 20) -> Tuple[List[Image.Image], List[List[int]]]:
        """
        Crop detected objects from the original image.
        
        Args:
            original_image: Original PIL Image
            detections: List of Roboflow detections (from detect_objects)
            min_size: Minimum width/height in pixels to keep a detection
            
        Returns:
            Tuple of (cropped_images, coordinates)
            - cropped_images: List of PIL Images, one per valid detection
            - coordinates: List of [x1, y1, x2, y2] for each crop
        """
        try:
            original_np = np.array(original_image)
            orig_w, orig_h = original_image.size  # PIL: (width, height)

            cropped_images = []
            coordinates = []

            for pred in detections:
                w = pred["width"]
                h = pred["height"]

                # Skip noise/tiny detections
                if w < min_size or h < min_size:
                    logger.debug(f"Skipping small detection: {w:.0f}x{h:.0f}")
                    continue

                cx, cy = pred["x"], pred["y"]

                # Compute bbox and clamp to image bounds
                x1 = max(0, int(cx - w / 2))
                y1 = max(0, int(cy - h / 2))
                x2 = min(orig_w, int(cx + w / 2))
                y2 = min(orig_h, int(cy + h / 2))

                # Ensure valid crop dimensions
                if x2 > x1 and y2 > y1:
                    cropped_region = original_np[y1:y2, x1:x2]
                    if cropped_region.size > 0:
                        cropped_images.append(Image.fromarray(cropped_region))
                        coordinates.append([x1, y1, x2, y2])

            logger.info(f"Cropped {len(cropped_images)} objects from detections")
            return cropped_images, coordinates
        except Exception as e:
            logger.error(f"Error cropping objects: {e}")
            raise
   
    async def preprocess_for_resnet(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for ResNet classification using EXACT same preprocessing as Gradio.
        This is the key fix - using the same preprocessing as tensorflow.keras.applications.resnet50.preprocess_input
       
        Args:
            image: PIL Image to preprocess
           
        Returns:
            Preprocessed numpy array
        """
        try:
            # EXACT same preprocessing as Gradio version
            # Resize to 256x256 and convert to RGB (matching Gradio version)
            img = image.resize((256, 256)).convert("RGB")
            img = np.asarray(img).astype(np.float32)
 
            # Match Keras preprocessing: mean/std normalization
            img -= np.mean(img)
            img /= np.std(img)
 
            # ONNX models from Keras usually expect (1, H, W, C)
            img_array = np.expand_dims(img, axis=0)
            logger.debug(f"Preprocessed image shape: {img_array.shape}, dtype: {img_array.dtype}")
           
            return img_array
           
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
   
           
    async def classify_products(self, cropped_images: List[Image.Image]) -> Tuple[List[str], List[float]]:
        """
        Classify cropped product images using ResNet.
        Fixed version matching Gradio logic exactly, but without entropy.
       
        Args:
            cropped_images: List of cropped PIL Images
           
        Returns:
            Tuple of (predicted_classes, confidences)
        """
        try:
            predicted_classes = []
            confidences = []
           
            if not cropped_images:
                logger.warning("No cropped images provided for classification")
                return predicted_classes, confidences
           
            logger.info("Starting classification process")
           
            # Get input details from ONNX model
            input_name = self.resnet_session.get_inputs()[0].name
            input_shape = self.resnet_session.get_inputs()[0].shape
            logger.info(f"ONNX model input name: {input_name}, shape: {input_shape}")
           
            # Process each image
            for i, img in enumerate(cropped_images):
                try:
                    # Preprocess image with EXACT same preprocessing as Gradio
                    img_array = await self.preprocess_for_resnet(img)
                    logger.debug(f"Processing image {i+1}/{len(cropped_images)}, shape: {img_array.shape}, dtype: {img_array.dtype}")
                   
                    # Run inference
                    outputs = self.resnet_session.run(None, {input_name: img_array})
                    predictions = outputs[0]
                    predicted_class = np.argmax(predictions)
                    confidence = np.max(predictions)
                   
                    if confidence < self.classification_threshold:
                        final_class = "Competitor Product"
                    else:
                        # Check if the predicted class is "Competitor"
                        if self.class_names[predicted_class] == "Competitor":
                            final_class = "Competitor Product"
                        else:
                            final_class = self.class_names[predicted_class]
                
                    predicted_classes.append(final_class)
                    confidences.append(float(confidence))
                    print(confidence,final_class)
                   
                    logger.debug(f"Image {i+1}: Raw prediction: {predicted_class}, Confidence: {confidence:.3f}, Final: {final_class}")
                   
                except Exception as e:
                    logger.error(f"Error processing image {i+1}: {e}")
                    # Add fallback values
                    predicted_classes.append("Competitor Product")
                    confidences.append(0.0)
           
            logger.info(f"Classification completed: {len(predicted_classes)} products classified")
            return predicted_classes, confidences
           
        except Exception as e:
            logger.error(f"Error in product classification: {e}")
            raise
    
    
   
    async def results(self, image: Image.Image) -> Dict[str, Any]:
        try:
            # Step 1: Load and resize image
            original_image, resized_image = await self.load_and_resize_image(image)
           
            # Step 2: Detect objects
            detections = await self.detect_objects(image)
           
            if not detections:
                return {
                    'total_products': 0,
                    'own_products': 0,
                    'competitor_products': 0,
                    'product_details': []
                }
           
            # Step 3: Crop detected objects
            cropped_images, coordinates = await self.crop_detected_objects(original_image, detections)
           
            # Step 4: Classify products
            predictions, confidences = await self.classify_products(cropped_images)
           
            # Clean up memory
            del original_image
            del resized_image
            del cropped_images
           
            # Compile results
            own_products = [p for p in predictions if p != "Competitor Product"]
            competitor_count = predictions.count("Competitor Product")
           
            product_details = []
            for i, (coords, prediction) in enumerate(zip(coordinates, predictions)):
                product_details.append({
                    'id': i + 1,
                    'class': prediction,
                    'coordinates': coords,
                    'is_competitor': prediction == "Competitor Product"
                })
           
            results = {
                'total_products': len(predictions),
                'own_products': len(own_products),
                'competitor_products': competitor_count,
                'own_product_list': own_products,
                'product_details': product_details
            }
           
            return results
           
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            raise
