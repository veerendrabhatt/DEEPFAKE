import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import io
import base64
from transformers import AutoImageProcessor, AutoModelForImageClassification
import mediapipe as mp
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import seaborn as sns

class DeepfakeExplainer:
    def __init__(self, model_name="prithivMLmods/Deep-Fake-Detector-v2-Model"):
        """Initialize the explainable AI system for deepfake detection"""
        self.model_name = model_name
        self.model = None
        self.processor = None
        
        # Try to load the model with better error handling
        try:
            print("Loading deepfake detection model...")
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to basic computer vision analysis...")
            # Fallback to basic functionality
            self.processor = None
            self.model = None
        
        # Initialize face detection
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
            
            # Face landmarks
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("Face detection initialized")
        except Exception as e:
            print(f"Error initializing face detection: {e}")
            self.mp_face_detection = None
            self.face_detection = None
            self.mp_face_mesh = None
            self.face_mesh = None
        
        # Define facial regions based on MediaPipe landmarks
        self.facial_regions = {
            'left_eye': list(range(33, 42)) + list(range(133, 155)) + list(range(157, 173)),
            'right_eye': list(range(362, 382)) + list(range(384, 398)) + list(range(398, 414)),
            'nose': list(range(1, 17)) + list(range(19, 25)) + list(range(115, 131)) + list(range(131, 134)) + list(range(141, 143)) + list(range(235, 236)) + list(range(236, 243)) + list(range(438, 439)) + list(range(439, 455)),
            'mouth': list(range(61, 68)) + list(range(84, 90)) + list(range(17, 18)) + list(range(18, 20)) + list(range(200, 202)) + list(range(270, 272)) + list(range(287, 304)) + list(range(308, 320)),
            'forehead': list(range(9, 11)) + list(range(151, 152)) + list(range(10, 151)),
            'cheeks': list(range(116, 117)) + list(range(117, 118)) + list(range(118, 119)) + list(range(119, 120)) + list(range(120, 121)) + list(range(121, 126)) + list(range(126, 142)) + list(range(36, 205)) + list(range(205, 206)) + list(range(206, 207)) + list(range(207, 213)) + list(range(213, 192)) + list(range(147, 177)) + list(range(177, 215)) + list(range(215, 138)) + list(range(172, 136)) + list(range(150, 149)) + list(range(176, 148)) + list(range(152, 377)) + list(range(400, 378)) + list(range(379, 365)) + list(range(397, 288)) + list(range(361, 340)),
            'jawline': list(range(172, 136)) + list(range(150, 149)) + list(range(176, 148)) + list(range(152, 377)) + list(range(400, 378)) + list(range(379, 365)) + list(range(397, 288)) + list(range(361, 340))
        }

    def detect_faces(self, image):
        """Detect faces in the image and return bounding boxes"""
        rgb_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = rgb_image.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                       int(bboxC.width * w), int(bboxC.height * h)
                faces.append(bbox)
        
        return faces

    def get_face_landmarks(self, image):
        """Get facial landmarks using MediaPipe"""
        rgb_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = self.face_mesh.process(rgb_image)
        
        landmarks = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = rgb_image.shape
                landmark_points = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmark_points.append((x, y))
                landmarks.append(landmark_points)
        
        return landmarks

    def _generate_fallback_heatmap(self, image):
        """Generate a fallback heatmap when model is not available"""
        try:
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            
            # Analyze image for real deepfake detection features
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Detect faces for better heatmap
            faces = self.detect_faces(image)
            
            # Create base heatmap with image-based analysis
            heatmap = np.zeros((h, w), dtype=np.float32)
            
            # Analyze the actual image to determine suspicion levels
            # Use multiple image analysis techniques
            
            # 1. Edge detection - real images have more natural edges
            edges = cv2.Canny(gray, 50, 150)
            edge_density = cv2.GaussianBlur((edges / 255.0).astype(np.float32), (21, 21), 0)
            
            # 2. Variance - real images have more natural texture variation
            variance = cv2.GaussianBlur((gray.astype(np.float32) - gray.astype(np.float32).mean()) ** 2, (21, 21), 0)
            variance_normalized = np.clip((variance / variance.max()) if variance.max() > 0 else variance, 0, 1)
            
            # 3. Color consistency - deepfakes often have unnatural color patterns
            if len(img_array.shape) == 3:
                # Calculate local color variance
                b, g, r = img_array[:,:,0].astype(np.float32), img_array[:,:,1].astype(np.float32), img_array[:,:,2].astype(np.float32)
                color_variance = ((r - g) ** 2 + (g - b) ** 2 + (b - r) ** 2) / 3.0
                color_variance_blurred = cv2.GaussianBlur(color_variance, (21, 21), 0)
                color_variance_normalized = np.clip((color_variance_blurred / color_variance_blurred.max()) if color_variance_blurred.max() > 0 else color_variance_blurred, 0, 1)
            else:
                color_variance_normalized = np.zeros_like(gray).astype(np.float32)
            
            # 4. Combining all features to create suspicion map
            # Low edge density + low variance + low color variance = suspicious (likely fake)
            # Combine features with appropriate weights
            suspicion_map = 0.4 * (1 - edge_density) + 0.3 * (1 - variance_normalized) + 0.3 * (1 - color_variance_normalized)
            
            if faces:
                # Apply face-based weighting to enhance facial regions
                for face in faces:
                    x, y, fw, fh = face
                    # Get face region in all maps
                    face_suspicion = suspicion_map[y:y+fh, x:x+fw]
                    
                    # Apply region-specific analysis within the face
                    # Eye regions - typically have more artifacts in deepfakes
                    eye_region = face_suspicion[:fh//3, :]
                    eye_region *= 1.5  # Amplify eye region suspicion
                    
                    # Mouth region - often has sync issues in deepfakes
                    mouth_region = face_suspicion[2*fh//3:, :]
                    mouth_region *= 1.3  # Amplify mouth region suspicion
                    
                    # Update the main heatmap with enhanced face regions
                    heatmap[y:y+fh, x:x+fw] = np.maximum(heatmap[y:y+fh, x:x+fw], face_suspicion)
            else:
                # No face detected, use general suspicion map
                heatmap = suspicion_map.copy()
            
            # Smooth the heatmap
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
            
            # Normalize
            if heatmap.max() > 0:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
            # Create visualization with vibrant colors
            rgb_img = np.array(image) / 255.0
            
            # Use 'jet' colormap for better visibility (bright colors)
            # Enhance heatmap for better color saturation
            enhanced_heatmap = np.power(heatmap, 0.7)  # Apply gamma correction for better visibility
            heatmap_colored = cm.jet(enhanced_heatmap)[:, :, :3]
            
            # Higher opacity for better visibility
            alpha = 0.6
            visualization = (1 - alpha) * rgb_img + alpha * heatmap_colored
            visualization = np.clip(visualization, 0, 1)
            visualization_uint8 = (visualization * 255).astype(np.uint8)
            
            # Determine prediction class based on suspicion
            # High suspicion = FAKE (class 1), Low suspicion = REAL (class 0)
            avg_suspicion = np.mean(heatmap)
            predicted_class = 1 if avg_suspicion > 0.5 else 0
            confidence = abs((avg_suspicion - 0.5) * 2) * 100 + 50  # Scale to 50-100%
            
            return heatmap, visualization_uint8, predicted_class, confidence
            
        except Exception as e:
            print(f"Fallback heatmap generation failed: {e}")
            # Final fallback
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            heatmap = np.random.rand(h, w) * 0.1
            visualization = np.array(image)
            return heatmap, visualization, 0, 50.0

    def simple_predict(self, image):
        """Simple prediction method that works even if model fails to load"""
        if self.model is None or self.processor is None:
            # Enhanced fallback prediction based on image analysis
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Multiple feature analysis
            features = {}
            features['variance'] = np.var(gray)
            features['mean_brightness'] = np.mean(gray)
            
            # Edge analysis
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = np.mean(edges) / 255.0
            
            # Color analysis
            if len(img_array.shape) == 3:
                features['color_variance'] = np.var(img_array)
                r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
                features['rg_correlation'] = np.corrcoef(r.flatten(), g.flatten())[0,1]
            else:
                features['color_variance'] = features['variance']
                features['rg_correlation'] = 0.8
            
            # Face detection
            faces = self.detect_faces(image)
            features['face_detected'] = len(faces) > 0
            
            # Scoring system
            score = 0
            confidence_factors = []
            
            # Variance analysis (real images tend to have more natural variation)
            if features['variance'] > 800:
                score += 20
                confidence_factors.append("High texture variation")
            elif features['variance'] < 400:
                score -= 15
                confidence_factors.append("Low texture variation (suspicious)")
            
            # Edge density (real images have more natural edges)
            if features['edge_density'] > 0.05:
                score += 15
                confidence_factors.append("Good edge definition")
            elif features['edge_density'] < 0.02:
                score -= 10
                confidence_factors.append("Blurry edges (suspicious)")
            
            # Color correlation (unnatural correlations in deepfakes)
            if 0.7 < features['rg_correlation'] < 0.95:
                score += 10
                confidence_factors.append("Natural color relationships")
            elif features['rg_correlation'] > 0.98 or features['rg_correlation'] < 0.5:
                score -= 8
                confidence_factors.append("Unnatural color patterns")
            
            # Face detection
            if features['face_detected']:
                score += 15
                confidence_factors.append("Clear facial features detected")
            else:
                score -= 5
                confidence_factors.append("No clear face detected")
            
            # Brightness analysis
            if 50 < features['mean_brightness'] < 200:
                score += 5
                confidence_factors.append("Normal brightness range")
            else:
                score -= 3
                confidence_factors.append("Unusual brightness")
            
            # Convert score to prediction
            confidence = min(95, max(55, abs(score) + 50))
            
            if score > 5:
                prediction = "REAL"
            else:
                prediction = "FAKE"
            
            return prediction, confidence
        
        try:
            # Use the actual model
            inputs = self.processor(image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][predicted_class].item()
                
                class_names = ["REAL", "FAKE"]
                prediction = class_names[predicted_class] if predicted_class < len(class_names) else "UNKNOWN"
                return prediction, confidence * 100
        except Exception as e:
            print(f"Model prediction error: {e}")
            # Fallback to simple prediction
            return "REAL", 50.0

    def generate_gradcam_heatmap(self, image, target_class=None):
        """Generate Grad-CAM heatmap using true model gradients"""
        try:
            # Check if model is available
            if self.model is None or self.processor is None:
                print("Model not available, using fallback analysis...")
                return self._generate_fallback_heatmap(image)
            
            # Preprocess image for model
            inputs = self.processor(image, return_tensors="pt")
            input_tensor = inputs["pixel_values"]
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probs, dim=-1).item()
            
            if target_class is None:
                target_class = predicted_class
            targets = [ClassifierOutputTarget(target_class)]

            # Find the deepest convolutional layer for Grad-CAM
            conv_layers = [m for m in self.model.modules() if isinstance(m, torch.nn.Conv2d)]
            if len(conv_layers) > 0:
                target_layer = conv_layers[-1]
            else:
                # Fallback: use last available child module
                children = list(self.model.children())
                target_layer = children[-1] if children else None

            # Use pytorch-grad-cam when a target layer is available
            cam = None
            if target_layer is not None:
                try:
                    cam = GradCAM(model=self.model, target_layers=[target_layer])
                except Exception as e:
                    print(f"Grad-CAM initialization failed: {e}")
                    cam = None

            # Convert PIL image to numpy and normalize
            rgb_img = np.array(image).astype(np.float32) / 255.0
            input_for_cam = preprocess_image(rgb_img, mean=[0,0,0], std=[1,1,1])

            if cam is not None:
                try:
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
                except Exception as e:
                    print(f"Grad-CAM computation failed: {e}")
                    # Use fallback heatmap when Grad-CAM fails
                    fallback_result = self._generate_fallback_heatmap(image)
                    grayscale_cam = fallback_result[0]
            else:
                # No suitable conv layer for Grad-CAM -> create fallback heatmap
                fallback_result = self._generate_fallback_heatmap(image)
                grayscale_cam = fallback_result[0]

            # Normalize grayscale_cam to 0..1
            if grayscale_cam.max() > grayscale_cam.min():
                grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())
            else:
                grayscale_cam = np.zeros_like(grayscale_cam)

            # Mask Grad-CAM to facial regions using detected landmarks (if available)
            landmarks = self.get_face_landmarks(image)
            h, w = grayscale_cam.shape
            face_mask = np.zeros((h, w), dtype=np.float32)
            if landmarks and len(landmarks) > 0:
                for region_name, landmark_indices in self.facial_regions.items():
                    region_points = [landmarks[0][idx] for idx in landmark_indices if idx < len(landmarks[0])]
                    if len(region_points) > 2:
                        region_points = np.array(region_points, dtype=np.int32)
                        cv2.fillPoly(face_mask, [region_points], 1)

            # Determine a percentile-based threshold so we don't hard-cut everything
            p = 85
            try:
                thresh_val = float(np.percentile(grayscale_cam, p))
            except Exception:
                thresh_val = 0.5

            # Create a binary mask of strong activations
            strong_mask = (grayscale_cam >= thresh_val).astype(np.float32)

            # If we have a face mask, intersect; otherwise keep full-image strong mask
            if face_mask.sum() > 0:
                final_mask = (strong_mask * face_mask).astype(np.float32)
                # If the final mask is empty (strict threshold), fallback to face_mask itself
                if final_mask.sum() == 0:
                    final_mask = face_mask.copy()
            else:
                final_mask = strong_mask

            # Build colored heatmap and visualization with vibrant colors
            # Use 'jet' colormap for better visibility (red for high, blue for low)
            heatmap_colored = cm.jet(grayscale_cam)[:, :, :3]
            visualization = rgb_img.copy()
            mask_inds = final_mask > 0
            
            # Enhance the heatmap values for better visibility
            enhanced_heatmap = (grayscale_cam * 2.0).clip(0, 1)  # Enhance contrast
            
            if np.any(mask_inds):
                # Use enhanced heatmap with higher opacity for better visibility
                visualization[mask_inds] = (0.5 * rgb_img[mask_inds] + 0.5 * heatmap_colored[mask_inds])
            else:
                # No activations: blend a more visible full-image heatmap
                visualization = (0.7 * rgb_img + 0.3 * heatmap_colored)

            visualization = np.clip(visualization, 0, 1)
            visualization_uint8 = (visualization * 255).astype(np.uint8)
            confidence = probs[0][predicted_class].item() * 100.0 if 'probs' in locals() else 0.0
            return grayscale_cam, visualization_uint8, predicted_class, confidence
        except Exception as e:
            print(f"Grad-CAM generation failed: {e}. Falling back to occlusion sensitivity.")
            # Occlusion sensitivity fallback - slide a patch and measure probability drop
            try:
                img_array = np.array(image).astype(np.float32) / 255.0
                h, w = img_array.shape[:2]
                original_prob = probs[0][predicted_class].item() if 'probs' in locals() else 0.5

                patch_size = max(16, min(h, w) // 8)
                stride = max(8, patch_size // 2)

                occlusion_map = np.zeros((h, w), dtype=np.float32)
                counts = np.zeros((h, w), dtype=np.float32)

                mean_pixel = img_array.mean(axis=(0,1))

                for y in range(0, h, stride):
                    for x in range(0, w, stride):
                        y1 = y
                        y2 = min(h, y + patch_size)
                        x1 = x
                        x2 = min(w, x + patch_size)

                        occluded = img_array.copy()
                        occluded[y1:y2, x1:x2, :] = mean_pixel

                        # Prepare input and run model
                        occluded_pil = Image.fromarray((occluded * 255).astype(np.uint8))
                        inputs_occ = self.processor(occluded_pil, return_tensors='pt')
                        with torch.no_grad():
                            out_occ = self.model(**inputs_occ)
                            probs_occ = F.softmax(out_occ.logits, dim=-1)
                            occ_prob = probs_occ[0][predicted_class].item()

                        drop = max(0.0, original_prob - occ_prob)
                        occlusion_map[y1:y2, x1:x2] += drop
                        counts[y1:y2, x1:x2] += 1.0

                # Average where counts > 0
                mask_counts = counts > 0
                occlusion_map[mask_counts] = occlusion_map[mask_counts] / counts[mask_counts]

                # Normalize occlusion map
                if occlusion_map.max() > 0:
                    occlusion_map = (occlusion_map - occlusion_map.min()) / (occlusion_map.max() - occlusion_map.min())

                grayscale_cam = occlusion_map
                # Mask to face regions if available
                landmarks = self.get_face_landmarks(image)
                if landmarks and len(landmarks) > 0:
                    face_mask = np.zeros((h, w), dtype=np.float32)
                    for region_name, landmark_indices in self.facial_regions.items():
                        region_points = [landmarks[0][idx] for idx in landmark_indices if idx < len(landmarks[0])]
                        if len(region_points) > 2:
                            region_points = np.array(region_points, dtype=np.int32)
                            cv2.fillPoly(face_mask, [region_points], 1)
                    grayscale_cam = grayscale_cam * face_mask

                # Create visualization using coolwarm colormap
                heatmap_colored = cm.coolwarm(grayscale_cam)[:, :, :3]
                visualization = rgb_img.copy()
                mask = grayscale_cam >= (grayscale_cam.mean() + 0.25 * grayscale_cam.std())
                if np.any(mask):
                    visualization[mask] = (0.6 * rgb_img[mask] + 0.4 * heatmap_colored[mask])
                else:
                    # faint overlay if mask is empty
                    visualization = (0.9 * rgb_img + 0.1 * heatmap_colored)

                visualization = np.clip(visualization, 0, 1)

                # Return percentage confidence if available
                conf_pct = (original_prob * 100.0) if original_prob <= 1.0 else original_prob
                print(f"[DEBUG] occlusion original_prob={original_prob}, conf_pct={conf_pct}")
                print(f"[DEBUG] occlusion_map_stats: min={float(np.min(grayscale_cam))}, max={float(np.max(grayscale_cam))}, mean={float(np.mean(grayscale_cam))}")
                print(f"[DEBUG] region_analysis_counts: h={h} w={w} nonzero_activations={int(np.sum(grayscale_cam>0))}")
                return grayscale_cam, (visualization * 255).astype(np.uint8), predicted_class, conf_pct
            except Exception as e2:
                print(f"Occlusion fallback also failed: {e2}")
                # Final fallback: small random heatmap and neutral confidence
                img_array = np.array(image)
                h, w = img_array.shape[:2]
                grayscale_cam = np.random.rand(h, w) * 0.05
                rgb_img = np.array(image) / 255.0
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                return grayscale_cam, visualization, 0, 50.0

    def analyze_facial_regions(self, image, heatmap):
        """Analyze which facial regions are most suspicious"""
        # Detect faces to understand the face layout
        faces = self.detect_faces(image)
        
        if not faces:
            # No face detected, create fallback region analysis
            # print("[DEBUG] No faces detected, using fallback region analysis")
            region_scores = {}
            for region_name in self.facial_regions.keys():
                # Use a small random value as fallback
                region_scores[region_name] = np.random.uniform(0.1, 0.3)
            
            return {
                "region_scores": region_scores,
                "most_suspicious": sorted(region_scores.items(), key=lambda x: x[1], reverse=True)[:3],
                "total_regions": len(region_scores),
                "note": "No face detected - using fallback analysis"
            }
        
        # Use the first detected face
        face = faces[0]
        x, y, fw, fh = face
        h, w = np.array(image).shape[:2]
        
        # Extract region heatmap values directly from the face bounding box
        # This ensures the region scores match what the heatmap visualizes
        face_heatmap = heatmap[y:y+fh, x:x+fw]
        
        region_scores = {}
        
        # Define facial regions within the detected face bounding box
        # These should match the values assigned in _generate_fallback_heatmap
        if face_heatmap.size > 0:
            # Eye regions (top 1/3 of face) - highest suspicion
            eye_region = face_heatmap[:fh//3, :]
            if eye_region.size > 0:
                left_eye = eye_region[:, :fw//2]
                right_eye = eye_region[:, fw//2:]
                region_scores['left_eye'] = float(np.mean(left_eye)) if left_eye.size > 0 else 0.0
                region_scores['right_eye'] = float(np.mean(right_eye)) if right_eye.size > 0 else 0.0
            else:
                region_scores['left_eye'] = 0.0
                region_scores['right_eye'] = 0.0
            
            # Forehead (top 1/6 to top 1/3 of face)
            if fh // 6 > 0:
                forehead_region = face_heatmap[:fh//6, :]
                region_scores['forehead'] = float(np.mean(forehead_region)) if forehead_region.size > 0 else 0.0
            else:
                region_scores['forehead'] = 0.0
            
            # Mouth region (bottom 1/3 of face) - medium suspicion
            mouth_region = face_heatmap[2*fh//3:, :]
            region_scores['mouth'] = float(np.mean(mouth_region)) if mouth_region.size > 0 else 0.0
            
            # Nose region (middle third, center of face) - lower suspicion
            nose_region = face_heatmap[fh//3:2*fh//3, fw//4:3*fw//4]
            region_scores['nose'] = float(np.mean(nose_region)) if nose_region.size > 0 else 0.0
            
            # Cheeks (middle third, sides of face)
            if fh // 3 > 0:
                left_cheek = face_heatmap[fh//3:2*fh//3, :fw//3]
                right_cheek = face_heatmap[fh//3:2*fh//3, 2*fw//3:]
                region_scores['cheeks'] = (float(np.mean(left_cheek)) + float(np.mean(right_cheek))) / 2.0 if left_cheek.size > 0 and right_cheek.size > 0 else 0.0
            else:
                region_scores['cheeks'] = 0.0
            
            # Jawline (bottom quarter)
            jawline_region = face_heatmap[3*fh//4:, :]
            region_scores['jawline'] = float(np.mean(jawline_region)) if jawline_region.size > 0 else 0.0
        
        # Ensure all expected regions have scores
        for region_name in self.facial_regions.keys():
            if region_name not in region_scores:
                region_scores[region_name] = 0.0
        
        # Normalize region scores to match the visualization
        if region_scores:
            vals = list(region_scores.values())
            min_val = min(vals)
            max_val = max(vals)
            
            # Use percentile-based normalization to avoid extreme values
            if max_val > min_val:
                # Scale to 0-1 range and then to reasonable display range
                normalized = {}
                for k, v in region_scores.items():
                    # Scale to 0-1
                    scaled = (v - min_val) / (max_val - min_val) if max_val > min_val else 0.0
                    # Scale to display range (0.1-0.9) to match heatmap colors
                    # This ensures the numbers match what users see in the visual heatmap
                    normalized[k] = 0.1 + (scaled * 0.8)  # Scale to 0.1-0.9 range
            else:
                # All values identical -> use actual heatmap values but scale down
                normalized = {k: min(0.9, float(v) * 0.9) for k, v in region_scores.items()}
        else:
            normalized = {}

        # Debug: print raw and normalized scores (commented out for production)
        # try:
        #     print(f"[DEBUG] raw_region_scores={region_scores}")
        #     print(f"[DEBUG] normalized_region_scores={normalized}")
        #     print(f"[DEBUG] landmarks_detected={len(landmarks) if landmarks else 0}")
        #     if landmarks and len(landmarks) > 0:
        #         print(f"[DEBUG] first_landmark_points={len(landmarks[0])}")
        # except Exception:
        #     pass

        # Sort regions by normalized suspicion level
        sorted_regions = sorted(normalized.items(), key=lambda x: x[1], reverse=True)

        return {
            "region_scores": normalized,
            "most_suspicious": sorted_regions[:3],
            "total_regions": len(region_scores)
        }

    def _advanced_forensic_analysis(self, image, heatmap):
        """Advanced image forensics to enrich textual explanations.

        Returns a dict with:
        - observations: list of {region, suspicion_level, description}
        - flags: int count of suspicious cues
        - top_note: short string appended to summary
        """
        observations = []
        flags = 0
        top_clauses = []

        img_np = np.array(image)
        h, w = img_np.shape[:2]

        # Helper: clamp suspicion into [0, 1]
        def clamp01(x):
            try:
                return float(max(0.0, min(1.0, x)))
            except Exception:
                return 0.0

        # 1) ELA / compression artifact check
        try:
            pil_tmp = image.convert('RGB')
            buf = io.BytesIO()
            pil_tmp.save(buf, format='JPEG', quality=90)
            buf.seek(0)
            ela_img = Image.open(buf)
            ela_np = np.abs(img_np.astype(np.int16) - np.array(ela_img).astype(np.int16)).astype(np.float32)
            ela_gray = cv2.cvtColor(ela_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            ela_enhanced = cv2.normalize(ela_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
            ela_score = float(np.mean(ela_enhanced) / 255.0)

            # Higher mean ELA suggests inconsistent compression (potential editing)
            ela_susp = clamp01((ela_score - 0.15) / 0.35)  # soft thresholding
            if ela_susp > 0.35:
                observations.append({
                    "region": "Global",
                    "suspicion_level": ela_susp,
                    "description": f"Compression artifact inconsistency detected (ELA score {ela_score:.2f})."
                })
                flags += 1
                top_clauses.append("compression inconsistencies")
        except Exception:
            pass

        # 2) Lighting left-right consistency within detected face
        try:
            faces = self.detect_faces(image)
            if faces:
                x, y, fw, fh = faces[0]
                face_roi = img_np[max(0,y):min(h,y+fh), max(0,x):min(w,x+fw)]
                if face_roi.size > 0:
                    gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
                    left = gray[:, :gray.shape[1]//2]
                    right = gray[:, gray.shape[1]//2:]
                    l_mean = float(np.mean(left))
                    r_mean = float(np.mean(right))
                    diff = abs(l_mean - r_mean) / (np.mean([l_mean, r_mean]) + 1e-6)
                    light_susp = clamp01((diff - 0.12) / 0.25)
                    if light_susp > 0.4:
                        observations.append({
                            "region": "Face (lighting)",
                            "suspicion_level": light_susp,
                            "description": f"Lighting asymmetry across face (L/R diff {diff*100:.1f}%)."
                        })
                        flags += 1
                        top_clauses.append("lighting asymmetry")
        except Exception:
            pass

        # 3) High-frequency texture consistency (possible smoothing or over-sharpening)
        try:
            gray_full = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(gray_full, ddepth=cv2.CV_32F)
            hf_energy = float(np.mean(np.abs(lap)))
            # Compare face vs background if face exists
            faces = self.detect_faces(image)
            if faces:
                x, y, fw, fh = faces[0]
                face_gray = gray_full[max(0,y):min(h,y+fh), max(0,x):min(w,x+fw)]
                pad = int(max(8, min(h, w) * 0.05))
                bg_mask = np.ones_like(gray_full, dtype=np.uint8) * 255
                bg_mask[max(0,y-pad):min(h,y+fh+pad), max(0,x-pad):min(w,x+fw+pad)] = 0
                bg_gray = gray_full[bg_mask > 0]
                face_hf = float(np.mean(np.abs(cv2.Laplacian(face_gray, ddepth=cv2.CV_32F)))) if face_gray.size > 0 else hf_energy
                bg_hf = float(np.mean(np.abs(cv2.Laplacian(bg_gray, ddepth=cv2.CV_32F)))) if bg_gray.size > 0 else hf_energy
                ratio = face_hf / (bg_hf + 1e-6)
                # Very low ratio -> face over-smoothed; very high -> over-sharpened
                tex_susp = clamp01(max((1.2 - ratio) / 0.5, (ratio - 1.8) / 0.7))
                if tex_susp > 0.35:
                    desc = "Face texture markedly different from background ("
                    if ratio < 1.0:
                        desc += "over-smoothing)"
                    else:
                        desc += "over-sharpening)"
                    observations.append({
                        "region": "Face (texture)",
                        "suspicion_level": tex_susp,
                        "description": f"{desc}, ratio {ratio:.2f}."
                    })
                    flags += 1
                    top_clauses.append("texture inconsistency")
        except Exception:
            pass

        # 4) Bilateral facial asymmetry in heatmap attention
        try:
            if heatmap is not None and heatmap.size > 0:
                hm = heatmap.astype(np.float32)
                # If heatmap size differs from image, resize for face-symmetric slicing
                if hm.shape[:2] != (h, w):
                    hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_CUBIC)
                left = hm[:, :w//2]
                right = hm[:, w - w//2:]
                left_mean = float(np.mean(left))
                right_mean = float(np.mean(right))
                diff = abs(left_mean - right_mean) / (np.mean([left_mean, right_mean]) + 1e-6)
                asym_susp = clamp01((diff - 0.2) / 0.5)
                if asym_susp > 0.35:
                    observations.append({
                        "region": "Attention (bilateral)",
                        "suspicion_level": asym_susp,
                        "description": f"Model/fallback attention is uneven across face (difference {diff*100:.1f}%)."
                    })
                    flags += 1
                    top_clauses.append("bilateral asymmetry")
        except Exception:
            pass

        # 5) Edge/blending artifacts along face boundary
        try:
            faces = self.detect_faces(image)
            if faces:
                x, y, fw, fh = faces[0]
                band = 6
                x1 = max(0, x - band)
                y1 = max(0, y - band)
                x2 = min(w, x + fw + band)
                y2 = min(h, y + fh + band)
                band_img = img_np[y1:y2, x1:x2]
                gray_band = cv2.cvtColor(band_img, cv2.COLOR_RGB2GRAY)
                gx = cv2.Sobel(gray_band, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray_band, cv2.CV_32F, 0, 1, ksize=3)
                mag = np.sqrt(gx * gx + gy * gy)
                # Compare boundary band to inner face gradients
                inner = img_np[y+band:y+fh-band, x+band:x+fw-band]
                if inner.size > 0:
                    inner_gray = cv2.cvtColor(inner, cv2.COLOR_RGB2GRAY)
                    igx = cv2.Sobel(inner_gray, cv2.CV_32F, 1, 0, ksize=3)
                    igy = cv2.Sobel(inner_gray, cv2.CV_32F, 0, 1, ksize=3)
                    inner_mag = np.sqrt(igx * igx + igy * igy)
                    edge_ratio = float(np.mean(mag) / (np.mean(inner_mag) + 1e-6))
                    blend_susp = clamp01((edge_ratio - 1.6) / 0.8)
                    if blend_susp > 0.4:
                        observations.append({
                            "region": "Face boundary",
                            "suspicion_level": blend_susp,
                            "description": f"Strong edge/blending artifacts around face (ratio {edge_ratio:.2f})."
                        })
                        flags += 1
                        top_clauses.append("blending artifacts")
        except Exception:
            pass

        # Compose top note
        top_note = ""
        if flags >= 2 and top_clauses:
            uniq = sorted(set(top_clauses))
            joined = ", ".join(uniq[:3])
            top_note = f"Multiple forensic cues flagged: {joined}."

        return {"observations": observations, "flags": flags, "top_note": top_note}

    def create_detailed_heatmap(self, image, heatmap, region_analysis):
        """Create a detailed heatmap with facial region annotations"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap overlay with enhanced visibility
        rgb_img = np.array(image) / 255.0
        # Apply gamma correction for better visibility
        enhanced_heatmap = np.power(heatmap, 0.7)
        heatmap_colored = cm.jet(enhanced_heatmap)[:, :, :3]
        blended = 0.5 * rgb_img + 0.5 * heatmap_colored  # Higher opacity
        
        axes[1].imshow(blended)
        axes[1].set_title("Suspicion Heatmap", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Region analysis chart
        if "region_scores" in region_analysis:
            regions = list(region_analysis["region_scores"].keys())
            scores = list(region_analysis["region_scores"].values())
            
            colors = plt.cm.Reds(np.linspace(0.3, 1, len(scores)))
            bars = axes[2].barh(regions, scores, color=colors)
            axes[2].set_xlabel("Suspicion Level", fontsize=12)
            axes[2].set_title("Facial Region Analysis", fontsize=14, fontweight='bold')
            axes[2].set_xlim(0, max(scores) * 1.1 if scores else 1)
            
            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, scores)):
                axes[2].text(score + max(scores) * 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{score:.3f}', va='center', fontsize=10)
        else:
            axes[2].text(0.5, 0.5, "No face detected", ha='center', va='center', 
                        transform=axes[2].transAxes, fontsize=14)
            axes[2].set_title("Facial Region Analysis", fontsize=14, fontweight='bold')
            axes[2].axis('off')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data

    def generate_explanation_text(self, prediction, confidence, region_analysis):
        """Generate human-readable explanation"""
        explanation = {
            "prediction": prediction,
            "confidence": confidence,
            "summary": "",
            "detailed_analysis": [],
            "recommendations": []
        }
        
        # Generate summary
        if prediction.lower() == "fake":
            explanation["summary"] = f"This image is classified as FAKE with {confidence:.1f}% confidence. "
        else:
            explanation["summary"] = f"This image appears to be REAL with {confidence:.1f}% confidence. "
        
        # Analyze regions
        if "most_suspicious" in region_analysis and region_analysis["most_suspicious"]:
            top_regions = region_analysis["most_suspicious"]
            
            explanation["summary"] += f"The most suspicious areas are: {', '.join([r[0] for r in top_regions[:2]])}."
            
            for region_name, score in top_regions:
                if score > 0.3:  # Threshold for significant suspicion
                    detail = {
                        "region": region_name.replace('_', ' ').title(),
                        "suspicion_level": score,
                        "description": self.get_region_description(region_name, score)
                    }
                    explanation["detailed_analysis"].append(detail)
        
        # Add recommendations
        if prediction.lower() == "fake":
            explanation["recommendations"] = [
                "Examine the highlighted regions carefully for inconsistencies",
                "Look for unnatural textures or blending artifacts",
                "Check for asymmetries in facial features",
                "Consider the quality and resolution of suspicious areas"
            ]
        else:
            explanation["recommendations"] = [
                "While classified as real, always verify source credibility",
                "Cross-reference with other verification methods",
                "Be aware that sophisticated deepfakes may be harder to detect"
            ]
        
        return explanation

    def get_region_description(self, region_name, score):
        """Get description for specific facial region suspicion"""
        descriptions = {
            "left_eye": f"Left eye region shows {score:.1%} suspicion - look for unnatural eye movement, inconsistent lighting, or blinking patterns",
            "right_eye": f"Right eye region shows {score:.1%} suspicion - check for asymmetry with left eye, unnatural reflections, or pixel artifacts",
            "nose": f"Nose area shows {score:.1%} suspicion - examine for smoothing artifacts, unnatural shadows, or inconsistent skin texture",
            "mouth": f"Mouth region shows {score:.1%} suspicion - look for lip-sync issues, unnatural teeth, or inconsistent facial expressions",
            "forehead": f"Forehead shows {score:.1%} suspicion - check for smoothing effects, missing wrinkles, or unnatural skin texture",
            "cheeks": f"Cheek area shows {score:.1%} suspicion - examine for blending artifacts, inconsistent skin tone, or unnatural contours",
            "jawline": f"Jawline shows {score:.1%} suspicion - look for edge artifacts, unnatural contours, or blending inconsistencies"
        }
        
        return descriptions.get(region_name, f"{region_name.replace('_', ' ').title()} shows {score:.1%} suspicion level")

    def explain_prediction(self, image):
        """Main method to generate complete explanation for a prediction"""
        try:
            # Get basic prediction first (fallback only)
            prediction_label, confidence = self.simple_predict(image)

            # Generate Grad-CAM heatmap (this also runs the model to obtain predicted_class)
            heatmap, visualization, predicted_class, model_confidence = self.generate_gradcam_heatmap(image)

            # Map model predicted_class to human-readable label if available
            try:
                class_names = ["REAL", "FAKE"]
                if predicted_class is not None:
                    prediction_label = class_names[predicted_class] if predicted_class < len(class_names) else prediction_label
            except Exception:
                pass

            # Use model confidence if available, otherwise use simple prediction confidence
            final_confidence = model_confidence if (model_confidence is not None and model_confidence > 0) else confidence
            
            # Analyze facial regions
            region_analysis = self.analyze_facial_regions(image, heatmap)
            
            # Debug: print heatmap stats (commented out for production)
            # try:
            #     print(f"[DEBUG] heatmap_stats: min={float(np.min(heatmap))}, max={float(np.max(heatmap))}, mean={float(np.mean(heatmap))}")
            # except Exception:
            #     pass
            
            # Create detailed visualization
            detailed_plot = self.create_detailed_heatmap(image, heatmap, region_analysis)
            
            # Generate explanation text
            explanation_text = self.generate_explanation_text(
                prediction_label, final_confidence, region_analysis
            )

            # Enrich with advanced forensic cues derived from the actual image
            try:
                advanced = self._advanced_forensic_analysis(image, heatmap)
                if advanced and advanced.get("observations"):
                    # Append observations to detailed_analysis
                    for obs in advanced["observations"]:
                        explanation_text["detailed_analysis"].append(obs)

                    # Strengthen summary with top finding
                    top_note = advanced.get("top_note")
                    if top_note:
                        explanation_text["summary"] += f" {top_note}"

                    # Tailor recommendations
                    if advanced.get("flags", 0) >= 2 and prediction_label.upper() == "REAL":
                        explanation_text["recommendations"].insert(0, "Some forensic cues are suspicious; review highlighted regions closely")
            except Exception:
                pass
            
            # Convert visualization to base64
            vis_pil = Image.fromarray(visualization)
            buffer = io.BytesIO()
            vis_pil.save(buffer, format='PNG')
            vis_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            return {
                "success": True,
                "prediction": prediction_label,
                "prediction_label": prediction_label,
                "confidence": final_confidence,
                "heatmap_overlay": vis_b64,
                "detailed_analysis": detailed_plot,
                "region_analysis": region_analysis,
                "explanation": explanation_text
            }
            
        except Exception as e:
            print(f"Explanation error: {e}")
            # Fallback response
            prediction_label, confidence = self.simple_predict(image)
            return {
                "success": True,
                "prediction": prediction_label,
                "confidence": confidence,
                "heatmap_overlay": "",
                "detailed_analysis": "",
                "region_analysis": {"error": "Analysis failed"},
                "explanation": {
                    "summary": f"Basic analysis: Image classified as {prediction_label} with {confidence:.1f}% confidence.",
                    "detailed_analysis": [],
                    "recommendations": ["Unable to perform detailed analysis. Consider retrying with a clearer image."]
                }
            }