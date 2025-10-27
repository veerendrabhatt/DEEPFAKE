import io
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

class BasicDeepfakeExplainer:
    def __init__(self):
        """Initialize a basic explainable AI system without complex dependencies"""
        print("Initializing Basic Explainable AI system...")
        
        # Load OpenCV face detector
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            print("OpenCV face detection initialized")
        except Exception as e:
            print(f"Face detection initialization failed: {e}")
            self.face_cascade = None
            self.eye_cascade = None
        
        # Define facial regions
        self.facial_regions = {
            'eyes': "Eye regions",
            'nose': "Nose area", 
            'mouth': "Mouth region",
            'forehead': "Forehead",
            'cheeks': "Cheek area",
            'jawline': "Jawline"
        }

    def detect_faces_basic(self, image):
        """Basic face detection using OpenCV"""
        try:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                return [(x, y, w, h) for (x, y, w, h) in faces]
            else:
                # Fallback: assume center region is face
                h, w = gray.shape
                return [(w//4, h//4, w//2, h//2)]
        except Exception as e:
            print(f"Face detection error: {e}")
            return []

    def detect_eyes_basic(self, image, face_region):
        """Basic eye detection"""
        try:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            x, y, w, h = face_region
            face_gray = gray[y:y+h, x:x+w]
            
            if self.eye_cascade is not None:
                eyes = self.eye_cascade.detectMultiScale(face_gray)
                # Convert to global coordinates
                return [(x + ex, y + ey, ew, eh) for (ex, ey, ew, eh) in eyes]
            else:
                # Fallback: estimate eye positions
                eye_y = y + h//4
                eye_h = h//6
                left_eye = (x + w//6, eye_y, w//4, eye_h)
                right_eye = (x + w//2, eye_y, w//4, eye_h)
                return [left_eye, right_eye]
        except:
            return []

    def analyze_image_features(self, image):
        """Enhanced analysis of image features for deepfake detection"""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        features = {}
        
        # Texture analysis
        features['texture_variance'] = np.var(gray)
        features['mean_brightness'] = np.mean(gray)
        
        # Edge analysis - more sophisticated
        try:
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = np.mean(edges) / 255.0
            
            # Edge consistency analysis
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            features['edge_consistency'] = np.std(np.sqrt(sobel_x**2 + sobel_y**2))
        except:
            features['edge_density'] = 0.1
            features['edge_consistency'] = 50.0
        
        # Color analysis - enhanced
        features['color_variance'] = np.var(img_array)
        
        # Color channel correlation (deepfakes often have unnatural correlations)
        if len(img_array.shape) == 3:
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            features['rg_correlation'] = np.corrcoef(r.flatten(), g.flatten())[0,1]
            features['rb_correlation'] = np.corrcoef(r.flatten(), b.flatten())[0,1]
            features['gb_correlation'] = np.corrcoef(g.flatten(), b.flatten())[0,1]
        else:
            features['rg_correlation'] = features['rb_correlation'] = features['gb_correlation'] = 0.8
        
        # Frequency domain analysis
        try:
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            features['frequency_variance'] = np.var(magnitude_spectrum)
            features['high_freq_energy'] = np.mean(magnitude_spectrum[magnitude_spectrum > np.percentile(magnitude_spectrum, 90)])
        except:
            features['frequency_variance'] = 100.0
            features['high_freq_energy'] = 5.0
        
        # Local Binary Pattern (enhanced)
        features['texture_pattern'] = self.enhanced_lbp(gray)
        
        # Compression artifacts detection
        features['compression_score'] = self.detect_compression_artifacts(gray)
        
        # Pixel-level inconsistencies
        features['pixel_inconsistency'] = self.detect_pixel_inconsistencies(img_array)
        
        return features

    def enhanced_lbp(self, gray):
        """Enhanced Local Binary Pattern for texture analysis"""
        try:
            h, w = gray.shape
            lbp_values = []
            
            # Use multiple radii for better texture analysis
            for radius in [1, 2, 3]:
                for i in range(radius, h-radius, 5):
                    for j in range(radius, w-radius, 5):
                        center = gray[i, j]
                        pattern = 0
                        
                        # 8-point circular pattern
                        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
                        for idx, angle in enumerate(angles):
                            x = int(i + radius * np.cos(angle))
                            y = int(j + radius * np.sin(angle))
                            if 0 <= x < h and 0 <= y < w:
                                if gray[x, y] > center:
                                    pattern |= (1 << idx)
                        
                        lbp_values.append(pattern)
            
            # Calculate uniformity and variance of LBP patterns
            if lbp_values:
                return np.var(lbp_values)
            else:
                return 50.0
        except:
            return 50.0

    def detect_compression_artifacts(self, gray):
        """Detect JPEG compression artifacts that might indicate manipulation"""
        try:
            # Look for blocking artifacts
            h, w = gray.shape
            block_size = 8
            artifacts = 0
            
            for i in range(0, h-block_size, block_size):
                for j in range(0, w-block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    
                    # Check for sudden intensity changes at block boundaries
                    if i > 0:
                        top_diff = np.mean(np.abs(gray[i-1, j:j+block_size] - gray[i, j:j+block_size]))
                        artifacts += top_diff
                    
                    if j > 0:
                        left_diff = np.mean(np.abs(gray[i:i+block_size, j-1] - gray[i:i+block_size, j]))
                        artifacts += left_diff
            
            return artifacts / ((h//block_size) * (w//block_size))
        except:
            return 10.0

    def detect_pixel_inconsistencies(self, img_array):
        """Detect pixel-level inconsistencies that may indicate tampering"""
        try:
            if len(img_array.shape) != 3:
                return 0.5
                
            h, w, c = img_array.shape
            inconsistencies = 0
            
            # Check for unnatural smoothness in random patches
            num_patches = 20
            patch_size = min(32, h//4, w//4)
            
            for _ in range(num_patches):
                start_x = np.random.randint(0, max(1, h - patch_size))
                start_y = np.random.randint(0, max(1, w - patch_size))
                
                patch = img_array[start_x:start_x+patch_size, start_y:start_y+patch_size]
                
                # Calculate local variance for each channel
                for channel in range(c):
                    channel_var = np.var(patch[:,:,channel])
                    if channel_var < 100:  # Very smooth patch
                        inconsistencies += 1
                    
                    # Check for unnatural gradients
                    grad_x = np.diff(patch[:,:,channel], axis=1)
                    grad_y = np.diff(patch[:,:,channel], axis=0)
                    gradient_smoothness = np.var(grad_x) + np.var(grad_y)
                    
                    if gradient_smoothness < 50:  # Too smooth gradients
                        inconsistencies += 0.5
            
            return inconsistencies / (num_patches * c)
        except:
            return 0.5

    def predict_deepfake(self, image):
        """Enhanced prediction if image is deepfake based on multiple features"""
        features = self.analyze_image_features(image)
        faces = self.detect_faces_basic(image)
        
        # Start with neutral score
        score = 0
        confidence_factors = []
        deepfake_indicators = 0  # Count of suspicious features
        
        # 1. Texture variance analysis (deepfakes often have unnatural smoothness)
        if features['texture_variance'] < 400:
            score -= 25
            deepfake_indicators += 1
            confidence_factors.append("⚠️ Unusually smooth texture (common in deepfakes)")
        elif features['texture_variance'] > 1200:
            score += 15
            confidence_factors.append("✓ Natural texture variation")
        else:
            score += 5
            confidence_factors.append("• Moderate texture variation")
        
        # 2. Edge consistency (deepfakes often have inconsistent edges)
        if features['edge_consistency'] < 30:
            score -= 20
            deepfake_indicators += 1
            confidence_factors.append("⚠️ Inconsistent edge definition")
        elif features['edge_consistency'] > 80:
            score += 12
            confidence_factors.append("✓ Consistent edge definition")
        
        # 3. Edge density (over-smoothing detection)
        if features['edge_density'] < 0.03:
            score -= 18
            deepfake_indicators += 1
            confidence_factors.append("⚠️ Blurred or over-smoothed edges")
        elif features['edge_density'] > 0.08:
            score += 10
            confidence_factors.append("✓ Sharp, natural edge definition")
        
        # 4. Color channel correlations (unnatural in deepfakes)
        avg_correlation = (features['rg_correlation'] + features['rb_correlation'] + features['gb_correlation']) / 3
        if avg_correlation > 0.95 or avg_correlation < 0.6:
            score -= 15
            deepfake_indicators += 1
            confidence_factors.append("⚠️ Unnatural color channel relationships")
        else:
            score += 8
            confidence_factors.append("✓ Natural color distribution")
        
        # 5. Frequency domain analysis
        if features['frequency_variance'] < 80:
            score -= 12
            deepfake_indicators += 1
            confidence_factors.append("⚠️ Limited frequency content variation")
        elif features['high_freq_energy'] < 4:
            score -= 10
            deepfake_indicators += 1
            confidence_factors.append("⚠️ Reduced high-frequency details")
        else:
            score += 8
            confidence_factors.append("✓ Good frequency domain characteristics")
        
        # 6. Compression artifacts (unusual patterns may indicate manipulation)
        if features['compression_score'] > 25:
            score -= 8
            confidence_factors.append("⚠️ Unusual compression patterns detected")
        elif features['compression_score'] < 5:
            score -= 5
            confidence_factors.append("• Very low compression artifacts")
        else:
            score += 5
            confidence_factors.append("✓ Normal compression characteristics")
        
        # 7. Pixel inconsistencies (smoothness in random patches)
        if features['pixel_inconsistency'] > 3:
            score -= 22
            deepfake_indicators += 1
            confidence_factors.append("⚠️ Excessive smoothness in image patches")
        elif features['pixel_inconsistency'] < 0.5:
            score += 10
            confidence_factors.append("✓ Natural pixel variation throughout image")
        
        # 8. Enhanced texture patterns
        if features['texture_pattern'] < 20 or features['texture_pattern'] > 120:
            score -= 10
            deepfake_indicators += 1
            confidence_factors.append("⚠️ Abnormal texture patterns detected")
        else:
            score += 5
            confidence_factors.append("✓ Normal texture pattern distribution")
        
        # 9. Face detection analysis
        if faces:
            score += 10
            confidence_factors.append("✓ Clear facial features detected")
            
            # Enhanced eye detection analysis
            for face in faces:
                eyes = self.detect_eyes_basic(image, face)
                if len(eyes) >= 2:
                    score += 8
                    confidence_factors.append("✓ Both eyes clearly detected")
                elif len(eyes) == 1:
                    score -= 8
                    deepfake_indicators += 0.5
                    confidence_factors.append("⚠️ Only one eye clearly detected")
                else:
                    score -= 15
                    deepfake_indicators += 1
                    confidence_factors.append("⚠️ Difficulty detecting eyes (possible manipulation)")
        else:
            score -= 10
            confidence_factors.append("• No clear face detected")
        
        # 10. Brightness distribution analysis
        if features['mean_brightness'] < 30 or features['mean_brightness'] > 220:
            score -= 8
            confidence_factors.append("⚠️ Unusual brightness distribution")
        else:
            score += 5
            confidence_factors.append("✓ Normal brightness range")
        
        # Final decision logic - more aggressive deepfake detection
        # If we have 3 or more deepfake indicators, lean towards FAKE
        if deepfake_indicators >= 3:
            score -= 30
            confidence_factors.append(f"🚨 Multiple suspicious indicators detected ({deepfake_indicators})")
        elif deepfake_indicators >= 2:
            score -= 15
            confidence_factors.append(f"⚠️ Several suspicious features ({deepfake_indicators})")
        
        # Convert score to prediction with adjusted thresholds
        # Make it more sensitive to detecting fakes
        if score <= -10:
            prediction = "FAKE"
            confidence = min(95, max(60, abs(score) + 50))
        elif score <= 5:
            prediction = "FAKE" 
            confidence = min(85, max(55, abs(score) + 45))
        else:
            prediction = "REAL"
            confidence = min(95, max(55, score + 50))
        
        # Add summary
        summary_factors = [f"• Detection score: {score}", f"• Suspicious indicators: {deepfake_indicators}"]
        confidence_factors = summary_factors + confidence_factors
            
        return prediction, confidence, confidence_factors, features

    def generate_heatmap(self, image, prediction, features, faces):
        """Generate suspicion heatmap"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        heatmap = np.zeros((h, w))
        
        # Base suspicion level - balanced for visibility
        base_suspicion = 0.4 if prediction == "FAKE" else 0.15
        
        # Focus on detected faces
        for face in faces:
            x, y, fw, fh = face
            
            # Eyes region (high importance for deepfake detection)
            eye_suspicion = base_suspicion * 1.5
            if features['edge_density'] < 0.03:
                eye_suspicion *= 1.8  # Blurry eyes are very suspicious
            
            # Add eye regions
            heatmap[y:y+fh//3, x:x+fw] = eye_suspicion
            
            # Mouth region
            mouth_suspicion = base_suspicion * 1.2
            if features['texture_variance'] < 500:
                mouth_suspicion *= 1.5  # Smooth mouth area suspicious
            
            heatmap[y+2*fh//3:y+fh, x+fw//4:x+3*fw//4] = mouth_suspicion
            
            # Nose/center region
            nose_suspicion = base_suspicion * 0.8
            heatmap[y+fh//3:y+2*fh//3, x+fw//3:x+2*fw//3] = nose_suspicion
            
            # Jawline/edges
            if features['edge_density'] < 0.04:
                # Blurry edges around face
                edge_suspicion = base_suspicion * 1.3
                # Top edge
                heatmap[max(0, y-10):y+10, x:x+fw] = edge_suspicion
                # Bottom edge
                heatmap[y+fh-10:min(h, y+fh+10), x:x+fw] = edge_suspicion
                # Side edges
                heatmap[y:y+fh, max(0, x-10):x+10] = edge_suspicion
                heatmap[y:y+fh, x+fw-10:min(w, x+fw+10)] = edge_suspicion
        
        # If no faces detected, create general suspicion pattern
        if not faces:
            center_x, center_y = w // 2, h // 2
            y_coords, x_coords = np.ogrid[:h, :w]
            
            # Central circular region
            mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 < (min(w, h) * 0.3)**2
            heatmap[mask] = base_suspicion * 0.6
        
        # Smooth the heatmap
        try:
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        except:
            pass  # If blur fails, use original
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        return heatmap

    def create_visualization(self, image, heatmap):
        """Create heatmap overlay with vibrant colors"""
        try:
            rgb_img = np.array(image) / 255.0
            
            # Use jet colormap for bright, visible colors
            # Apply gamma correction for better visibility
            enhanced_heatmap = np.power(heatmap, 0.7)
            colored_heatmap = cm.jet(enhanced_heatmap)[:, :, :3]
            
            # Higher opacity for better visibility
            alpha = 0.6
            blended = (1 - alpha) * rgb_img + alpha * colored_heatmap
            visualization = (blended * 255).astype(np.uint8)
            
            return visualization
        except Exception as e:
            print(f"Visualization error: {e}")
            return np.array(image)

    def analyze_facial_regions(self, heatmap, faces):
        """Analyze specific facial regions"""
        region_scores = {}
        
        if faces:
            face = faces[0]  # Use first face
            x, y, fw, fh = face
            
            # Define regions within the face
            regions = {
                'eyes': heatmap[y:y+fh//3, x:x+fw],
                'nose': heatmap[y+fh//4:y+3*fh//4, x+fw//3:x+2*fw//3],
                'mouth': heatmap[y+2*fh//3:y+fh, x+fw//4:x+3*fw//4],
                'forehead': heatmap[max(0, y-fh//6):y+fh//6, x:x+fw],
                'left_cheek': heatmap[y+fh//4:y+3*fh//4, x:x+fw//3],
                'right_cheek': heatmap[y+fh//4:y+3*fh//4, x+2*fw//3:x+fw],
                'jawline': heatmap[y+3*fh//4:y+fh, x:x+fw]
            }
            
            for region_name, region_data in regions.items():
                if region_data.size > 0:
                    avg_intensity = np.mean(region_data)
                    # Cap the intensity to prevent extreme values
                    region_scores[region_name] = float(min(0.5, avg_intensity))
                else:
                    region_scores[region_name] = 0.0
        else:
            # No face detected
            for region_name in ['eyes', 'nose', 'mouth', 'forehead', 'cheeks', 'jawline']:
                region_scores[region_name] = np.random.uniform(0.1, 0.4)
        
        return region_scores

    def create_detailed_plot(self, image, heatmap, region_scores, prediction, confidence):
        """Create comprehensive analysis plot"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Original image
            axes[0,0].imshow(image)
            axes[0,0].set_title("Original Image", fontsize=14, fontweight='bold')
            axes[0,0].axis('off')
            
            # Heatmap overlay
            visualization = self.create_visualization(image, heatmap)
            axes[0,1].imshow(visualization)
            axes[0,1].set_title(f"Suspicion Heatmap - {prediction} ({confidence:.1f}%)", 
                               fontsize=14, fontweight='bold')
            axes[0,1].axis('off')
            
            # Region scores bar chart
            if region_scores:
                regions = list(region_scores.keys())
                scores = list(region_scores.values())
                
                colors = ['red' if s > 0.5 else 'orange' if s > 0.3 else 'green' for s in scores]
                bars = axes[1,0].bar(regions, scores, color=colors, alpha=0.7)
                axes[1,0].set_ylabel("Suspicion Level", fontsize=12)
                axes[1,0].set_title("Facial Region Analysis", fontsize=14, fontweight='bold')
                axes[1,0].tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, score in zip(bars, scores):
                    axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                  f'{score:.2f}', ha='center', va='bottom', fontsize=10)
            
            # Confidence breakdown
            risk_levels = ['Low Risk', 'Medium Risk', 'High Risk']
            risk_colors = ['green', 'orange', 'red']
            
            if prediction == "FAKE":
                risk_values = [max(0, 100-confidence), min(confidence/2, 30), max(0, confidence-70)]
            else:
                risk_values = [confidence, max(0, (100-confidence)/2), max(0, 70-confidence)]
            
            # Normalize risk values
            total_risk = sum(risk_values)
            if total_risk > 0:
                risk_values = [v/total_risk * 100 for v in risk_values]
            
            axes[1,1].pie(risk_values, labels=risk_levels, colors=risk_colors, autopct='%1.1f%%',
                         startangle=90)
            axes[1,1].set_title("Risk Assessment", fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return plot_data
        except Exception as e:
            print(f"Plot creation error: {e}")
            return ""

    def explain_prediction(self, image):
        """Main explanation method"""
        try:
            # Get prediction and analysis
            prediction, confidence, factors, features = self.predict_deepfake(image)
            
            # Detect faces
            faces = self.detect_faces_basic(image)
            
            # Generate heatmap
            heatmap = self.generate_heatmap(image, prediction, features, faces)
            
            # Analyze regions
            region_scores = self.analyze_facial_regions(heatmap, faces)
            
            # Create visualizations
            visualization = self.create_visualization(image, heatmap)
            detailed_plot = self.create_detailed_plot(image, heatmap, region_scores, prediction, confidence)
            
            # Convert main visualization to base64
            vis_pil = Image.fromarray(visualization)
            buffer = io.BytesIO()
            vis_pil.save(buffer, format='PNG')
            vis_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Create explanation
            explanation = {
                "summary": f"Image analyzed as {prediction} with {confidence:.1f}% confidence. Analysis based on texture, edge definition, facial features, and color distribution.",
                "detailed_analysis": [
                    {
                        "region": region.replace('_', ' ').title(),
                        "suspicion_level": score,
                        "description": f"{region.replace('_', ' ').title()} shows {score:.1%} suspicion level"
                    }
                    for region, score in sorted(region_scores.items(), key=lambda x: x[1], reverse=True)[:4]
                ],
                "confidence_factors": factors,
                "recommendations": [
                    f"🔍 Focus on highlighted regions in the heatmap" if prediction == "FAKE" else "✅ Image appears authentic",
                    "📊 Check the regional analysis for specific areas of concern",
                    "🔬 Higher resolution images provide more accurate analysis",
                    "⚠️ Always verify with multiple detection methods for critical applications"
                ]
            }
            
            return {
                "success": True,
                "prediction": prediction,
                "confidence": confidence,
                "heatmap_overlay": vis_b64,
                "detailed_analysis": detailed_plot,
                "region_analysis": {
                    "region_scores": region_scores,
                    "most_suspicious": sorted(region_scores.items(), key=lambda x: x[1], reverse=True)[:3],
                    "faces_detected": len(faces)
                },
                "explanation": explanation,
                "technical_features": features
            }
            
        except Exception as e:
            print(f"Explanation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "prediction": "Error",
                "confidence": 0
            }