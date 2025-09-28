# app.py - FIXED VERSION
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Pancreatic Cancer AI Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5D5D5D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-box {
        border: 2px dashed #2E86AB;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        background-color: #F8F9FA;
        margin: 20px 0;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 25px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .normal-card {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
    }
    .tumor-card {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
    }
    .confidence-bar {
        height: 25px;
        border-radius: 12px;
        margin: 10px 0;
        background: rgba(255,255,255,0.3);
        overflow: hidden;
    }
    .stat-box {
        background: rgba(255,255,255,0.2);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
    }
    .recommendation-box {
        background: rgba(255,255,255,0.9);
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        color: #333;
        border-left: 5px solid #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

class MedicalAIDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.model_parts = [
            "quantum-cnn-streamlit/model_part_1.pth", 
            "quantum-cnn-streamlit/model_part_2.pth", 
            "quantum-cnn-streamlit/model_part_3.pth"
        ]

    def check_model_parts(self):
        """Check if all model parts are available"""
        status = {}
        for part in self.model_parts:
            if os.path.exists(part):
                size = os.path.getsize(part) / (1024*1024)
                status[part] = {"exists": True, "size": f"{size:.1f} MB"}
            else:
                status[part] = {"exists": False, "size": "Missing"}
        return status

    def ensure_rgb_image(self, image):
        """Ensure image has 3 channels (RGB)"""
        if image.mode != 'RGB':
            return image.convert('RGB')
        return image

    def combine_and_load_model(self):
        """Combine model parts and load the complete model"""
        try:
            # Check all parts first
            status = self.check_model_parts()
            missing_parts = [part for part in self.model_parts if not status[part]["exists"]]
            
            if missing_parts:
                st.error(f"Missing model parts: {missing_parts}")
                return False
            
            st.info("🔗 Combining model parts...")
            
            # Combine all parts
            combined_state_dict = {}
            for part_file in self.model_parts:
                part_dict = torch.load(part_file, map_location='cpu')
                combined_state_dict.update(part_dict)
            
            # Create model architecture
            self.model = SimpleCNN().to(self.device)
            self.model.load_state_dict(combined_state_dict)
            self.model.eval()
            
            st.success("🎉 Model loaded successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False

    def predict_image(self, image):
        """Make prediction using combined model"""
        if self.model is None:
            if not self.combine_and_load_model():
                return None
        
        try:
            # Ensure image is RGB
            image_rgb = self.ensure_rgb_image(image)
            
            # Image preprocessing
            image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # Model prediction
            with torch.no_grad():
                output = self.model(image_tensor)
                probability = torch.sigmoid(output).item()
            
            # Prepare results
            prediction = "Pancreatic Tumor" if probability > 0.5 else "Normal"
            confidence = probability if probability > 0.5 else 1 - probability
            
            return {
                'prediction': prediction,
                'confidence': confidence * 100,
                'probability': probability,
                'normal_probability': (1 - probability) * 100,
                'tumor_probability': probability * 100,
                'is_high_confidence': confidence > 0.7
            }
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return None

    def create_medical_report(self, image, prediction_result, filename):
        """Create comprehensive medical report"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Medical Imaging Analysis Report', fontsize=16, fontweight='bold')
        
        # Original image
        display_image = self.ensure_rgb_image(image)
        ax1.imshow(display_image)
        ax1.set_title('Uploaded Medical Image', fontweight='bold')
        ax1.axis('off')
        
        # Probability chart
        labels = ['Normal Tissue', 'Pancreatic Tumor']
        probabilities = [prediction_result['normal_probability'], prediction_result['tumor_probability']]
        colors = ['#56ab2f', '#ff416c']
        
        bars = ax2.bar(labels, probabilities, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Confidence (%)', fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.set_title('AI Analysis Results', fontweight='bold')
        
        # Add value labels
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Confidence indicator
        confidence_level = prediction_result['confidence']
        ax3.barh(['Model Confidence'], [confidence_level], color='#2E86AB', alpha=0.7)
        ax3.set_xlim(0, 100)
        ax3.set_xlabel('Confidence Level (%)', fontweight='bold')
        ax3.set_title('Analysis Reliability', fontweight='bold')
        ax3.text(confidence_level/2, 0, f'{confidence_level:.1f}%', 
                ha='center', va='center', color='white', fontweight='bold')
        
        # Risk assessment
        risk_level = "HIGH" if prediction_result['prediction'] == "Pancreatic Tumor" else "LOW"
        risk_color = '#ff416c' if risk_level == "HIGH" else '#56ab2f'
        ax4.text(0.5, 0.6, risk_level, fontsize=40, ha='center', va='center', 
                color=risk_color, fontweight='bold')
        ax4.text(0.5, 0.3, 'RISK LEVEL', fontsize=15, ha='center', va='center')
        ax4.set_title('Clinical Risk Assessment', fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        return buf

# Model Architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.resnet(x)

def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Pancreatic Cancer AI Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Deep Learning for Medical Image Analysis</p>', unsafe_allow_html=True)
    
    # Initialize detector - FIXED: Using correct class name
    detector = MedicalAIDetector()
    
    # Model status check
    parts_status = detector.check_model_parts()
    all_parts_available = all(status["exists"] for status in parts_status.values())
    
    if not all_parts_available:
        st.error("❌ Model files not found. Please ensure all model parts are available.")
        st.write("Required model parts:")
        for part, status in parts_status.items():
            st.write(f"- {part}: {'✅ Found' if status['exists'] else '❌ Missing'}")
        return
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="upload-box">
            <div style="font-size: 2rem; margin-bottom: 10px;">📤</div>
            <h3>Upload Medical Image</h3>
            <p>CT Scan • MRI • Medical Imaging</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Select medical image for analysis",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Supported formats: PNG, JPG, JPEG, BMP"
        )
        
        if uploaded_file is not None:
            # Process uploaded image
            image = Image.open(uploaded_file)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image_rgb = image.convert('RGB')
                st.info(f"🔄 Image converted from {image.mode} to RGB")
            else:
                image_rgb = image
            
            st.image(image_rgb, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)
            
            # Analysis button
            if st.button("🧠 Analyze Medical Image", type="primary", use_container_width=True):
                with st.spinner("🔍 Analyzing image with AI model..."):
                    result = detector.predict_image(image)
                    
                    if result:
                        with col2:
                            # Display results
                            st.markdown("""
                            <div style="text-align: center; margin-bottom: 20px;">
                                <div style="font-size: 2rem; margin-bottom: 10px;">📊</div>
                                <h3>Analysis Results</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Result card
                            card_class = "tumor-card" if result['prediction'] == "Pancreatic Tumor" else "normal-card"
                            result_icon = "⚠️" if result['prediction'] == "Pancreatic Tumor" else "✅"
                            
                            st.markdown(f"""
                            <div class="result-card {card_class}">
                                <h2>{result_icon} {result['prediction']}</h2>
                                <div class="stat-box">
                                    <h3>AI Confidence: {result['confidence']:.1f}%</h3>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Probability distribution
                            st.write("### Confidence Distribution")
                            
                            col_prob1, col_prob2 = st.columns(2)
                            with col_prob1:
                                st.metric("Normal Tissue", f"{result['normal_probability']:.1f}%")
                                st.progress(result['normal_probability']/100)
                            
                            with col_prob2:
                                st.metric("Pancreatic Tumor", f"{result['tumor_probability']:.1f}%")
                                st.progress(result['tumor_probability']/100)
                            
                            # Medical recommendations
                            st.markdown("### 🩺 Clinical Recommendations")
                            
                            if result['prediction'] == "Pancreatic Tumor":
                                st.markdown("""
                                <div class="recommendation-box">
                                    <h4>🚨 Urgent Action Required:</h4>
                                    <ul>
                                        <li>Consult oncologist immediately</li>
                                        <li>Schedule follow-up CT/MRI</li>
                                        <li>Blood tests (CA19-9, CEA)</li>
                                        <li>Multidisciplinary team review</li>
                                        <li>Consider biopsy confirmation</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div class="recommendation-box">
                                    <h4>✅ Routine Monitoring:</h4>
                                    <ul>
                                        <li>Regular follow-up in 6-12 months</li>
                                        <li>Maintain healthy lifestyle</li>
                                        <li>Monitor for symptoms</li>
                                        <li>Annual health check-ups</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Generate and display report
                            report_buf = detector.create_medical_report(image, result, uploaded_file.name)
                            st.image(report_buf, caption="Comprehensive Medical Analysis Report", use_container_width=True)
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Full Medical Report",
                                data=report_buf.getvalue(),
                                file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                    else:
                        st.error("❌ Analysis failed. Please try with a different image.")
    
    # Features section when no file is uploaded
    if uploaded_file is None:
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 40px 20px;">
                <div style="font-size: 2rem; margin-bottom: 10px;">🔍</div>
                <h3>How It Works</h3>
            </div>
            """, unsafe_allow_html=True)
            
            features = [
                {"icon": "📷", "title": "Upload CT Scan", "desc": "Upload abdominal CT scan images for analysis"},
                {"icon": "🧠", "title": "AI Analysis", "desc": "Deep learning model analyzes image patterns"},
                {"icon": "📊", "title": "Get Results", "desc": "Receive detailed probability analysis"},
                {"icon": "🩺", "title": "Clinical Guidance", "desc": "Evidence-based medical recommendations"}
            ]
            
            for feature in features:
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #2E86AB;">
                    <div style="font-size: 1.5rem; margin-bottom: 5px;">{feature['icon']}</div>
                    <strong>{feature['title']}</strong><br>
                    <small>{feature['desc']}</small>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>🔬 Pancreatic Cancer AI Detection System</strong></p>
    <p><small>For research and educational purposes. Always consult healthcare professionals for medical diagnosis.</small></p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
