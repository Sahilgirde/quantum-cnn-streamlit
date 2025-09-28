# app.py - FIXED IMAGE CHANNELS ISSUE
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
    page_title="Pancreatic Cancer Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    .normal-result {
        border-color: #28a745;
        background-color: #d4edda;
    }
    .tumor-result {
        border-color: #dc3545;
        background-color: #f8d7da;
    }
    .probability-bar {
        height: 30px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .success-banner {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .part-status {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        text-align: center;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class ModelPartsCombiner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.model_parts = ["quantum-cnn-streamlit/model_part_1.pth", "quantum-cnn-streamlit/model_part_2.pth", "quantum-cnn-streamlit/model_part_3.pth"]

    
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
            st.info(f"🔄 Converting image from {image.mode} to RGB")
            return image.convert('RGB')
        return image
    
    def combine_and_load_model(self):
        """Combine model parts and load the complete model"""
        try:
            # Check all parts first
            status = self.check_model_parts()
            missing_parts = [part for part in self.model_parts if not status[part]["exists"]]
            
            if missing_parts:
                st.error(f"❌ Missing model parts: {missing_parts}")
                return False
            
            st.info("🔗 Combining model parts...")
            
            # Combine all parts
            combined_state_dict = {}
            for part_file in self.model_parts:
                part_dict = torch.load(part_file, map_location='cpu')
                combined_state_dict.update(part_dict)
                st.success(f"✅ Loaded: {part_file} ({status[part_file]['size']})")
            
            # Create model architecture - EXACTLY matching the saved weights
            self.model = SimpleCNN().to(self.device)
            
            # Load combined weights
            self.model.load_state_dict(combined_state_dict)
            self.model.eval()
            
            st.success("🎉 Model parts successfully combined and loaded!")
            return True
            
        except Exception as e:
            st.error(f"❌ Error combining model parts: {e}")
            return False
    
    def predict_image(self, image):
        """Make prediction using combined model"""
        if self.model is None:
            if not self.combine_and_load_model():
                return None
        
        try:
            # Ensure image is RGB (3 channels)
            image_rgb = self.ensure_rgb_image(image)
            
            # Debug: Show image info
            st.info(f"📷 Image mode: {image.mode} → {image_rgb.mode}, Size: {image_rgb.size}")
            
            # Image preprocessing
            image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # Debug: Show tensor shape
            st.info(f"🔢 Tensor shape: {image_tensor.shape}")
            
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
                'model_type': 'Trained CNN Model'
            }
            
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            # Show detailed tensor info
            try:
                if 'image_tensor' in locals():
                    st.info(f"📊 Tensor details - Shape: {image_tensor.shape}, Min: {image_tensor.min():.3f}, Max: {image_tensor.max():.3f}")
            except:
                pass
            return None
    
    def create_report_image(self, image, prediction_result):
        """Create analysis report image"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image (convert to RGB for display)
        display_image = self.ensure_rgb_image(image)
        ax1.imshow(display_image)
        ax1.set_title('Uploaded CT Scan', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Probability chart
        labels = ['Normal', 'Pancreatic Tumor']
        probabilities = [prediction_result['normal_probability'], prediction_result['tumor_probability']]
        colors = ['#28a745', '#dc3545']
        
        bars = ax2.bar(labels, probabilities, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Probability (%)', fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.set_title('Model Predictions', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        return buf

# CORRECT MODEL ARCHITECTURE - Matching your saved weights
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Using ResNet18 with custom fc layer (EXACTLY as per your saved weights)
        self.resnet = models.resnet18(weights=None)
        
        # This matches the saved weights: "resnet.fc.0.weight", "resnet.fc.0.bias", etc.
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 256),           # resnet.fc.0
            nn.BatchNorm1d(256),           # resnet.fc.1
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),           # resnet.fc.4
            nn.BatchNorm1d(128),           # resnet.fc.5
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)              # resnet.fc.8
        )
    
    def forward(self, x):
        return self.resnet(x)

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Pancreatic Cancer Detection System</h1>', 
                unsafe_allow_html=True)
    st.markdown("### 🔗 **Trained CNN Model** - Medical AI Diagnosis")
    
    # Image format warning
    st.markdown("""
    <div class="warning-box">
    📷 <strong>Image Requirements:</strong> Model expects 3-channel RGB images. 
    Grayscale/BW images will be automatically converted to RGB.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize detector
    detector = ModelPartsCombiner()
    
    # Check model parts status
    st.subheader("📁 Model Parts Status")
    parts_status = detector.check_model_parts()
    
    col1, col2, col3 = st.columns(3)
    all_parts_available = True
    
    for i, (part, status) in enumerate(parts_status.items()):
        if status["exists"]:
            bg_color = "#d4edda"
            if i == 0: 
                col1.markdown(f'<div class="part-status" style="background-color: {bg_color}">✅ {part}<br>{status["size"]}</div>', unsafe_allow_html=True)
            elif i == 1: 
                col2.markdown(f'<div class="part-status" style="background-color: {bg_color}">✅ {part}<br>{status["size"]}</div>', unsafe_allow_html=True)
            else: 
                col3.markdown(f'<div class="part-status" style="background-color: {bg_color}">✅ {part}<br>{status["size"]}</div>', unsafe_allow_html=True)
        else:
            all_parts_available = False
            if i == 0: 
                col1.error(f"❌ {part}\nMissing")
            elif i == 1: 
                col2.error(f"❌ {part}\nMissing")
            else: 
                col3.error(f"❌ {part}\nMissing")
    
    if all_parts_available:
        st.markdown("""
        <div class="success-banner">
        🎉 <strong>ALL MODEL PARTS READY!</strong> | 
        🔗 <strong>3 Parts Combined</strong> | 
        🧠 <strong>Trained CNN Active</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ Please ensure all 3 model parts are available in the same directory")
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=100)
        st.title("Navigation")
        
        if all_parts_available:
            st.markdown("""
            <div class="success-banner">
            🔬 <strong>Model Ready</strong><br>
            ✅ 3 Parts Integrated
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ **Model Parts Missing**")
        
        st.info("""
        **Instructions:**
        1. Upload CT scan image (RGB/Grayscale)
        2. Image auto-converted to RGB
        3. Click 'Analyze with Trained Model'
        4. Get AI diagnosis
        """)
        
        st.markdown("---")
        st.subheader("Image Requirements")
        st.write("""
        **Supported Formats:**
        - RGB Images (3 channels)
        - Grayscale (auto-converted)
        - PNG, JPG, JPEG, BMP
        
        **Model Input:**
        - Size: 224×224 pixels
        - Channels: 3 (RGB)
        - Normalized: ImageNet stats
        """)
        
        st.markdown("---")
        st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload CT Scan Image")
        
        uploaded_file = st.file_uploader(
            "Choose a pancreatic CT scan image",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Supported formats: PNG, JPG, JPEG, BMP - RGB or Grayscale"
        )
        
        if uploaded_file is not None and all_parts_available:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            # Show image info
            st.info(f"📷 Original Image: {image.mode} mode, Size: {image.size}")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image_rgb = image.convert('RGB')
                st.success(f"🔄 Converted to RGB: {image_rgb.mode} mode")
                display_image = image_rgb
            else:
                display_image = image
            
            st.image(display_image, caption="Uploaded CT Scan (RGB)", use_column_width=True)
            
            # Analysis button
            if st.button("🔬 Analyze with Trained Model", type="primary", use_container_width=True):
                with st.spinner("Processing image and analyzing..."):
                    # Get prediction from combined model
                    result = detector.predict_image(image)  # Use original image, conversion happens inside
                    
                    if result:
                        # Display results
                        with col2:
                            st.subheader("📊 Medical Analysis Results")
                            
                            # Result box
                            result_class = "tumor-result" if result['prediction'] == "Pancreatic Tumor" else "normal-result"
                            result_icon = "⚠️" if result['prediction'] == "Pancreatic Tumor" else "✅"
                            
                            st.markdown(f"""
                            <div class="result-box {result_class}">
                                <h3>{result_icon} {result['prediction']}</h3>
                                <p><strong>Model Confidence:</strong> {result['confidence']:.1f}%</p>
                                <p><strong>Model Type:</strong> {result['model_type']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Probability bars
                            st.write("**Probability Distribution:**")
                            
                            col21, col22 = st.columns(2)
                            with col21:
                                st.write("Normal:")
                                st.progress(result['normal_probability']/100)
                                st.write(f"{result['normal_probability']:.1f}%")
                            
                            with col22:
                                st.write("Pancreatic Tumor:")
                                st.progress(result['tumor_probability']/100)
                                st.write(f"{result['tumor_probability']:.1f}%")
                            
                            # Recommendation
                            if result['prediction'] == "Pancreatic Tumor":
                                st.error("""
                                **🚨 Medical Recommendation:** 
                                - Consult oncologist immediately
                                - Further diagnostic tests required
                                - Multidisciplinary team evaluation
                                - Urgent treatment planning
                                """)
                            else:
                                st.success("""
                                **✅ Medical Recommendation:** 
                                - Routine follow-up in 6-12 months
                                - Continue healthy lifestyle
                                - Regular health check-ups
                                - No immediate concerns
                                """)
                            
                            # Generate and display report image
                            report_buf = detector.create_report_image(image, result)
                            st.image(report_buf, caption="Medical Analysis Report", use_column_width=True)
                            
                            # Download report
                            st.download_button(
                                label="📥 Download Analysis Report",
                                data=report_buf.getvalue(),
                                file_name=f"medical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                    else:
                        st.error("❌ Prediction failed. Please check the image format.")
    
    # Info section if no file uploaded
    if uploaded_file is None and all_parts_available:
        with col2:
            st.subheader("ℹ️ Image Processing Info")
            
            st.info("""
            **Automatic Image Processing:**
            
            🔄 **Format Conversion:**
            - Grayscale → RGB (3 channels)
            - Black & White → RGB  
            - RGBA → RGB (alpha removed)
            
            📊 **Preprocessing Steps:**
            1. Resize to 224×224 pixels
            2. Convert to RGB if needed
            3. Normalize with ImageNet statistics
            4. Convert to PyTorch tensor
            
            ✅ **Supported Modes:**
            - RGB (3 channels)
            - L (Grayscale)
            - LA (Grayscale + Alpha)
            - RGBA (RGB + Alpha)
            """)

# Footer
st.markdown("---")
footer = """
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🧬 <strong>Pancreatic Cancer Detection System</strong> | Trained CNN Model</p>
    <p><small>⚠️ Disclaimer: This tool uses a trained AI model for research purposes. Always consult healthcare professionals for medical diagnosis.</small></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

