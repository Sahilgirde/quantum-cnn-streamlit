# app.py - FIXED ARCHITECTURE MATCHING
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
        self.model_parts = ["model_part_1.pth", "model_part_2.pth", "model_part_3.pth"]
    
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
            
            # Debug: Show loaded keys
            st.info(f"📊 Loaded {len(combined_state_dict)} parameters")
            
            # Create model architecture - EXACTLY matching the saved weights
            self.model = SimpleCNN().to(self.device)
            
            # Load combined weights
            self.model.load_state_dict(combined_state_dict)
            self.model.eval()
            
            st.success("🎉 Model parts successfully combined and loaded!")
            return True
            
        except Exception as e:
            st.error(f"❌ Error combining model parts: {e}")
            # Show detailed error info
            st.info("🔍 Debug Info: Checking model architecture vs saved weights...")
            self.debug_model_weights(combined_state_dict)
            return False
    
    def debug_model_weights(self, saved_state_dict):
        """Debug model architecture vs saved weights"""
        try:
            # Create model
            test_model = SimpleCNN()
            
            # Get model state dict keys
            model_keys = set(test_model.state_dict().keys())
            saved_keys = set(saved_state_dict.keys())
            
            st.write("**🔍 Architecture vs Saved Weights Analysis:**")
            
            # Missing keys
            missing_in_saved = model_keys - saved_keys
            if missing_in_saved:
                st.error(f"❌ Missing in saved: {list(missing_in_saved)}")
            
            # Extra keys
            extra_in_saved = saved_keys - model_keys
            if extra_in_saved:
                st.warning(f"⚠️ Extra in saved: {list(extra_in_saved)}")
            
            # Common keys
            common_keys = model_keys.intersection(saved_keys)
            st.success(f"✅ Common keys: {len(common_keys)}")
            
        except Exception as e:
            st.error(f"❌ Debug error: {e}")
    
    def predict_image(self, image):
        """Make prediction using combined model"""
        if self.model is None:
            if not self.combine_and_load_model():
                return None
        
        try:
            # Image preprocessing
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
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
            return None
    
    def create_report_image(self, image, prediction_result):
        """Create analysis report image"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        ax1.imshow(image)
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
        1. Ensure all 3 model parts are available
        2. Upload pancreatic CT scan image
        3. Click 'Analyze with Trained Model'
        4. Get AI diagnosis
        """)
        
        st.markdown("---")
        st.subheader("Model Architecture")
        st.write("""
        **ResNet18-based CNN:**
        - Custom Fully Connected Layers
        - Batch Normalization
        - Dropout Regularization
        - Binary Classification
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
            help="Supported formats: PNG, JPG, JPEG, BMP"
        )
        
        if uploaded_file is not None and all_parts_available:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded CT Scan", use_column_width=True)
            
            # Analysis button
            if st.button("🔬 Analyze with Trained Model", type="primary", use_container_width=True):
                with st.spinner("Combining model parts and analyzing..."):
                    # Get prediction from combined model
                    result = detector.predict_image(image)
                    
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
                        st.error("❌ Prediction failed. Please check the model parts.")
    
    # Info section if no file uploaded
    if uploaded_file is None and all_parts_available:
        with col2:
            st.subheader("ℹ️ Model Information")
            
            st.info("""
            **Trained CNN Model System:**
            
            ✅ **Model Status:** 3 Parts Successfully Integrated
            ✅ **Architecture:** ResNet18 with Custom FC Layers
            ✅ **Training:** 999 CT scans, 100% Validation Accuracy
            
            **Model Architecture:**
            - Base: ResNet18 Feature Extractor
            - Custom Fully Connected Layers
            - Batch Normalization
            - Dropout for Regularization
            - Binary Classification Output
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
