import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
from PIL import Image

# --- Page Configuration ---
# This must be the first Streamlit command
st.set_page_config(
    page_title="Retinal OCT Analyzer",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Import Recommendations ---
# Try to import from recommendation.py, provide placeholders if it fails
try:
    from recommendation import cnv, dme, drusen, normal
except ImportError:
    st.warning("`recommendation.py` not found. Using placeholder text.")
    cnv = "### 🩺 Choroidal Neovascularization (CNV)\n\n**Description:** CNV is the growth of abnormal blood vessels under the retina, which can leak fluid and blood, leading to rapid and severe vision loss. It is a common sign of 'wet' age-related macular degeneration (AMD).\n\n**Common Treatment:** Anti-VEGF injections are the standard of care to stop vessel growth and reduce leakage."
    dme = "### 🩺 Diabetic Macular Edema (DME)\n\n**Description:** DME is a complication of diabetes caused by fluid accumulation in the macula. This leads to swelling and blurred, washed-out vision.\n\n**Common Treatment:** Treatments include Anti-VEGF injections, laser therapy, and corticosteroid implants to reduce swelling."
    drusen = "### 🟡 Drusen\n\n**Description:** Drusen are small, yellow deposits of lipids and proteins that build up under the retina. While a few small drusen are normal with aging, large or numerous drusen are a key sign of early-stage age-related macular degeneration (AMD).\n\n**Common Treatment:** No specific treatment for drusen, but monitoring is crucial. Lifestyle changes (e.g., specific vitamins (AREDS2), quitting smoking, healthy diet) may slow the progression to advanced AMD."
    normal = "### ✅ Normal Retina\n\n**Description:** The scan shows a healthy retina with a distinct foveal contour and no signs of fluid, swelling, or abnormal deposits. All retinal layers appear normal.\n\n**Action:** No immediate action is required. Continue with regular eye check-ups as recommended by your specialist."

# --- Class & Recommendation Dictionaries ---
CLASS_NAMES = {
    0: 'CNV',
    1: 'DME',
    2: 'DRUSEN',
    3: 'NORMAL'
}

RECOMMENDATIONS = {
    'CNV': cnv,
    'DME': dme,
    'DRUSEN': drusen,
    'NORMAL': normal
}

# --- Custom CSS for Appeal ---
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        /* background-color: #f0f2f6; */ /* Optional: Light gray background */
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #004d40; /* Dark teal */
        color: white;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: white;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0f2f1; /* Light teal text for sidebar info */
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #00796b; /* Teal */
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #004d40; /* Darker teal on hover */
    }
    
    /* Result box styling */
    .result-box-normal {
        background-color: #e0f2f1; /* Light green */
        border-left: 6px solid #4CAF50; /* Green border */
        padding: 15px 20px;
        border-radius: 5px;
        color: #000000; 
    }
    .result-box-abnormal {
        background-color: #ffebee; /* Light red */
        border-left: 6px solid #f44336; /* Red border */
        padding: 15px 20px;
        border-radius: 5px;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Loading (Cached) ---
@st.cache_resource  # Caches the loaded model for performance
def load_my_model():
    """
    Loads the trained Keras model.
    """
    try:
        model = tf.keras.models.load_model("Trained_Eye_disease_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model 'Trained_Model.keras': {e}", icon="🚨")
        st.error("Please ensure the model file is in the same directory as app.py")
        return None

model = load_my_model()

# --- Prediction Function (Updated) ---
def model_prediction(image_buffer, model):
    """
    Takes an image buffer and a model, returns prediction and confidence.
    """
    if model is None:
        return None, None
    
    try:
        # Open and resize the image using PIL
        img = Image.open(image_buffer).resize((224, 224))
        
        # Convert to array
        x = tf.keras.utils.img_to_array(img)
        
        # Handle different image channels (e.g., grayscale, RGBA)
        if x.ndim == 2:  # Grayscale
            x = np.stack((x,) * 3, axis=-1)
        elif x.shape[-1] == 4:  # RGBA (drop alpha channel)
            x = x[..., :3]
        elif x.shape[-1] == 1: # Grayscale with 1 channel
             x = np.concatenate([x, x, x], axis=-1)
             
        # Ensure it's 3 channels for the model
        if x.shape[-1] != 3:
            st.error(f"Uploaded image has {x.shape[-1]} channels. Please upload an RGB image.", icon="🖼️")
            return None, None

        # Preprocess the image
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Make prediction
        predictions = model.predict(x)
        predicted_index = np.argmax(predictions)
        confidence = np.max(predictions)
        
        class_label = CLASS_NAMES.get(predicted_index, "Unknown")
        return class_label, confidence

    except Exception as e:
        st.error(f"An error occurred during image processing: {e}", icon="⚙️")
        return None, None

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("👁️ Retinal OCT Analyzer")
    st.markdown("Automated detection of retinal diseases using AI.")
    st.markdown("---")
    
    app_mode = st.selectbox("Navigation", 
                            ["Home", "Disease Identification", "About the Project"])
    
    st.markdown("---")
    st.markdown("This tool is for informational purposes and is not a substitute for professional medical advice.")

# --- Page 1: Home ---
if app_mode == "Home":
    st.title("AI-Powered Retinal Disease Detection")
    st.markdown("### Welcome to the Retinal OCT Analysis Platform. Upload your OCT scans for an instant, AI-driven analysis.")
    st.markdown("---")

    with st.container():
        col1, col2 = st.columns([2, 1.5])
        with col1:
            st.markdown("""
            **Optical Coherence Tomography (OCT)** is a powerful, non-invasive imaging technique that provides high-resolution, cross-sectional images of the retina. It's a cornerstone of modern ophthalmology for detecting and monitoring eye conditions that can lead to vision loss.
            
            This platform leverages a state-of-the-art Deep Learning model to analyze OCT scans, providing rapid and accurate classification into four key categories:
            
            -   **✅ NORMAL:** A healthy retina with no signs of disease.
            -   **💧 CNV (Choroidal Neovascularization):** A key sign of "wet" age-related macular degeneration.
            -   **💧 DME (Diabetic Macular Edema):** A common complication of diabetes.
            -   **🟡 DRUSEN:** Deposits that are an early sign of age-related macular degeneration.
            """)
            st.info("**Get Started:**\n1. Navigate to the **Disease Identification** page.\n2. Upload your OCT scan.\n3. Receive an instant analysis.")
        
        with col2:
            # IMPORTANT: Replace 'home_image.png' with a path to an actual image
            try:
                st.image("home_image.png", caption="Example of OCT scan categories", use_container_width=True)
            except:
                st.info("Add an image named 'home_image.png' to this folder to display it here.")

    st.markdown("---")
    
    st.header("Understanding the Detections")
    tab1, tab2, tab3, tab4 = st.tabs(["👁️ NORMAL", "💧 CNV", "💧 DME", "🟡 DRUSEN"])

    with tab1:
        st.subheader("Normal Retina")
        st.markdown("A normal retina scan shows a preserved foveal contour and a clear distinction between retinal layers, with no signs of fluid, edema, or deposits.")
        # Optional: st.image("normal_example.png", use_column_width=True)

    with tab2:
        st.subheader("Choroidal Neovascularization (CNV)")
        st.markdown("CNV is the growth of abnormal blood vessels from the choroid layer into the subretinal space. On an OCT, this often appears as a neovascular membrane with associated subretinal fluid, disrupting the normal retinal structure.")
        # Optional: st.image("cnv_example.png", use_column_width=True)

    with tab3:
        st.subheader("Diabetic Macular Edema (DME)")
        st.markdown("DME is characterized by retinal thickening and the accumulation of intraretinal fluid (arrows) within the macula, often appearing as cyst-like spaces. This is a common complication of diabetic retinopathy.")
        # Optional: st.image("dme_example.png", use_column_width=True)

    with tab4:
        st.subheader("Drusen")
        st.markdown("Drusen are small, yellowish deposits of extracellular material that accumulate under the retina. On an OCT, they appear as small bumps or elevations (arrowheads) of the retinal pigment epithelium (RPE).")
        # Optional: st.image("drusen_example.png", use_column_width=True)

# --- Page 2: Disease Identification (Prediction) ---
elif app_mode == "Disease Identification":
    st.title("Retinal Scan Analysis")
    st.markdown("Upload your OCT scan below to get an AI-powered analysis. This tool supports `PNG`, `JPG`, and `JPEG` formats.")
    st.markdown("---")

    test_image = st.file_uploader("Upload an OCT Image:", type=["png", "jpg", "jpeg"])
    
    if test_image is not None:
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.image(test_image, caption="Uploaded OCT Scan", use_container_width=True)
        
        with col2:
            st.info("Click 'Analyze Scan' to process the image.")
            
            if st.button("Analyze Scan"):
                if model is not None:
                    with st.spinner("Analyzing... Please wait."):
                        
                        # Call the prediction function
                        class_label, confidence = model_prediction(test_image, model)
                        
                        if class_label:
                            st.markdown("---")
                            st.subheader("Analysis Result")
                            
                            # Display result with custom styling
                            if class_label == 'NORMAL':
                                st.markdown(f'<div class="result-box-normal"><strong>Diagnosis: {class_label}</strong><br>Confidence: {confidence*100:.2f}%</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="result-box-abnormal"><strong>Diagnosis: {class_label}</strong><br>Confidence: {confidence*100:.2f}%</div>', unsafe_allow_html=True)
                            
                            # Display the detailed recommendation in an expander
                            with st.expander("Learn more about this condition", expanded=True):
                                st.markdown(RECOMMENDATIONS[class_label], unsafe_allow_html=True)
                else:
                    st.error("Model is not loaded. Cannot perform analysis.", icon="🚨")
    else:
        st.info("Please upload an image to begin analysis.", icon="⬆️")

# --- Page 3: About the Project ---
elif app_mode == "About the Project":
    st.title("About the Project and Dataset")
    st.markdown("This tool is a demonstration of a deep learning model trained on a public dataset of retinal OCT images.")
    st.markdown("---")

    st.subheader("About the Dataset")
    st.markdown(
        """
        The dataset used for training consists of **84,495 high-resolution OCT images** (JPEG) organized into four categories: NORMAL, CNV, DME, and DRUSEN.

        -   **Source:** The images were selected from retrospective cohorts of adult patients from multiple medical centers, including the Shiley Eye Institute (UC San Diego) and the California Retinal Research Foundation, among others, between 2013 and 2017.
        -   **Data Quality:** Each image underwent a rigorous, multi-tiered grading system by trained graders, ophthalmologists, and
            senior retinal specialists to verify and correct labels, ensuring high diagnostic accuracy.
        """
    )
    st.subheader("About the Model")
    st.markdown(
        """
        The model is a **Convolutional Neural Network (CNN)**, specifically a `MobileNetV3` architecture, that was fine-tuned on this dataset. It was trained to classify the images into one of the four categories.
        
        The goal of this project is to showcase the potential of deep learning in ophthalmology to assist medical professionals by automating the analysis of OCT scans, potentially reducing time and increasing diagnostic accuracy.
        """
    )
    st.warning("This tool is for educational and informational purposes only and is not a substitute for professional medical diagnosis or advice.", icon="⚠️")