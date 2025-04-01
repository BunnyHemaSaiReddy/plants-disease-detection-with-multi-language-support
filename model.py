import streamlit as st
import io
import base64
from PIL import Image
from ultralytics import YOLO
import google.generativeai as genai
import numpy as np
from gtts import gTTS

# Configure the Generative AI model
API_KEY = "AIzaSyCEn5YfcEEUnKFTRhLYXO-ebLrAmSW6AUE"  # Apply provided API key
genai.configure(api_key=API_KEY)
model_ai = genai.GenerativeModel("gemini-1.5-flash")

# Load YOLOv8 model
model_yolo = YOLO("best.pt")  # Update path if needed

def process_detections(image, results):
    """Draw bounding boxes and labels on the image."""
    image_np = np.array(image)
    disease_detected = False
    detected_class = ""
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model_yolo.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw bounding box (red outline)
            image_np[y1:y2, x1:x1+3] = [255, 0, 0]
            image_np[y1:y2, x2-3:x2] = [255, 0, 0]
            image_np[y1:y1+3, x1:x2] = [255, 0, 0]
            image_np[y2-3:y2, x1:x2] = [255, 0, 0]
            
            disease_detected = True
            detected_class = class_name
    
    return Image.fromarray(image_np), disease_detected, detected_class

def generate_text(prompt: str) -> str:
    """Generates AI-generated content based on the given prompt."""
    try:
        response = model_ai.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI Setup
st.set_page_config(page_title="Leaf Disease Detector", layout="wide")
st.markdown("""
    <style>
        .main-container { text-align: center; }
        .disease-box { 
            background-color: #f8f9fa; 
            padding: 15px; 
            border-radius: 10px; 
            box-shadow: 2px 2px 5px rgba(0,0,0,0.2); 
        }
        .banner-box { 
            text-align: center; 
            padding: 20px; 
            background-color: #e0f7fa; 
            border-radius: 15px; 
            margin-bottom: 20px; 
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1); 
        }
        .image-container { text-align: center; margin-top: 20px; }
        .divider { margin: 30px auto; width: 80%; height: 1px; background-color: #ddd; }
    </style>
""", unsafe_allow_html=True)

#st.title("🌿 Leaf Disease Detection Using YOLOv8 & AI")

import streamlit as st

st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; background-color: #e0f7fa; padding: 20px; border-radius: 15px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
    <div style="flex: 1; padding-right: 20px;">
        <h2>🌱 Welcome to the Leaf Disease Detection Platform!</h2>
        <p style="text-align: left;font-size: 30px;">
            <h5>🍃 How It Works:</h5>
            - 📸 Capture or upload a leaf image.<br>
            - 🔍 Detect diseases with <b>state-of-the-art AI</b>.<br>
            - 🏥 Get <b>detailed insights</b> and remedies.<br>
            - 🚀 Enhance your <b>crop health and yield.</b><br>
        </p>
    </div>
    <div style="flex-shrink: 0;">
        <img src="https://media.istockphoto.com/id/503646746/photo/farmer-spreading-fertilizer-in-the-field-wheat.jpg?s=612x612&w=0&k=20&c=Lgxsjbz0jaYyQrvfzhyAsW2zELtshRP4AtLzkpmcLiE="
             alt="Farmer Spreading Fertilizer"
             style="height: 200px; width: auto; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.2);">
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown("""
            <br><hr>
            """, unsafe_allow_html=True)




# Image Upload or Camera Input slider
col1, col2 = st.columns([6, 7])

with col1:
    input_method = st.radio("Select Input Method", ('📷 Camera', '📂 Upload Image'), horizontal=True, index=1)

with col2:
    # List of all existing languages
    all_languages = [
        "English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Hindi", "Arabic", 
        "Portuguese", "Russian", "Italian", "Dutch", "Bengali", "Punjabi", "Greek", "Turkish", "Vietnamese", 
        "Hebrew", "Swedish", "Danish", "Finnish", "Norwegian", "Polish", "Czech", "Thai", "Malay", "Indonesian", 
        "Tamil", "Telugu", "Marathi", "Urdu", "Persian", "Hungarian", "Romanian", "Slovak", "Serbian", 
        "Ukrainian", "Lithuanian", "Latvian", "Estonian", "Tagalog", "Swahili", "Afrikaans", "Sinhala"
    ]

    selected_language = st.selectbox("Select Language", all_languages)


st.markdown("""
            <hr>
            """, unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])
with col1:
    if input_method == '📷 Camera':
        uploaded_image = st.camera_input("Take a picture")
    else:
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    #image = image.resize((280, 280))  # Adjusted size for better alignment
    
    results = model_yolo(image)
    processed_image, disease_detected, detected_class = process_detections(image, results)
    
    with col2:
        st.write("         ")
        image = image.resize((500,300))  # Resize image before displaying
        st.image(image, caption="📌 Uploaded Image", width=200)
    st.markdown("""
            <hr>
            """, unsafe_allow_html=True)
    buffered = io.BytesIO()
    processed_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()


    # First Div - Processed Image and Detected Disease Info
    # Assuming img_str contains the Base64-encoded image data
        
    # Generate text descriptions in the selected language
    disease_description = generate_text(f"Describe {detected_class} leaf disease or pests and causes. In 30 lines only" + " give the output in this language: " + selected_language)
   

    # Function to convert text to speech and return base64-encoded audio
    def text_to_speech_base64(text, lang):
        tts = gTTS(text=text, lang=lang)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_bytes = audio_buffer.getvalue()
        return base64.b64encode(audio_bytes).decode("utf-8")

    # Convert both texts to speech
    # Change "en" to selected_language's code if supported
    # Change "en" to selected_language's code if supported

    # Display the disease description and processed image
    st.markdown('<div class="disease-output">', unsafe_allow_html=True)

    if disease_detected:
        st.markdown(f"""
        <div style="display: flex; align-items: flex-start; border: 2px solid #e74c3c; border-radius: 10px; padding: 15px; background-color: #fff3e3; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 20px;">
            <div style="flex: 1; margin-right: 20px;">
                <h3 style="color: #c0392b;">🦠 Detected Disease: {detected_class}</h3>
                <p>{disease_description}</p>
            </div>
            <div style="flex-shrink: 0;">
                <h3>🔍 Processed Image</h3>
                <img src="data:image/png;base64,{img_str}" width="350" style="border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"/>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ This image does not contain a leaf or a detectable disease.")
        
    disease_audio_base64 = text_to_speech_base64(disease_description, "en")
    st.audio(disease_audio_base64)
    st.markdown('</div>', unsafe_allow_html=True)
    remedies_response = generate_text(f"Provide remedies for {detected_class} leaf disease or pests and mention 4 fertilizers of 2 natural and 2 broughable products in a tabular format. In 20 lines only" + " give the output in this language: " + selected_language)
    # Second Div - Remedies Section
    st.markdown(f"""
    <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 15px; background-color: #e8f5e9; margin-top: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <h3 style="color: #2c3e50;">💡 Remedies:</h3>
        <p>{remedies_response.strip()}</p>
    </div><br>
    """, unsafe_allow_html=True)
    remedies_audio_base64 = text_to_speech_base64(remedies_response, "en") 
    st.audio(remedies_audio_base64)
st.markdown("""
            <hr><br><br>
                <br>
                """, unsafe_allow_html=True)





st.markdown("""
### 🌾 Benefits for Farmers 🌱

**🍃 Accurate Disease Detection:**  
- The website uses advanced deep learning models trained on datasets like PlantVillage to identify diseases with high precision (up to 99.99% accuracy).  
- Early detection prevents the spread of diseases across crops, saving farmers from potential losses.  

**⏱️ Time and Cost Efficiency:**  
- Automated systems reduce the need for manual inspections, saving time and labor costs.  
- Farmers can focus on other critical aspects of farming while relying on AI for disease identification.  

**📱 Accessible Technology:**  
- The platform is designed to be user-friendly, allowing farmers with minimal technical expertise to upload images and receive instant results.  
- Mobile integration ensures accessibility even in rural areas.  

**🔍 Comprehensive Insights:**  
- Provides detailed information about detected diseases, including symptoms, causes, and remedies.  
- Offers recommendations for pesticides or other treatments tailored to specific diseases.  

**🌱 Improved Crop Management:**  
- Enables farmers to monitor crop health regularly and make data-driven decisions.  
- Supports sustainable agricultural practices by optimizing resource usage.  
""")

st.markdown("""
                <hr><br><br>
                """, unsafe_allow_html=True)
# Key Features Section
st.markdown("""
### ✨ Key Features of the Website ✨

**🤖 AI-Powered Detection:**  
- Utilizes models like YOLOv4 and DeepPlantNet for real-time disease classification.  
- Handles multiple crops and diseases efficiently.  

**📊 High-Quality Dataset Integration:**  
- Trained on datasets like PlantVillage with over 50,000 labeled images of healthy and diseased leaves.  
- Ensures robust performance across diverse plant species.  

**📱 Mobile-Friendly Interface:**  
- Farmers can upload leaf images directly from their smartphones.  
- Instant feedback helps in timely decision-making.  

**💡 Remedy Suggestions:**  
- Offers actionable insights, including chemical treatments or organic solutions for disease management.  

**🌍 Sustainability Focus:**  
- Promotes eco-friendly practices by reducing unnecessary pesticide use through targeted interventions.  
""")



# Footer Section
st.markdown("""
<div style="text-align: center; padding-top: 20px;">
<hr>
    🌟 Powered by Bunny | © bunnyhemasaireddy@gmail.com 🌟
</div>
""", unsafe_allow_html=True)




