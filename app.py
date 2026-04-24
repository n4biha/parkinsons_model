import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import cv2
from skimage.feature import hog
from PIL import Image


FEATURE_DISPLAY_NAMES = {
    'Age': 'Your age',
    'DietQuality': 'Diet quality',
    'TraumaticBrainInjury': 'History of brain injury',
    'Diabetes': 'Diabetes',
    'Depression': 'Depression',
    'UPDRS': 'Severity of tremor/stiffness symptoms',
    'MoCA': 'Memory and cognitive function',
    'FunctionalAssessment': 'Ability to perform daily tasks',
    'Tremor': 'Tremor symptoms',
    'Rigidity': 'Muscle rigidity',
    'Bradykinesia': 'Slowness of movement',
    'PosturalInstability': 'Balance problems',
    'SleepDisorders': 'Sleep disorders',
}

clinical_model = joblib.load('models/clinical_model.pkl')
clinical_scaler = joblib.load('models/clinical_scaler.pkl')
selected_features = joblib.load('models/selected_features.pkl')
drawing_model = joblib.load('models/drawing_model.pkl')
drawing_pca = joblib.load('models/drawing_pca.pkl')

st.set_page_config(page_title="Parkinson's Screening Tool", layout="wide")

st.title("Parkinson's Disease Screening Tool")
st.markdown("*A two-part screening tool combining health questionnaire and spiral drawing analysis*")

st.header("Part 1: Health Questionnaire")
st.write("Please answer the following questions based on your current health.")

# Demographics
st.subheader("Demographics")
age = st.slider("Age", min_value=50, max_value=89, value=65)

# Lifestyle
st.subheader("Lifestyle")
diet_quality = st.slider("Diet Quality (0 = poor, 10 = excellent)", 0.0, 10.0, 5.0)
sleep_disorders = st.radio("Do you have diagnosed sleep disorders?", ["No", "Yes"])

# Medical History
st.subheader("Medical History")
tbi = st.radio("History of traumatic brain injury?", ["No", "Yes"])
diabetes = st.radio("Do you have diabetes?", ["No", "Yes"])
depression = st.radio("Have you been diagnosed with depression?", ["No", "Yes"])

# Motor Symptoms
st.subheader("Motor Symptoms")
tremor = st.radio("Do you experience involuntary shaking (tremor)?", ["No", "Yes"])
rigidity = st.radio("Do your muscles feel stiff or resistant to movement?", ["No", "Yes"])
bradykinesia = st.radio("Have your movements become noticeably slower?", ["No", "Yes"])
postural = st.radio("Do you have balance problems or tendency to fall?", ["No", "Yes"])

# Cognitive & Functional (proxy features)
st.subheader("Cognitive & Functional Assessment")
updrs = st.slider(
    "How severely do tremors or stiffness affect your daily life? (0 = not at all, 10 = severely)",
    0, 10, 3
)
functional = st.slider(
    "How much difficulty do you have with everyday tasks like writing, buttoning clothes, or cutting food? (0 = extreme difficulty, 10 = no difficulty)",
    0, 10, 7
)
memory_issues = st.radio("Do you struggle to remember recent events?", ["No", "Yes"])
confusion = st.radio("Do you sometimes get confused or disoriented?", ["No", "Yes"])
word_finding = st.radio("Do you have trouble finding the right words?", ["No", "Yes"])

# Predict button (no wiring yet)
if st.button("Predict Risk"):
    yes_no_map = {"Yes": 1, "No": 0}
    
    # Build input dict matching training column names
    input_data = {
        'Age': age,
        'Gender': 0,
        'Ethnicity': 0,
        'EducationLevel': 1,
        'BMI': 27,
        'Smoking': 0,
        'AlcoholConsumption': 5,
        'PhysicalActivity': 5,
        'DietQuality': diet_quality,
        'SleepQuality': 7,
        'FamilyHistoryParkinsons': 0,
        'TraumaticBrainInjury': yes_no_map[tbi],
        'Hypertension': 0,
        'Diabetes': yes_no_map[diabetes],
        'Depression': yes_no_map[depression],
        'Stroke': 0,
        'SystolicBP': 130,
        'DiastolicBP': 80,
        'CholesterolTotal': 200,
        'CholesterolLDL': 100,
        'CholesterolHDL': 50,
        'CholesterolTriglycerides': 150,
        'UPDRS': updrs * 20,
        'MoCA': 30 - (yes_no_map[memory_issues] + yes_no_map[confusion] + yes_no_map[word_finding]) * 5,
        'FunctionalAssessment': functional,
        'Tremor': yes_no_map[tremor],
        'Rigidity': yes_no_map[rigidity],
        'Bradykinesia': yes_no_map[bradykinesia],
        'PosturalInstability': yes_no_map[postural],
        'SpeechProblems': 0,
        'SleepDisorders': yes_no_map[sleep_disorders],
        'Constipation': 0,
    }
    
    # Convert to DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Scale
    scaled = clinical_scaler.transform(df_input)
    
    # Select features
    feature_indices = [list(df_input.columns).index(f) for f in selected_features]
    selected = scaled[:, feature_indices]
    
    # Predict
    probability = clinical_model.predict_proba(selected)[0, 1]
    
    # Display
    st.subheader("Prediction Result")
    st.progress(float(probability))
    
    if probability < 0.35:
        st.success(f"Low Risk — {probability*100:.1f}% probability")
    elif probability < 0.65:
        st.warning(f"Moderate Risk — {probability*100:.1f}% probability")
    else:
        st.error(f"High Risk — {probability*100:.1f}% probability")
    

    # SHAP explanation for this specific prediction
    st.subheader("What's driving this prediction?")
    
    explainer = shap.TreeExplainer(clinical_model)
    shap_values = explainer.shap_values(selected)
    
    # Get SHAP values for class 1 (PD) for this single prediction
    shap_vals = shap_values[0, :, 1]
    
    # Pair with feature names and sort by absolute impact
    feature_impact = list(zip(selected_features, shap_vals))
    feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Display top 3
    #POTENTIALLY REMOVE
    st.write("The top 3 factors influencing your score:")
    for feat, val in feature_impact[:3]:
        direction = "increased" if val > 0 else "decreased"
        friendly_name = FEATURE_DISPLAY_NAMES.get(feat, feat)
        st.write(f"- **{friendly_name}** {direction} your risk")

st.header("Part 2: Spiral Drawing Analysis")
st.write(
    "Draw a spiral or wave on a blank piece of paper, take a clear photo of it, "
    "and upload below. For best results, ensure good lighting and the "
    "spiral fills most of the frame."
)

uploaded_file = st.file_uploader(
    "Upload your spiral drawing (JPG or PNG)",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Your uploaded drawing", width=400)
    
    if st.button("Analyze Drawing"):
    # Load image
        image = Image.open(uploaded_file).convert('L')  # grayscale
        img_array = np.array(image)
        
        # Resize to 128x128
        img_resized = cv2.resize(img_array, (128, 128))
        
        # Extract HOG features
        hog_features = hog(
            img_resized,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        
        # Apply PCA
        hog_pca = drawing_pca.transform([hog_features])
        
        # Predict
        drawing_prob = drawing_model.predict_proba(hog_pca)[0, 1]
        prediction = "Parkinson's" if drawing_prob >= 0.5 else "Healthy"
        
        # Display
        st.subheader("Drawing Analysis Result")
        if prediction == "Parkinson's":
            st.error(f"Prediction: {prediction} — {drawing_prob*100:.1f}% confidence")
        else:
            st.success(f"Prediction: {prediction} — {(1-drawing_prob)*100:.1f}% confidence")