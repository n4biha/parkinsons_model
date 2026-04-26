import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from PIL import Image
from fusion import fuse_predictions
import torch
import torch.nn as nn
from torchvision import models, transforms
import json


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

# Load clinical models
clinical_model = joblib.load('models/clinical_model.pkl')
clinical_scaler = joblib.load('models/clinical_scaler.pkl')
selected_features = joblib.load('models/selected_features.pkl')

# Load CNN drawing model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
cnn_model = models.mobilenet_v2(weights=None)
num_features = cnn_model.classifier[1].in_features
cnn_model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(num_features, 2)
)
cnn_model.load_state_dict(torch.load('models/drawing_cnn_final.pth', map_location=device))
cnn_model = cnn_model.to(device)
cnn_model.eval()

cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

st.set_page_config(page_title="Parkinson's Screening Tool", layout="wide")

page = st.sidebar.radio("Navigation", ["Screening Tool", "Dashboard"])

with st.sidebar:
    st.title("About")
    
    st.subheader("How It Works")
    st.write(
        "This tool combines two machine learning models to screen for Parkinson's disease risk. "
        "Part 1 analyzes your self-reported symptoms through a questionnaire. "
        "Part 2 analyzes a spiral drawing you upload using a CNN with transfer learning. "
        "The results are combined using late fusion logic."
    )

    st.subheader("Datasets")
    st.write(
        "**Clinical data:** Rabie El Kharoua Parkinson's Disease Dataset (Kaggle) — "
        "2,105 patients with 32 features. Note: this is a synthetic dataset."
    )
    st.write(
        "**Drawing data:** Original Parkinson's Drawings (Kaggle, kmader) — "
        "102 spiral drawings from 51 healthy and 51 PD patients. "
        "Models evaluated using 5-fold cross-validation."
    )   
    
    st.subheader("Disclaimer")
    st.caption(
        "This is a screening tool for educational purposes only. "
        "It is not a substitute for professional medical diagnosis. "
        "Please consult a healthcare provider for proper evaluation."
    )

if 'clinical_prob' not in st.session_state:
    st.session_state.clinical_prob = None
if 'drawing_prob' not in st.session_state:
    st.session_state.drawing_prob = None


if page == 'Screening Tool':
    st.title("Parkinson's Disease Screening Tool")
    st.markdown("*A two-part screening tool combining health questionnaire and spiral drawing analysis*")

    st.header("Part 1: Health Questionnaire")
    st.write("Please answer the following questions based on your current health.")

    # Demographics
    st.subheader("Demographics")
    age = st.slider("Age", min_value=50, max_value=89, value=65)

    # Lifestyle
    st.subheader("Lifestyle")
    diet_quality = st.slider("Diet Quality (0 = poor, 10 = excellent)", 0, 10, 5)
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

    # Predict button
    if st.button("Predict Risk"):
        yes_no_map = {"Yes": 1, "No": 0}
        
        input_data = {
            'Age': age, 'Gender': 0, 'Ethnicity': 0, 'EducationLevel': 1,
            'BMI': 27, 'Smoking': 0, 'AlcoholConsumption': 5, 'PhysicalActivity': 5,
            'DietQuality': diet_quality, 'SleepQuality': 7,
            'FamilyHistoryParkinsons': 0,
            'TraumaticBrainInjury': yes_no_map[tbi],
            'Hypertension': 0,
            'Diabetes': yes_no_map[diabetes],
            'Depression': yes_no_map[depression],
            'Stroke': 0, 'SystolicBP': 130, 'DiastolicBP': 80,
            'CholesterolTotal': 200, 'CholesterolLDL': 100,
            'CholesterolHDL': 50, 'CholesterolTriglycerides': 150,
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
        
        df_input = pd.DataFrame([input_data])
        scaled = clinical_scaler.transform(df_input)
        feature_indices = [list(df_input.columns).index(f) for f in selected_features]
        selected = scaled[:, feature_indices]
        
        probability = clinical_model.predict_proba(selected)[0, 1]
        st.session_state.clinical_prob = probability
        
        st.subheader("Prediction Result")
        st.progress(float(probability))
        
        if probability < 0.35:
            st.success(f"Low Risk — {probability*100:.1f}% probability of PD")
        elif probability < 0.65:
            st.warning(f"Moderate Risk — {probability*100:.1f}% probability of PD")
        else:
            st.error(f"High Risk — {probability*100:.1f}% probability of PD")
        
        # SHAP explanation
        st.subheader("What's driving this prediction?")
        explainer = shap.TreeExplainer(clinical_model)
        shap_values = explainer.shap_values(selected)
        shap_vals = shap_values[0, :, 1]
        feature_impact = list(zip(selected_features, shap_vals))
        feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
        
        st.write("The top 3 factors influencing your score:")
        for feat, val in feature_impact[:3]:
            direction = "increased" if val > 0 else "decreased"
            friendly_name = FEATURE_DISPLAY_NAMES.get(feat, feat)
            st.write(f"- **{friendly_name}** {direction} your risk")

    st.header("Part 2: Spiral Drawing Analysis")
    st.write(
        "Draw a spiral on a blank piece of paper, take a clear photo of it, "
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
            image = Image.open(uploaded_file).convert('L')
            
            # Apply CNN transforms
            img_tensor = cnn_transform(image).unsqueeze(0).to(device)
            
            # Predict

            with torch.no_grad():
                outputs = cnn_model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                drawing_prob = probs[1].item()
            
            prediction = "Parkinson's" if drawing_prob >= 0.5 else "Healthy"
            st.session_state.drawing_prob = drawing_prob
            
            st.subheader("Drawing Analysis Result")
            if prediction == "Parkinson's":
                st.error(f"Prediction: {prediction} — {drawing_prob*100:.1f}% probability of PD")
            else:
                st.success(f"Prediction: {prediction} — {(1-drawing_prob)*100:.1f}% confidence")

    st.header("Combined Analysis")

    if st.session_state.clinical_prob is not None and st.session_state.drawing_prob is not None:
        if st.button("Run Combined Analysis"):
            clinical_p = st.session_state.clinical_prob
            drawing_p = st.session_state.drawing_prob
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Questionnaire Risk", f"{clinical_p*100:.1f}%")
            with col2:
                st.metric("Drawing Analysis Risk", f"{drawing_p*100:.1f}%")
            
            result = fuse_predictions(clinical_p, drawing_p)
            
            st.subheader("Combined Interpretation")
            if "High Risk" in result:
                st.error(result)
            elif "Low Risk" in result:
                st.success(result)
            elif "Mixed Signals" in result:
                st.warning(result)
            else:
                st.info(result)
            
            st.caption(
                "⚠️ This is a screening tool, not a medical diagnosis. "
                "Please consult a healthcare professional for proper evaluation."
            )
    else:
        st.info("Complete both Part 1 and Part 2 to run combined analysis.")

if page == "Dashboard":
    import plotly.express as px
    import plotly.graph_objects as go
    
    st.title("Model Performance Dashboard")
    st.caption("Comprehensive evaluation of clinical and drawing-based classifiers")
    
    class_dist = pd.read_csv('models/class_distribution.csv')
    feature_importance = pd.read_csv('models/feature_importance.csv')
    roc_clinical = pd.read_csv('models/roc_clinical.csv')
    roc_drawing = pd.read_csv('models/roc_drawing.csv')
    cm_clinical = pd.read_csv('models/confusion_matrix.csv', index_col=0)
    cm_drawing = pd.read_csv('models/confusion_matrix_drawing.csv', index_col=0)
    model_comparison = pd.read_csv('models/model_comparison.csv')
    
    st.subheader("Key Metrics")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Patients (Clinical)", f"{class_dist['Count'].sum():,}")
    kpi2.metric("Clinical Accuracy", "92.9%")
    kpi3.metric("Clinical ROC AUC", "0.956")
    kpi4.metric("Drawing Accuracy", '92.1%')
    kpi5.metric("Drawing ROC AUC", "0.944")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Class Distribution")
        fig = px.pie(class_dist, values='Count', names='Class',
                     hole=0.4,
                     color_discrete_sequence=['#2E86AB', '#A23B72'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Feature Importance (Clinical)")
        fi_sorted = feature_importance.sort_values('Importance', ascending=True)
        fig = px.bar(fi_sorted, x='Importance', y='Feature',
                     orientation='h',
                     color='Importance',
                     color_continuous_scale='Blues')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("ROC Curves — Model Comparison")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roc_clinical['FPR'], y=roc_clinical['TPR'],
                         mode='lines', name='Clinical Model (AUC = 0.956)',
                         line=dict(color='#2E86AB', width=3)))
    fig.add_trace(go.Scatter(x=roc_drawing['FPR'], y=roc_drawing['TPR'],
                         mode='lines', name='Drawing Model (AUC = 0.944)',
                         line=dict(color='#A23B72', width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                            mode='lines', name='Random Chance',
                            line=dict(color='gray', width=2, dash='dash')))
    fig.update_layout(xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Confusion Matrix — Clinical")
        fig = px.imshow(cm_clinical.values,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Healthy', 'PD'],
                        y=['Healthy', 'PD'],
                        text_auto=True,
                        color_continuous_scale='Blues')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        st.subheader("Confusion Matrix — Drawing (CNN)")
        fig = px.imshow(cm_drawing.values,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Healthy', 'PD'],
                        y=['Healthy', 'PD'],
                        text_auto=True,
                        color_continuous_scale='Purples')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("Logistic Regression vs Random Forest (Clinical)")
    st.dataframe(model_comparison, use_container_width=True, hide_index=True)

    st.subheader("Drawing Model — Cross-Validation Performance")

    with open('models/cv_results.json', 'r') as f:
        cv = json.load(f)

    # CV bar chart
    fold_df = pd.DataFrame({
        'Fold': [f'Fold {i+1}' for i in range(5)],
        'Accuracy': cv['fold_accuracies']
    })

    fig = px.bar(fold_df, x='Fold', y='Accuracy',
                color='Accuracy',
                color_continuous_scale='Purples',
                text='Accuracy')
    fig.update_layout(
        height=400,
        yaxis_range=[80, 100],
        showlegend=False
    )
    fig.add_hline(y=cv['mean_accuracy'], line_dash="dash", line_color="white",
                annotation_text=f"Mean: {cv['mean_accuracy']}%")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"5-fold cross-validation on 102 spiral drawings. Mean: {cv['mean_accuracy']}% ± {cv['std_accuracy']}%")