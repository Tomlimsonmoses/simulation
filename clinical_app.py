"""
Clinical Decision Support Interface
===================================
Professional EHR-style interface for the PK-PD Modeling Engine.
All controls visible on one screen with automatic updates.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os
from typing import Dict, List, Tuple

# Ensure we can import simulate.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page Configuration
st.set_page_config(
    page_title="PK-PD EHR System",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed"  # Changed to collapsed to maximize main area
)

# Custom CSS for better layout
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* App Background Base */
    .stApp {
        background: #f4f7f6;
    }

    /* Main container styling */
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 100%;
    }
    
    /* Metric cards - Premium Glassmorphism Look */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(255,255,255,0.8));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.4);
        padding: 1.5rem;
        border-radius: 16px;
        color: #2b3674;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.06);
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        margin-top: 0.5rem;
        color: #111c43;
        background: -webkit-linear-gradient(45deg, #4318ff, #868cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 13px;
        font-weight: 600;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #8f9bba;
    }
    
    /* Control panels */
    .control-panel {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(226, 232, 240, 0.8);
    }
    
    .panel-title {
        font-size: 18px;
        font-weight: 700;
        color: #111c43;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #f1f5f9;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Status indicators */
    .status-fatal {
        background: linear-gradient(135deg, #ffe5e5 0%, #ffcaca 100%);
        color: #d63031;
        border-left: 5px solid #d63031;
        padding: 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .status-recovery {
        background: linear-gradient(135deg, #e3fbe3 0%, #c4f3c4 100%);
        color: #1e8a36;
        border-left: 5px solid #2ecc71;
        padding: 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .status-critical {
        background: linear-gradient(135deg, #fff4e6 0%, #fee4c3 100%);
        color: #d35400;
        border-left: 5px solid #f39c12;
        padding: 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Streamlit Inputs Overrides */
    .stSelectbox div[data-baseweb="select"], .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        background-color: #f8fafc;
        transition: all 0.2s ease;
    }
    
    .stSelectbox div[data-baseweb="select"]:hover, .stNumberInput > div > div > input:hover {
        border-color: #cbd5e1;
        background-color: #ffffff;
    }
    
    /* Clean up expanders */
    .stExpander {
        border: 1px solid #f1f5f9;
        border-radius: 12px;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }

    /* Slider styling */
    .stSlider > div {
        margin-top: 0.5rem;
    }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #4318ff 0%, #868cff 100%);
        color: white;
        border: none;
        padding: 0.6rem;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(67, 24, 255, 0.2);
        transition: all 0.2s ease;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(67, 24, 255, 0.3);
        color: white;
    }
    
    /* Hide annoying Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Backend
# ---------------------------------------------------------
# ---------------------------------------------------------
# Load Backend
# ---------------------------------------------------------
@st.cache_resource
def load_backend():
    try:
        from simulate import PKPDSimulator
        simulator = None
        if os.path.exists("checkpoints/simple_mlp_model.pth"):
            simulator = PKPDSimulator("checkpoints/simple_mlp_model.pth")
        
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from expanded_drug_database import ExpandedDrugDatabase
        from disease_specific_models import DiseaseSpecificModeler
        drug_db = ExpandedDrugDatabase()
        disease_modeler = DiseaseSpecificModeler()
        
        import pandas as pd
        organ_df = pd.read_csv("./real_data/organ_network.csv")
        organs = organ_df['organ_name'].tolist()
        
        return simulator, drug_db, disease_modeler, organs
    except Exception as e:
        import traceback
        st.error(f"Backend load error: {traceback.format_exc()}")
        return None, None, None, []

simulator, drug_db, disease_modeler, AVAILABLE_ORGANS = load_backend()

DISEASE_TO_DRUG_CLASS_MAP = {
    'chronic_kidney_disease': ['NSAIDs', 'Antibiotics'],
    'liver_cirrhosis': ['Analgesics', 'Antibiotics'],
    'heart_failure': ['Cardiovascular'],
    'diabetes_mellitus': ['Antidiabetics'],
    'copd': ['Antibiotics', 'Analgesics'],
    'epilepsy': ['Psychiatric'],
}

PATHOGEN_TO_DRUG_CLASS_MAP = {
    'Staphylococcus aureus': ['Antibiotics'],
    'Escherichia coli': ['Antibiotics'],
    'Streptococcus pneumoniae': ['Antibiotics'],
    'Influenza A': ['Analgesics', 'NSAIDs'],
    'SARS-CoV-2': ['Analgesics', 'NSAIDs']
}

# ---------------------------------------------------------
# Session State Initialize
# ---------------------------------------------------------
default_patient = {
    'id': 'MRN-100293',
    'age': 45,
    'weight': 75.0,
    'height': 175,
    'sex': 'Male',
    'ethnicity': 'Caucasian',
    'blood_type': 'O+',
    'allergies': 'None',
    'comorbidities': 'None'
}
# Dynamically add all organs to default_patient
for org in (AVAILABLE_ORGANS if AVAILABLE_ORGANS else []):
    default_patient[f"organ_{org}"] = 85 if org == 'liver' else (95 if org == 'kidneys' else 100)

if 'patient_data' not in st.session_state:
    st.session_state.patient_data = default_patient.copy()
else:
    for key, val in default_patient.items():
        st.session_state.patient_data.setdefault(key, val)

if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

if 'drug_selection' not in st.session_state:
    st.session_state.drug_selection = {
        'drug1': 'Acetaminophen',
        'dose1': 250.0,
        'drug2': None,
        'dose2': 0.0,
        'combination': False
    }

if 'biological_state' not in st.session_state:
    st.session_state.biological_state = {
        'immune_status': 75,
        'inflammation_level': 50,
        'organ_health': 80,
        'metabolic_rate': 70
    }

# Available options
AVAILABLE_DRUGS = drug_db.get_all_drug_names() if drug_db else ['Acetaminophen', 'Ibuprofen']
AVAILABLE_PATHOGENS = ['None'] + list(PATHOGEN_TO_DRUG_CLASS_MAP.keys())
AVAILABLE_DISEASES = ['None'] + list(disease_modeler.disease_database.keys()) if disease_modeler else ['None']

# ---------------------------------------------------------
# Main Layout
# ---------------------------------------------------------
st.title("⚕️ Human Biological System Response Simulator")
st.markdown("---")

# Top Row - Patient Summary and Quick Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">🆔 Patient ID</div>
        <div class="metric-value">{}</div>
        <div class="metric-label">📅 Age: {} years</div>
    </div>
    """.format(st.session_state.patient_data['id'], st.session_state.patient_data['age']), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">⚖️ Physical Stats</div>
        <div class="metric-value">{:.1f} kg</div>
        <div class="metric-label">📏 {:.0f} cm | 🧬 {}</div>
    </div>
    """.format(st.session_state.patient_data['weight'], 
               st.session_state.patient_data['height'],
               st.session_state.patient_data['sex']), unsafe_allow_html=True)

with col3:
    kidneys = st.session_state.patient_data.get('organ_kidneys', 95)
    liver = st.session_state.patient_data.get('organ_liver', 85)
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">🔬 Organ Function</div>
        <div class="metric-value">eGFR: {}</div>
        <div class="metric-label">Liver: {}% | 🩸 Type: {}</div>
    </div>
    """.format(kidneys, liver, st.session_state.patient_data['blood_type']), unsafe_allow_html=True)

with col4:
    engine_status = "✅ Online" if simulator else "⚠️ Demo Mode"
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">⚡ System Status</div>
        <div class="metric-value">{}</div>
        <div class="metric-label">🎯 Auto-update: ON</div>
    </div>
    """.format(engine_status), unsafe_allow_html=True)

st.markdown("---")

# Main Content - Split into two columns
left_col, right_col = st.columns([1.2, 1.8])

with left_col:
    # Patient Configuration Panel
    with st.container():
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">👤 Patient Biological Configuration</div>', unsafe_allow_html=True)
        
        with st.expander("📋 Demographics & Vitals", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                age = st.slider("Age (years)", 0, 100, st.session_state.patient_data['age'], 
                              key="age_slider", help="Affects drug metabolism and clearance")
                weight = st.number_input("Weight (kg)", 20.0, 200.0, st.session_state.patient_data['weight'],
                                       step=1.0, key="weight_input")
                sex = st.selectbox("Sex", ["Male", "Female"], 
                                 index=0 if st.session_state.patient_data['sex'] == "Male" else 1,
                                 key="sex_select")
            with col_b:
                height = st.number_input("Height (cm)", 100, 250, st.session_state.patient_data['height'],
                                       step=1, key="height_input")
                ethnicity = st.selectbox("Ethnicity", ["Caucasian", "Asian", "African", "Hispanic", "Other"],
                                       key="ethnicity_select")
                blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
                                        key="blood_type_select")
        
        with st.expander("🏥 Organ Function & Health Status", expanded=True):
            st.markdown("Adjust individual organ health status (%)")
            organ_cols = st.columns(2)
            
            # Map our list to column chunks
            for i, org in enumerate(AVAILABLE_ORGANS):
                col_idx = i % 2
                org_name = org.replace('_', ' ').title()
                with organ_cols[col_idx]:
                    val = st.slider(f"{org_name}", 0, 100, st.session_state.patient_data.get(f"organ_{org}", 100), key=f"org_{org}_slider")
                    st.session_state.patient_data[f"organ_{org}"] = val

        with st.expander("🦠 Disease & Pathogen Exposure", expanded=True):
            disease = st.selectbox("Primary Disease Condition", [d.replace('_', ' ').title() if d != 'None' else 'None' for d in AVAILABLE_DISEASES], key="disease_select")
            disease_raw = AVAILABLE_DISEASES[[d.replace('_', ' ').title() if d != 'None' else 'None' for d in AVAILABLE_DISEASES].index(disease)]
            pathogen = st.selectbox("Pathogen Exposure", AVAILABLE_PATHOGENS, key="pathogen_select")
            show_all_drugs = st.checkbox("Show All Drugs (Override Filter)", value=False, help="Uncheck to filter drugs based on the selected disease or pathogen.")
            
            # Biological state parameters
            st.markdown("**System Biological State**")
            immune_status = st.slider("Immune System Strength (%)", 0, 100, 
                                    st.session_state.biological_state['immune_status'],
                                    key="immune_slider")
            inflammation = st.slider("Inflammation Level (%)", 0, 100,
                                   st.session_state.biological_state['inflammation_level'],
                                   key="inflammation_slider")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Drug Selection Panel
    with st.container():
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">💊 Drug Therapy Configuration</div>', unsafe_allow_html=True)
        
        # Smart drug filtering
        filtered_drugs = []
        if not show_all_drugs:
            relevant_classes = set()
            if disease_raw in DISEASE_TO_DRUG_CLASS_MAP:
                relevant_classes.update(DISEASE_TO_DRUG_CLASS_MAP[disease_raw])
            if pathogen in PATHOGEN_TO_DRUG_CLASS_MAP:
                relevant_classes.update(PATHOGEN_TO_DRUG_CLASS_MAP[pathogen])
            
            if relevant_classes and drug_db:
                for cls in relevant_classes:
                    filtered_drugs.extend(drug_db.get_drugs_by_class(cls))
            
            if not filtered_drugs:
                filtered_drugs = AVAILABLE_DRUGS
        else:
            filtered_drugs = AVAILABLE_DRUGS
            
        # Ensure unique array
        filtered_drugs = list(dict.fromkeys(filtered_drugs))
        
        combination = st.checkbox("Combination Therapy", key="combination_checkbox",
                                 help="Enable for multi-drug treatment")
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            drug1 = st.selectbox("Primary Drug", filtered_drugs, key="drug1_select")
            dose1 = st.number_input("Dosage (mg)", 0.0, 2000.0, 250.0, step=25.0, key="dose1_input",
                                   help="Enter drug dosage in milligrams")
            route = st.selectbox("Administration Route", ["Oral", "Intravenous", "Subcutaneous"], key="route_select")
        
        with col_d2:
            if combination:
                drug2 = st.selectbox("Secondary Drug", ["None"] + filtered_drugs, key="drug2_select")
                if drug2 != "None":
                    dose2 = st.number_input("Secondary Dosage (mg)", 0.0, 2000.0, 100.0, step=25.0, key="dose2_input")
                else:
                    dose2 = 0.0
            else:
                drug2 = None
                dose2 = 0.0
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Update session state
    st.session_state.patient_data.update({
        'age': age, 'weight': weight, 'height': height, 'sex': sex,
        'ethnicity': ethnicity, 'blood_type': blood_type
    })
    st.session_state.biological_state.update({
        'immune_status': immune_status, 'inflammation_level': inflammation
    })

with right_col:
    # Auto-run simulation when any parameter changes
    def run_simulation():
        """Run PK-PD simulation with current parameters"""
        try:
            # Simulate drug concentrations and responses
            time_points = np.linspace(0, 48, 200)  # 48 hours simulation
            
            # Calculate PK parameters based on patient characteristics
            age_factor = 1.0 - (max(0, age - 40) * 0.005)  # Reduced clearance with age
            kidneys = st.session_state.patient_data.get('organ_kidneys', 95)
            liver = st.session_state.patient_data.get('organ_liver', 85)
            renal_factor = kidneys / 100.0
            liver_factor = liver / 100.0
            weight_factor = weight / 70.0
            
            # Drug-specific parameters
            props1 = drug_db.get_drug_properties(drug1) if drug_db else None
            if props1:
                params = {
                    'halflife': props1.get('half_life', 3.0),
                    'therapeutic_range': props1.get('therapeutic_range', (5, 20)),
                    'toxic_threshold': props1.get('toxicity_threshold', 50),
                    'class': props1.get('class', 'Unknown'),
                    'metabolism': props1.get('metabolism', 'unknown'),
                    'excretion': props1.get('excretion', 'unknown')
                }
            else:
                params = {'halflife': 3.0, 'therapeutic_range': (5, 20), 'toxic_threshold': 50, 'class': 'Unknown', 'metabolism': 'hepatic', 'excretion': 'renal'}
            
            # Calculate effective half-life with patient factors
            effective_halflife = params['halflife'] / (max(0.1, renal_factor * liver_factor * age_factor))
            
            # Generate PK curve
            if route == "Oral":
                absorption_rate = 0.5  # Oral absorption
                pk_concentration = dose1 * absorption_rate * weight_factor * np.exp(-0.693/effective_halflife * time_points)
                # Add absorption phase
                absorption_phase = 1 - np.exp(-2 * time_points)
                pk_concentration = pk_concentration * absorption_phase
            else:
                # IV administration
                pk_concentration = dose1 * weight_factor * np.exp(-0.693/effective_halflife * time_points)
            
            # Add second drug effect if combination therapy
            if combination and drug2 and drug2 != "None":
                props2 = drug_db.get_drug_properties(drug2) if drug_db else None
                params2 = {'halflife': props2.get('half_life', 3.0)} if props2 else {'halflife': 3.0}
                pk_concentration2 = dose2 * weight_factor * np.exp(-0.693/params2['halflife'] * time_points)
                pk_concentration = pk_concentration + pk_concentration2 * 0.3  # 30% synergy
            
            # Calculate PD response (efficacy)
            pd_params = drug_db.calculate_pd_parameters(drug1, st.session_state.patient_data) if drug_db else None
            ec50 = pd_params['ec50'] if pd_params else params['therapeutic_range'][0]
            hill_coefficient = pd_params['hill_coefficient'] if pd_params else 1.5
            emax = pd_params['emax'] if pd_params else 100
            onset_time = pd_params['onset_time'] if pd_params else 1.0
            
            pd_response = np.zeros_like(pk_concentration)
            for i, t in enumerate(time_points):
                if t >= onset_time:
                    pd_response[i] = (emax * (pk_concentration[i] ** hill_coefficient)) / ((ec50 ** hill_coefficient) + (pk_concentration[i] ** hill_coefficient))
            
            # Modify response based on biological state and disease
            immune_modifier = immune_status / 100.0
            inflammation_modifier = 1 + (inflammation / 100.0) * 0.5
            disease_modifier = 0.7 if disease != "None" else 1.0
            pathogen_modifier = 0.6 if pathogen != "None" else 1.0
            
            pd_response = pd_response * immune_modifier * inflammation_modifier * disease_modifier * pathogen_modifier
            
            # Determine outcome
            max_response = np.max(pd_response)
            max_concentration = np.max(pk_concentration)
            
            side_effects = []
            toxic_indices = np.where(pk_concentration > params['toxic_threshold'])[0]
            if len(toxic_indices) > 0:
                toxic_time = time_points[toxic_indices[0]]
                if params['metabolism'] == 'hepatic':
                    side_effects.append(f"Hepatotoxicity onset predicted at T={toxic_time:.1f}h.")
                if params['excretion'] == 'renal':
                    side_effects.append(f"Nephrotoxicity onset predicted at T={toxic_time:.1f}h.")
                if not side_effects:
                    side_effects.append(f"Systemic toxicity onset predicted at T={toxic_time:.1f}h.")
            
            if max_concentration > params['toxic_threshold']:
                outcome = "⚠️ FATAL"
                outcome_color = "fatal"
                recommendation = "Reduce dosage immediately. Toxic levels detected. " + " ".join(side_effects)
            elif max_response < 30:
                outcome = "🔴 CRITICAL"
                outcome_color = "critical"
                recommendation = "Increase dosage or consider alternative therapy. Inadequate response."
            elif max_response > 70:
                outcome = "✅ RECOVERY"
                outcome_color = "recovery"
                recommendation = "Therapeutic regimen effective. Continue monitoring."
                if side_effects:
                    recommendation += " Note: " + " ".join(side_effects)
            else:
                outcome = "🟡 MONITOR"
                outcome_color = "critical"
                recommendation = "Partial response. Consider adjusting therapy."
                if side_effects:
                    recommendation += " Note: " + " ".join(side_effects)
            
            return {
                'time': time_points,
                'pk_concentration': pk_concentration,
                'pd_response': pd_response,
                'max_concentration': max_concentration,
                'max_response': max_response,
                'outcome': outcome,
                'outcome_color': outcome_color,
                'recommendation': recommendation,
                'drug_params': params,
                'onset_time': onset_time,
                'side_effects': side_effects
            }
        except Exception as e:
            return None
    
    # Auto-run simulation
    simulation_results = run_simulation()
    
    if simulation_results:
        # Simulation Visualization
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📊 Real-time Simulation Results</div>', unsafe_allow_html=True)
        
        # Create interactive plot
        fig = go.Figure()
        
        # PK Concentration
        fig.add_trace(go.Scatter(
            x=simulation_results['time'],
            y=simulation_results['pk_concentration'],
            name='Drug Concentration',
            line=dict(color='#3498db', width=3),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.1)',
            yaxis='y'
        ))
        
        # PD Response
        fig.add_trace(go.Scatter(
            x=simulation_results['time'],
            y=simulation_results['pd_response'],
            name='Therapeutic Response',
            line=dict(color='#2ecc71', width=3, dash='dash'),
            yaxis='y2'
        ))
        
        # Add therapeutic range shading
        if simulation_results['drug_params']['therapeutic_range']:
            fig.add_hrect(
                y0=simulation_results['drug_params']['therapeutic_range'][0],
                y1=simulation_results['drug_params']['therapeutic_range'][1],
                fillcolor="rgba(46, 204, 113, 0.2)",
                line_width=0,
                annotation_text="Therapeutic Range",
                annotation_position="top left"
            )
        
        # Add toxic threshold line
        fig.add_hline(
            y=simulation_results['drug_params']['toxic_threshold'],
            line_dash="dash",
            line_color="red",
            annotation_text="Toxic Threshold",
            annotation_position="top right"
        )
        
        fig.update_layout(
            title=f"{drug1} @ {dose1}mg - System Response Prediction",
            xaxis=dict(title="Time (hours)", gridcolor='lightgray'),
            yaxis=dict(title="Concentration (mg/L)", side='left', gridcolor='lightgray'),
            yaxis2=dict(title="Response Effect (%)", side='right', overlaying='y', 
                       range=[0, 100], gridcolor='lightgray'),
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Outcome and Recommendation
        st.markdown("---")
        
        outcome_class = f"status-{simulation_results['outcome_color']}"
        st.markdown(f"""
        <div class="{outcome_class}">
            <h3>{simulation_results['outcome']}</h3>
            <p style="margin-top: 10px;">{simulation_results['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Peak Concentration", f"{simulation_results['max_concentration']:.1f} mg/L",
                     delta=f"Toxicity at {simulation_results['drug_params']['toxic_threshold']:.0f} mg/L")
        with col_m2:
            st.metric("Maximum Response", f"{simulation_results['max_response']:.1f}%",
                     delta="Therapeutic goal: >70%")
        with col_m3:
            time_to_peak = simulation_results['time'][np.argmax(simulation_results['pk_concentration'])]
            st.metric("Time to Peak", f"{time_to_peak:.1f} hours")
        with col_m4:
            half_life = simulation_results['drug_params']['halflife']
            st.metric("Elimination Half-life", f"{half_life:.1f} hours")
        
        # Best combination recommendation
        if not combination:
            st.markdown("---")
            st.markdown("### 💡 Optimal Combination Recommendation")
            
            # Simple logic to recommend best combination
            if simulation_results['max_response'] < 50:
                recommended_drugs = [d for d in AVAILABLE_DRUGS if d != drug1][:2]
                st.info(f"**Recommended Combination:** Consider adding {recommended_drugs[0]} at 50-100mg or {recommended_drugs[1]} at 100-200mg for synergistic effect")
            elif simulation_results['max_concentration'] > simulation_results['drug_params']['toxic_threshold'] * 0.8:
                st.warning(f"**Dosage Adjustment:** Reduce {drug1} to {dose1 * 0.7:.0f}mg to avoid toxicity while maintaining efficacy")
            else:
                st.success("**Current regimen appears optimal. Continue monitoring.**")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Simulation engine error. Please check your configuration.")

# Bottom Information
st.markdown("---")
st.caption("⚕️ Clinical Decision Support System | Real-time PK-PD Simulation | Auto-updates on parameter change")

# Auto-run trigger (any change in parameters will trigger re-run due to Streamlit's reactive nature)
# This is handled automatically by Streamlit's execution model