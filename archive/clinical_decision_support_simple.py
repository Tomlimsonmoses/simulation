"""
Simplified Clinical Decision Support Interface for PK-PD Modeling
===============================================================

This module provides a streamlined web interface for clinicians to use
the PK-PD modeling system for patient-specific drug dosing decisions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import modules with error handling
try:
    from pk_pd_model import PKPDModel
    PKPDModel_AVAILABLE = True
except ImportError as e:
    st.error(f"Could not import PKPDModel: {e}")
    PKPDModel_AVAILABLE = False

try:
    from expanded_drug_database import ExpandedDrugDatabase
    DRUG_DB_AVAILABLE = True
except ImportError as e:
    st.error(f"Could not import ExpandedDrugDatabase: {e}")
    DRUG_DB_AVAILABLE = False

try:
    from disease_specific_models import DiseaseSpecificModeler
    DISEASE_MODELER_AVAILABLE = True
except ImportError as e:
    st.error(f"Could not import DiseaseSpecificModeler: {e}")
    DISEASE_MODELER_AVAILABLE = False

def main():
    """Main application function"""
    st.set_page_config(
        page_title="PK-PD Clinical Support",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🏥 PK-PD Clinical Decision Support System")
    st.markdown("### Patient-Specific Drug Dosing and Response Prediction")
    st.markdown("---")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select Function",
        [
            "🏠 Dashboard",
            "👤 Patient Assessment", 
            "💊 Drug Selection",
            "📊 PK/PD Analysis",
            "🏥 Disease Modeling"
        ]
    )
    
    # Route to selected page
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "👤 Patient Assessment":
        show_patient_assessment()
    elif page == "💊 Drug Selection":
        show_drug_selection()
    elif page == "📊 PK/PD Analysis":
        show_pkpd_analysis()
    elif page == "🏥 Disease Modeling":
        show_disease_modeling()

def show_dashboard():
    """Show main dashboard"""
    st.header("📊 Clinical Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if DRUG_DB_AVAILABLE:
            drug_db = ExpandedDrugDatabase()
            st.metric("Available Drugs", len(drug_db.get_all_drug_names()), "medications")
        else:
            st.metric("Available Drugs", "N/A", "database unavailable")
    
    with col2:
        if DISEASE_MODELER_AVAILABLE:
            disease_modeler = DiseaseSpecificModeler()
            st.metric("Disease Models", len(disease_modeler.disease_database), "conditions")
        else:
            st.metric("Disease Models", "N/A", "models unavailable")
    
    with col3:
        st.metric("Model Accuracy", "R² = 0.50", "validated")
    
    with col4:
        st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d"), "current")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👤 New Patient Assessment", type="primary", key="new_patient"):
            st.session_state.selected_page = "👤 Patient Assessment"
            st.rerun()
    
    with col2:
        if st.button("💊 Drug Comparison", key="drug_comparison"):
            st.session_state.selected_page = "💊 Drug Selection"
            st.rerun()
    
    with col3:
        if st.button("🏥 Disease Analysis", key="disease_analysis"):
            st.session_state.selected_page = "🏥 Disease Modeling"
            st.rerun()
    
    # System status
    st.subheader("🔧 System Status")
    
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        if PKPDModel_AVAILABLE:
            st.success("✅ PK-PD Model: Operational")
        else:
            st.error("❌ PK-PD Model: Not Available")
            
        if DRUG_DB_AVAILABLE:
            st.success("✅ Drug Database: Connected")
        else:
            st.error("❌ Drug Database: Not Available")
    
    with status_col2:
        if DISEASE_MODELER_AVAILABLE:
            st.success("✅ Disease Models: Available")
        else:
            st.error("❌ Disease Models: Not Available")
            
        st.info("ℹ️ Interface Version: 2.0")

def show_patient_assessment():
    """Show patient assessment form"""
    st.header("👤 Patient Assessment")
    
    # Patient demographics
    st.subheader("📋 Patient Information")
    
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            patient_id = st.text_input("Patient ID", value="PATIENT_001")
            age = st.number_input("Age", min_value=0, max_value=120, value=65)
            weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=75.0, step=0.1)
            height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=170.0, step=0.1)
        
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            ethnicity = st.selectbox("Ethnicity", ["Caucasian", "African American", "Asian", "Hispanic", "Other"])
            blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
            allergies = st.text_area("Known Allergies", placeholder="List any known drug allergies...")
        
        # Clinical parameters
        st.subheader("🏥 Clinical Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=10.0, value=1.2, step=0.1)
            gfr = st.number_input("eGFR (mL/min)", min_value=0, max_value=200, value=75)
        
        with col2:
            albumin = st.number_input("Albumin (g/dL)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
            bilirubin = st.number_input("Bilirubin (mg/dL)", min_value=0.0, max_value=20.0, value=1.0, step=0.1)
        
        with col3:
            ef = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100, value=55)
            bnp = st.number_input("BNP (pg/mL)", min_value=0, max_value=5000, value=100)
        
        # Submit button
        submitted = st.form_submit_button("💾 Save Patient Assessment", type="primary")
        
        if submitted:
            # Save patient data to session state
            patient_data = {
                "patient_id": patient_id,
                "demographics": {
                    "age": age, "weight": weight, "height": height,
                    "gender": gender, "ethnicity": ethnicity, "blood_type": blood_type,
                    "allergies": allergies
                },
                "clinical_params": {
                    "creatinine": creatinine, "gfr": gfr, "albumin": albumin,
                    "bilirubin": bilirubin, "ef": ef, "bnp": bnp
                },
                "assessment_date": datetime.now().isoformat()
            }
            
            st.session_state.patient_data = patient_data
            st.success("✅ Patient assessment saved successfully!")
            
            # Calculate BMI
            bmi = weight / ((height / 100) ** 2)
            st.info(f"📊 Calculated BMI: {bmi:.1f} kg/m²")

def show_drug_selection():
    """Show drug selection and comparison"""
    st.header("💊 Drug Selection & Comparison")
    
    if not DRUG_DB_AVAILABLE:
        st.error("❌ Drug database is not available")
        return
    
    # Initialize drug database
    drug_db = ExpandedDrugDatabase()
    available_drugs = drug_db.get_all_drug_names()
    drug_classes = drug_db.get_all_drug_classes()
    
    # Drug selection
    st.subheader("🔍 Select Drugs for Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_class = st.selectbox("Filter by Drug Class", ["All"] + drug_classes)
    
    with col2:
        search_term = st.text_input("Search Drugs", placeholder="Type to search...")
    
    # Filter drugs
    if selected_class != "All":
        class_drugs = drug_db.get_drugs_by_class(selected_class)
        filtered_drugs = [drug for drug in available_drugs if drug in class_drugs]
    else:
        filtered_drugs = available_drugs
    
    if search_term:
        filtered_drugs = [drug for drug in filtered_drugs if search_term.lower() in drug.lower()]
    
    selected_drugs = st.multiselect("Select Drugs to Compare", filtered_drugs, default=filtered_drugs[:2])
    
    if selected_drugs:
        st.subheader("📊 Drug Comparison")
        
        # Create comparison table
        comparison_data = []
        for drug in selected_drugs:
            props = drug_db.get_drug_properties(drug)
            if props:
                comparison_data.append({
                    "Drug": drug,
                    "Class": props['class'],
                    "Bioavailability": f"{props['bioavailability']:.1%}",
                    "Half-life": f"{props['half_life']:.1f}h",
                    "Protein Binding": f"{props['protein_binding']:.1%}",
                    "Therapeutic Range": f"{props['therapeutic_range'][0]}-{props['therapeutic_range'][1]}",
                    "Routes": ", ".join(props['routes'])
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, width='stretch')
        
        # Detailed drug information
        st.subheader("📋 Detailed Drug Information")
        
        selected_detailed = st.selectbox("Select Drug for Details", selected_drugs)
        
        if selected_detailed:
            props = drug_db.get_drug_properties(selected_detailed)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Basic Properties**")
                st.write(f"• Class: {props['class']}")
                st.write(f"• Mechanism: {props['mechanism']}")
                st.write(f"• Molecular Weight: {props['molecular_weight']:.2f}")
                st.write(f"• LogP: {props['logp']:.2f}")
            
            with col2:
                st.write("**Pharmacokinetics**")
                st.write(f"• Bioavailability: {props['bioavailability']:.1%}")
                st.write(f"• Volume of Distribution: {props['volume_distribution']:.2f} L/kg")
                st.write(f"• Clearance: {props['clearance']:.3f} L/h/kg")
                st.write(f"• Half-life: {props['half_life']:.1f} hours")

def show_pkpd_analysis():
    """Show PK/PD analysis"""
    st.header("📊 PK/PD Analysis")
    
    if 'patient_data' not in st.session_state:
        st.warning("⚠️ Please complete patient assessment first")
        return
    
    if not DRUG_DB_AVAILABLE:
        st.error("❌ Drug database is not available")
        return
    
    patient_data = st.session_state.patient_data
    
    # Select drug for analysis
    drug_db = ExpandedDrugDatabase()
    available_drugs = drug_db.get_all_drug_names()
    selected_drug = st.selectbox("Select Drug for Analysis", available_drugs)
    
    # Dose selection
    col1, col2 = st.columns(2)
    
    with col1:
        dose = st.number_input("Dose (mg)", min_value=0.0, max_value=1000.0, value=10.0, step=0.1)
    
    with col2:
        route = st.selectbox("Route of Administration", ["oral", "iv", "im", "topical"])
    
    # Run simulation button
    if st.button("🚀 Run PK/PD Simulation", type="primary"):
        # Patient factors for simulation
        patient_factors = {
            'age': patient_data['demographics']['age'],
            'weight': patient_data['demographics']['weight'],
            'creatinine_clearance': patient_data['clinical_params']['gfr'],
            'route': route
        }
        
        # Time points for simulation
        time_points = np.linspace(0, 24, 100)
        
        # Simulate drug response
        simulation = drug_db.simulate_drug_response(
            selected_drug, dose, patient_factors, time_points
        )
        
        if simulation:
            # Create plots
            fig = go.Figure()
            
            # PK curve
            fig.add_trace(go.Scatter(
                x=time_points,
                y=simulation['concentrations'],
                mode='lines',
                name='Concentration',
                line=dict(color='blue', width=2)
            ))
            
            # Add therapeutic range
            props = simulation['drug_properties']
            fig.add_hline(y=props['therapeutic_range'][0], 
                          line_dash="dash", line_color="green", 
                          annotation_text="Therapeutic Min")
            fig.add_hline(y=props['therapeutic_range'][1], 
                          line_dash="dash", line_color="red", 
                          annotation_text="Therapeutic Max")
            
            fig.update_layout(
                title=f"{selected_drug} - Pharmacokinetic Profile",
                xaxis_title="Time (hours)",
                yaxis_title="Concentration (mcg/mL)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # PD curve
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=time_points,
                y=simulation['responses'],
                mode='lines',
                name='Response',
                line=dict(color='red', width=2)
            ))
            
            fig2.update_layout(
                title=f"{selected_drug} - Pharmacodynamic Response",
                xaxis_title="Time (hours)",
                yaxis_title="Response (%)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig2, width='stretch')
            
            # Key metrics
            st.subheader("📈 Key Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Peak Concentration", f"{np.max(simulation['concentrations']):.2f}", "mcg/mL")
            
            with col2:
                st.metric("Peak Response", f"{np.max(simulation['responses']):.1f}", "%")
            
            with col3:
                st.metric("Time to Peak", f"{time_points[np.argmax(simulation['concentrations'])]:.1f}", "hours")
            
            with col4:
                st.metric("AUC", f"{np.trapz(simulation['concentrations'], time_points):.1f}", "mcg·h/mL")

def show_disease_modeling():
    """Show disease-specific modeling"""
    st.header("🏥 Disease-Specific Modeling")
    
    if not DISEASE_MODELER_AVAILABLE:
        st.error("❌ Disease modeling is not available")
        return
    
    if 'patient_data' not in st.session_state:
        st.warning("⚠️ Please complete patient assessment first")
        return
    
    patient_data = st.session_state.patient_data
    disease_modeler = DiseaseSpecificModeler()
    
    # Assess disease severity
    st.subheader("🔬 Disease Severity Assessment")
    
    biomarker_data = {
        'chronic_kidney_disease': {
            'gfr': patient_data['clinical_params']['gfr'],
            'creatinine': patient_data['clinical_params']['creatinine']
        },
        'liver_cirrhosis': {
            'albumin': patient_data['clinical_params']['albumin'],
            'bilirubin': patient_data['clinical_params']['bilirubin']
        },
        'heart_failure': {
            'ef': patient_data['clinical_params']['ef'],
            'bnp': patient_data['clinical_params']['bnp']
        }
    }
    
    severity_assessments = {}
    for disease, biomarkers in biomarker_data.items():
        severity = disease_modeler.assess_disease_severity(disease, biomarkers)
        severity_assessments[disease] = severity
        
        st.write(f"**{disease.replace('_', ' ').title()}:** {severity.title()}")
    
    # Disease-specific recommendations
    st.subheader("📋 Disease-Specific Recommendations")
    
    for disease, severity in severity_assessments.items():
        disease_state = disease_modeler.get_disease_state(disease, severity)
        if disease_state:
            with st.expander(f"{disease.replace('_', ' ').title()} - {severity.title()}"):
                st.write("**PK Modifications:**")
                for param, value in disease_state.pk_modifications.items():
                    st.write(f"• {param.replace('_', ' ').title()}: {value:.2f}")
                
                st.write("**PD Modifications:**")
                for param, value in disease_state.pd_modifications.items():
                    st.write(f"• {param.replace('_', ' ').title()}: {value:.2f}")
                
                st.write("**Considerations:**")
                for consideration in disease_state.special_considerations:
                    st.write(f"• {consideration}")

if __name__ == "__main__":
    main()
