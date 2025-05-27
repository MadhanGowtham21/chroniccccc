import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Health Predictor", layout="wide")

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("💡 Dashboard Sections")
section = st.sidebar.radio("Go to", [
    "User Dashboard",
    "Prediction Dashboard",
    "Analytics Dashboard",
    "Doctor Appointment",
    "Medication Tracker",  # ← New
    "Nutrition & Diet Planner", # New
    "Health Profile Management",
    "Fitness & Activity Log",
    "Chat with AI Health Assistant",
])

# -----------------------------
# Helper Functions
# -----------------------------
def load_sample_data():
    return pd.DataFrame({
        "Date": ["2024-01-01", "2024-02-10", "2024-03-15"],
        "Prediction": ["Normal", "At Risk", "Critical"],
        "Disease": ["Diabetes", "BP", "Heart"],
        "Confidence": ["92%", "75%", "64%"]
    })

def yes_no_input(label):
    return 1 if st.selectbox(label, ["No", "Yes"]) == "Yes" else 0

@st.cache_data
def load_model(disease):
    path = f"models/model_{disease}.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"No model found for {disease}")
        st.stop()

# -----------------------------
# User Dashboard
# -----------------------------
if section == "User Dashboard":
    st.title("👤 User Dashboard")
    st.markdown("Manage your health records and predictions.")

    # Profile
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://www.w3schools.com/howto/img_avatar.png", width=100)
    with col2:
        st.subheader("Welcome, John Doe!")
        st.text("Email: john@example.com")
        st.text("Member since: Jan 2024")

    st.divider()

    # Upload Medical Records
    st.header("📁 Upload / View Medical Records")
    uploaded_file = st.file_uploader("Upload your medical report (PDF or CSV)", type=["pdf", "csv"])
    if uploaded_file:
        st.success(f"{uploaded_file.name} uploaded successfully!")
        if uploaded_file.name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    st.divider()

    # Prediction History
    st.header("📊 Recent Predictions History")
    history_df = load_sample_data()
    st.dataframe(history_df)

    st.divider()

    # Health Tips
    st.header("💡 Personalized Health Tips")
    st.info("Stay hydrated and monitor your sugar levels daily.")
    st.success("Daily 30-minute walk can reduce heart risk.")

    st.divider()

    # Reminder
    st.header("⏰ Set Health Reminders")
    reminder_date = st.date_input("Select date for next checkup")
    reminder_note = st.text_input("Reminder Note", "e.g., Blood test appointment")
    if st.button("Set Reminder"):
        st.success(f"Reminder set for {reminder_date}: {reminder_note}")

    st.divider()

    # Export Report
    st.header("📤 Export Health Report")
    export_format = st.selectbox("Choose export format", ["PDF", "CSV"])
    if st.button("Export"):
        st.success(f"Report will be generated in {export_format} format (Mock Functionality).")

# -----------------------------
# Prediction Dashboard
# -----------------------------
elif section == "Prediction Dashboard":
    st.header("🔮 Disease Prediction Dashboard")

    disease = st.selectbox("Select Disease to Predict", [
        "asthma", "cardiovascular", "ckd", "copd", "diabetes", "hypertension", "liver"
    ])

    model_data = load_model(disease)
    model = model_data["model"]
    scaler = model_data["scaler"]
    label_encoders = model_data["label_encoders"]
    feature_selector = model_data["feature_selector"]

    st.subheader(f"Input Patient Data for {disease.title()} Prediction")

    # Basic inputs
    age = st.slider("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    gender_encoded = 1 if gender == "Male" else 0

    bmi = st.slider("BMI", 10.0, 50.0, 22.0)
    systolic_bp = st.slider("Systolic BP", 80, 200, 120)
    diastolic_bp = st.slider("Diastolic BP", 50, 130, 80)
    blood_sugar = st.slider("Blood Sugar (mg/dL)", 50, 400, 100)
    cholesterol = st.slider("Cholesterol (mg/dL)", 100, 400, 180)
    smoking = yes_no_input("Smoking?")
    alcohol = yes_no_input("Alcohol Consumption?")
    physical_activity = yes_no_input("Physical Activity?")
    family_history = yes_no_input("Family History of Disease?")

    # Conditional advanced inputs
    with st.expander("🔬 Advanced Medical Inputs"):
        creatinine = st.slider("Creatinine", 0.2, 10.0, 1.0)
        urea = st.slider("Urea", 5.0, 100.0, 20.0)
        ast = st.slider("AST", 0, 100, 20)
        alt = st.slider("ALT", 0, 100, 20)
        fev1 = st.slider("FEV1", 0.1, 10.0, 3.0)
        breathing_difficulty = yes_no_input("Breathing Difficulty?")
        cough_frequency = st.slider("Cough Frequency (times per day)", 0, 20, 0)
        chest_pain = yes_no_input("Chest Pain?")

    input_dict = {
        "Age": age, "Gender": gender_encoded, "BMI": bmi,
        "Systolic_BP": systolic_bp, "Diastolic_BP": diastolic_bp,
        "Blood_Sugar": blood_sugar, "Cholesterol": cholesterol,
        "Smoking": smoking, "Alcohol": alcohol,
        "Physical_Activity": physical_activity, "Family_History": family_history,
        "Creatinine": creatinine, "Urea": urea,
        "AST": ast, "ALT": alt, "FEV1": fev1,
        "Breathing_Difficulty": breathing_difficulty,
        "Cough_Frequency": cough_frequency, "Chest_Pain": chest_pain
    }

    df_input = pd.DataFrame([input_dict])

    # Encode categorical fields
    for col, le in label_encoders.items():
        if col in df_input.columns and col != "Gender":
            try:
                df_input[col] = le.transform(df_input[col].astype(str))
            except Exception:
                df_input[col] = -1

    X_scaled = scaler.transform(df_input)
    X_selected = feature_selector.transform(X_scaled)

    if st.button("Predict"):
        prediction_proba = model.predict_proba(X_selected)[0]
        prediction_class = model.predict(X_selected)[0]

        risk_map = {
            0: ("✅ Normal", "green"),
            1: ("⚠️ At Risk", "orange"),
            2: ("🔴 Critical", "red")
        }

        status_text, status_color = risk_map.get(prediction_class, ("Unknown", "black"))

        st.markdown(f"### Prediction Result: <span style='color:{status_color}'>{status_text}</span>",
                    unsafe_allow_html=True)
        st.write(f"Model confidence: {prediction_proba[prediction_class] * 100:.2f}%")

        if prediction_class == 0:
            st.success("No immediate concerns. Maintain healthy lifestyle.")
        elif prediction_class == 1:
            st.warning("Moderate risk. Consider consulting a healthcare provider.")
        else:
            st.error("High risk! Please seek medical attention immediately.")

        # ✅ Store predicted disease in session state for Nutrition & Diet Planner
        st.session_state["predicted_disease"] = disease



# -----------------------------
# Analytics Dashboard
# -----------------------------
elif section == "Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    st.markdown("Visual summary of usage, model performance, and prediction trends.")

    try:
        df = pd.read_csv("prediction_logs.csv", parse_dates=["date"])
    except FileNotFoundError:
        st.warning("No analytics data available yet.")
    else:
        st.sidebar.subheader("🔎 Filter Options")
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        date_range = st.sidebar.date_input("Select date range", [min_date, max_date])

        selected_diseases = st.sidebar.multiselect("Select diseases", options=df['prediction'].unique(),
                                                   default=df['prediction'].unique())
        age_range = st.sidebar.slider("Select age range", int(df['age'].min()), int(df['age'].max()), (20, 60))

        filtered_df = df[
            (df['date'].dt.date.between(date_range[0], date_range[1])) &
            (df['prediction'].isin(selected_diseases)) &
            (df['age'].between(age_range[0], age_range[1]))
        ]

        st.subheader("📈 Usage Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Users", filtered_df['user_id'].nunique() if 'user_id' in df.columns else "N/A")
        col2.metric("Total Predictions", len(filtered_df))
        col3.metric("Unique Diseases Detected", filtered_df['prediction'].nunique())

        st.subheader("📊 Prediction Trends Over Time")
        trend_df = filtered_df.groupby(['date', 'prediction']).size().reset_index(name='count')
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=trend_df, x='date', y='count', hue='prediction', ax=ax1)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        st.pyplot(fig1)

        st.subheader("⚖️ Prediction Distribution")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.countplot(data=filtered_df, x='prediction', ax=ax2)
        st.pyplot(fig2)

# -----------------------------
# Doctor Appointment
# -----------------------------
elif section == "Doctor Appointment":
    st.title("🩺 Book a Doctor Appointment")
    st.markdown("Easily schedule a consultation with a healthcare professional.")

    st.subheader("🔍 Choose Specialization")
    specialization = st.selectbox("Specialist", [
        "General Physician", "Cardiologist", "Diabetologist", "Pulmonologist", "Nephrologist", "Gastroenterologist"
    ])

    doctor_map = {
        "General Physician": ["Dr. A. Kumar", "Dr. S. Nair"],
        "Cardiologist": ["Dr. R. Sharma", "Dr. M. Iyer"],
        "Diabetologist": ["Dr. N. Patel", "Dr. T. Rao"],
        "Pulmonologist": ["Dr. F. Thomas", "Dr. D. George"],
        "Nephrologist": ["Dr. L. Bose", "Dr. K. Mehta"],
        "Gastroenterologist": ["Dr. B. Singh", "Dr. V. Khan"]
    }

    st.subheader("👨‍⚕️ Choose Doctor")
    selected_doctor = st.selectbox("Available Doctors", doctor_map[specialization])

    st.subheader("📅 Choose Date and Time")
    appt_date = st.date_input("Select Appointment Date")
    appt_time = st.time_input("Select Time")

    patient_name = st.text_input("Your Name")
    contact = st.text_input("Contact Number")

    if st.button("Book Appointment"):
        if patient_name and contact:
            st.success(f"Appointment booked with **{selected_doctor}** on **{appt_date}** at **{appt_time}**.")
        else:
            st.error("Please fill all the required details.")

elif section == "Medication Tracker":
    def medication_tracker():
        st.title("💊 Medication Tracker")
        st.write("Track and manage your medications easily.")

        if "medications" not in st.session_state:
            st.session_state.medications = []

        with st.form("med_form"):
            col1, col2 = st.columns(2)
            with col1:
                med_name = st.text_input("Medication Name")
                dosage = st.text_input("Dosage (e.g. 500mg)")
            with col2:
                time = st.selectbox("Time of Day", ["Morning", "Afternoon", "Night"])
                start_date = st.date_input("Start Date")
                end_date = st.date_input("End Date")

            submitted = st.form_submit_button("Add Medication")
            if submitted:
                st.session_state.medications.append({
                    "Medication": med_name,
                    "Dosage": dosage,
                    "Time": time,
                    "Start Date": start_date.strftime("%Y-%m-%d"),
                    "End Date": end_date.strftime("%Y-%m-%d")
                })
                st.success(f"{med_name} added to tracker.")

        if st.session_state.medications:
            st.subheader("📋 Current Medications")
            df = pd.DataFrame(st.session_state.medications)
            st.dataframe(df, use_container_width=True)

    # Call the function
    medication_tracker()


elif section == "Nutrition & Diet Planner":
    def nutrition_and_diet():
        st.title("🥗 Nutrition & Diet Planner")
        st.write("Personalized diet recommendations based on your latest health prediction.")

        # Update button to refresh prediction data
        if st.button("🔄 Update Diet Plan"):
            st.session_state["use_latest_disease"] = True

        # Default disease value (None if not predicted yet)
        predicted_disease = st.session_state.get("predicted_disease", None) if st.session_state.get("use_latest_disease") else None

        st.subheader("🍽️ Your Recommended Diet")

        if predicted_disease:
            st.success(f"Detected Disease: **{predicted_disease}**")
        else:
            st.info("No disease detected. Showing general healthy diet.")

        # Diet mapping
        diet_recommendations = {
            "Diabetes": [
                "Whole grains (brown rice, oats)",
                "Leafy greens and non-starchy vegetables",
                "High-fiber fruits (apples, berries)",
                "Lean protein (chicken, fish, tofu)",
                "Avoid sugary drinks and refined carbs"
            ],
            "Hypertension": [
                "Low-sodium foods",
                "Leafy vegetables (spinach, kale)",
                "Fruits rich in potassium (bananas, oranges)",
                "Whole grains, nuts, seeds",
                "Avoid salty snacks and processed meats"
            ],
            "Heart Disease": [
                "Oats, barley, whole grains",
                "Berries, avocados, leafy greens",
                "Nuts and seeds (in moderation)",
                "Olive oil, fish rich in omega-3",
                "Avoid saturated fat and fried foods"
            ],
            "Psoriasis": [
                "Omega-3 rich fish (salmon, mackerel)",
                "Colorful vegetables (carrots, spinach)",
                "Whole grains and nuts",
                "Limit red meat, dairy, and processed foods"
            ]
        }

        # Default general diet
        general_diet = [
            "Fruits and vegetables (5 servings/day)",
            "Whole grains (brown rice, quinoa, oats)",
            "Lean proteins (chicken, fish, legumes)",
            "Plenty of water (8 glasses/day)",
            "Limit sugar, salt, and trans fats"
        ]

        # Show relevant diet plan
        if predicted_disease in diet_recommendations:
            for item in diet_recommendations[predicted_disease]:
                st.markdown(f"- {item}")
        else:
            for item in general_diet:
                st.markdown(f"- {item}")

    nutrition_and_diet()


# -----------------------------
# Coming Soon Section
# -----------------------------

elif section == "Health Profile Management":
    st.header("🩺 Health Profile Management")

    # Initialize or load profile data from session state or persistent storage
    if "health_profile" not in st.session_state:
        st.session_state["health_profile"] = {
            "Name": "",
            "Age": 0,
            "Weight": 0.0,
            "Height": 0.0,
            "Allergies": "",
            "Existing Conditions": "",
            "Vitals History": []  # To store vitals trend over time
        }

    profile = st.session_state["health_profile"]

    st.subheader("Update Your Personal Information")
    profile["Name"] = st.text_input("Name", value=profile["Name"])
    profile["Age"] = st.number_input("Age", min_value=0, max_value=120, value=profile["Age"])
    profile["Weight (kg)"] = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=profile["Weight"])
    profile["Height (cm)"] = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=profile["Height"])
    profile["Allergies"] = st.text_area("Allergies", value=profile["Allergies"])
    profile["Existing Conditions"] = st.text_area("Existing Conditions", value=profile["Existing Conditions"])

    st.subheader("Add Basic Vitals for Trend Monitoring")
    col1, col2, col3 = st.columns(3)
    with col1:
        bp_systolic = st.number_input("Systolic BP", min_value=50, max_value=250, step=1)
    with col2:
        bp_diastolic = st.number_input("Diastolic BP", min_value=30, max_value=150, step=1)
    with col3:
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, step=1)

    if st.button("Add Vitals Reading"):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        vitals_entry = {
            "timestamp": timestamp,
            "Systolic_BP": bp_systolic,
            "Diastolic_BP": bp_diastolic,
            "Heart_Rate": heart_rate
        }
        profile["Vitals History"].append(vitals_entry)
        st.success(f"Vitals added at {timestamp}")

    if profile["Vitals History"]:
        st.subheader("Vitals Trend History")
        vitals_df = pd.DataFrame(profile["Vitals History"])
        st.line_chart(vitals_df.set_index("timestamp")[["Systolic_BP", "Diastolic_BP", "Heart_Rate"]])

    # Save back to session state
    st.session_state["health_profile"] = profile

    st.info("Future: Integration with wearable devices for automatic vitals tracking.")

elif section == "Fitness & Activity Log":
    st.header("🏃 Fitness & Activity Log")

    # Load health profile and disease prediction from session state (or set defaults)
    profile = st.session_state.get("health_profile", {"Age": 30})
    predicted_disease = st.session_state.get("last_prediction_disease", None)  # store disease predicted elsewhere

    st.subheader("Log Your Daily Activities")

    date = st.date_input("Date", value=pd.Timestamp.now().date())
    walking_steps = st.number_input("Walking Steps", min_value=0, max_value=50000, value=0)
    running_minutes = st.number_input("Running Minutes", min_value=0, max_value=300, value=0)
    workouts_minutes = st.number_input("Workout Minutes", min_value=0, max_value=300, value=0)
    heart_rate = st.number_input("Average Heart Rate (bpm)", min_value=30, max_value=200, value=30)


    if "fitness_log" not in st.session_state:
        st.session_state["fitness_log"] = []

    if st.button("Add Activity Log"):
        entry = {
            "date": date.strftime("%Y-%m-%d"),
            "walking_steps": walking_steps,
            "running_minutes": running_minutes,
            "workouts_minutes": workouts_minutes,
            "heart_rate": heart_rate
        }
        st.session_state["fitness_log"].append(entry)
        st.success(f"Activity log for {entry['date']} added!")

    # Convert fitness log to DataFrame for visualization
    if st.session_state["fitness_log"]:
        fitness_df = pd.DataFrame(st.session_state["fitness_log"])
        fitness_df["date"] = pd.to_datetime(fitness_df["date"])

        st.subheader("Activity Progress Charts")

        # Line chart for steps and workouts over time
        st.line_chart(fitness_df.set_index("date")[["walking_steps", "running_minutes", "workouts_minutes"]])

        # Bar chart for heart rate
        st.bar_chart(fitness_df.set_index("date")["heart_rate"])

    else:
        st.info("No activity logs yet. Add your daily activities above.")

    # Exercise suggestions based on age and disease
    st.subheader("Suggested Exercises")

    age = profile.get("Age", 30)
    disease = predicted_disease if predicted_disease else "none"

    def suggest_exercises(age, disease):
        # Basic example suggestions, expand as needed
        suggestions = []

        if disease == "asthma":
            suggestions = [
                "Light walking daily",
                "Breathing exercises",
                "Yoga and stretching"
            ]
        elif disease == "cardiovascular":
            suggestions = [
                "Moderate walking or cycling",
                "Low-impact aerobics",
                "Strength training with light weights"
            ]
        elif disease == "diabetes":
            suggestions = [
                "Brisk walking",
                "Resistance training",
                "Flexibility exercises"
            ]
        else:
            # General suggestions by age groups
            if age < 30:
                suggestions = ["Running", "HIIT workouts", "Strength training"]
            elif age < 50:
                suggestions = ["Walking", "Yoga", "Moderate cardio"]
            else:
                suggestions = ["Walking", "Stretching", "Balance exercises"]

        return suggestions

    exercises = suggest_exercises(age, disease)
    for ex in exercises:
        st.markdown(f"- {ex}")

elif section == "Chat with AI Health Assistant":
    st.header("🤖 Chat with AI Health Assistant")

    st.write("Ask about your disease prediction results, get health tips, schedule appointments, or FAQs about diseases.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    disease_faqs = {
        "asthma": {
            "symptoms": "Common asthma symptoms include wheezing, coughing, shortness of breath, and chest tightness.",
            "prevention": "Avoid allergens, use inhalers as prescribed, and monitor air quality to prevent asthma attacks.",
            "treatment": "Treatment usually involves inhalers, bronchodilators, and avoiding triggers."
        },
        "cardiovascular": {
            "symptoms": "Symptoms include chest pain, shortness of breath, palpitations, and fatigue.",
            "prevention": "Maintain healthy diet, exercise regularly, avoid smoking, and control blood pressure.",
            "treatment": "Treatment can include medications, lifestyle changes, and sometimes surgical interventions."
        },
        "ckd": {
            "symptoms": "Symptoms of Chronic Kidney Disease include fatigue, swelling in legs, and changes in urination.",
            "prevention": "Control blood pressure and diabetes, maintain hydration, and avoid nephrotoxic drugs.",
            "treatment": "Treatment focuses on managing symptoms and slowing progression, sometimes dialysis."
        },
        "copd": {
            "symptoms": "COPD symptoms include chronic cough, mucus production, and difficulty breathing.",
            "prevention": "Avoid smoking, reduce exposure to lung irritants, and get vaccinated against flu and pneumonia.",
            "treatment": "Inhalers, oxygen therapy, and pulmonary rehabilitation are common treatments."
        },
        "diabetes": {
            "symptoms": "Symptoms include increased thirst, frequent urination, fatigue, and blurred vision.",
            "prevention": "Maintain healthy weight, exercise regularly, and monitor blood sugar levels.",
            "treatment": "Includes insulin therapy, oral medications, diet, and exercise."
        },
        "hypertension": {
            "symptoms": "Often called the silent killer, hypertension may show no symptoms until complications arise.",
            "prevention": "Eat low salt diet, exercise, manage stress, and avoid excessive alcohol.",
            "treatment": "Medications, lifestyle changes, and regular monitoring are important."
        },
        "liver": {
            "symptoms": "Liver disease symptoms include jaundice, fatigue, nausea, and abdominal pain.",
            "prevention": "Avoid excessive alcohol, vaccinate for hepatitis, and maintain healthy weight.",
            "treatment": "Depends on cause; may include medications, lifestyle changes, or surgery."
        }
    }

    def generate_response(user_input):
        user_input = user_input.lower()

        # Explain prediction results
        if any(word in user_input for word in ["prediction", "result", "risk", "status"]):
            return ("Your prediction results show your current health risk level for specific diseases. "
                    "Use these insights to consult healthcare professionals for personalized care.")

        # Health tips
        if any(word in user_input for word in ["tip", "advice", "healthy", "prevent", "diet", "exercise"]):
            return ("Maintain a balanced diet rich in fruits and vegetables, stay active with regular exercise, "
                    "avoid smoking and excessive alcohol, and manage stress effectively.")

        # Appointment scheduling
        if any(word in user_input for word in ["schedule", "appointment", "book", "consult"]):
            return ("To schedule an appointment, please contact your healthcare provider or use the app's appointment feature (coming soon).")

        # FAQs by disease
        for disease, faqs in disease_faqs.items():
            if disease in user_input:
                if "symptom" in user_input:
                    return faqs["symptoms"]
                elif "prevent" in user_input or "prevention" in user_input:
                    return faqs["prevention"]
                elif "treat" in user_input or "treatment" in user_input or "manage" in user_input:
                    return faqs["treatment"]
                else:
                    return (f"You asked about {disease.title()}. "
                            "You can ask about symptoms, prevention, or treatment.")

        # General disease FAQs fallback
        if any(word in user_input for word in ["disease", "health", "illness", "condition"]):
            return ("I can help with information about asthma, cardiovascular diseases, CKD, COPD, diabetes, hypertension, and liver diseases. "
                    "Please specify the disease or ask about symptoms, prevention, or treatment.")

        # Default response
        return ("I'm here to help! You can ask me about your prediction results, health tips, appointment scheduling, "
                "or disease-related FAQs.")

    user_input = st.text_input("You:", key="user_input")

    if st.button("Send") and user_input.strip() != "":
        st.session_state.chat_history.append(("You", user_input))
        bot_response = generate_response(user_input)
        st.session_state.chat_history.append(("AI Assistant", bot_response))

    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**AI Assistant:** {message}")




