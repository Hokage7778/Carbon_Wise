import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, db
from PIL import Image
import base64
import json
import os
import re
from datetime import datetime
from huggingface_hub import InferenceClient
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import requests
import json

# ------------------------------------------------------------------
# Configuration
SERVICE_ACCOUNT_PATH = "Service.json"
HF_API_KEY = "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"
FIREBASE_API_KEY = "AIzaSyC3aC_hW9he4VoG_lv3AFUWHVbJbRYNGq4"  # Add your Firebase Web API key here
# ------------------------------------------------------------------
# Firebase Initialization
def initialize_firebase():
    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://sample-project-050225-default-rtdb.firebaseio.com'
            })
            return True
        except Exception as e:
            st.error(f"Failed to initialize Firebase: {str(e)}")
            return False
    return True

# ------------------------------------------------------------------
# Helper function to safely rerun the app
def rerun_app():
    try:
        st.experimental_rerun()
    except AttributeError:
        st.stop()

# ------------------------------------------------------------------
# Function to fetch and aggregate CO₂ savings from Firebase for the logged-in user
def get_co2_summary(user_uid):
    summary = {"Today": 0, "This Week": 0, "This Month": 0, "This Year": 0, "Overall": 0}
    try:
        ref = db.reference(f'users/{user_uid}/activities')
        activities = ref.get()
        if activities:
            now = datetime.now()
            for key, activity in activities.items():
                timestamp_str = activity.get("timestamp", None)
                if timestamp_str:
                    try:
                        dt = datetime.fromisoformat(timestamp_str)
                    except Exception:
                        continue
                    co2 = activity.get("activity_details", {}).get("co2_savings", 0)
                    summary["Overall"] += co2
                    if dt.date() == now.date():
                        summary["Today"] += co2
                    if dt.isocalendar()[1] == now.isocalendar()[1] and dt.year == now.year:
                        summary["This Week"] += co2
                    if dt.month == now.month and dt.year == now.year:
                        summary["This Month"] += co2
                    if dt.year == now.year:
                        summary["This Year"] += co2
        return summary
    except Exception as e:
        print("Error fetching CO2 summary:", e)
        return summary

# ------------------------------------------------------------------
# Function to generate trivia based on overall CO₂ savings
def generate_trivia(overall_co2):
    trees = overall_co2 / 22.0
    tree_count = max(1, int(trees)) if overall_co2 >= 1 else "less than 1"
    km = overall_co2 / 0.2
    days_not_driven = overall_co2 / (0.2 * 30)
    bulbs = overall_co2 / 0.12

    trivia = (
        f"💡 **Did You Know?** Your overall CO₂ savings of **{overall_co2:.0f} kg** is equivalent to:\n\n"
        f"- The annual CO₂ absorption of about **{tree_count} mature tree{'s' if tree_count != 1 else ''}**.\n"
        f"- Not driving approximately **{int(km):,} km**.\n"
        f"- Avoiding driving for around **{int(days_not_driven)} days** (assuming 30 km per day).\n"
        f"- Powering roughly **{int(bulbs):,} LED bulbs** for one day."
    )
    return trivia
# ------------------------------------------------------------------
# Image Processing Functions
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def describe_image(image_path):
    try:
        client = InferenceClient(api_key=HF_API_KEY)
        image_b64 = encode_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": (
                         "Analyze this image in detail for eco-friendly activities. "
                         "Look for any evidence of recycling (e.g., reusable items or PCR content), "
                         "public transport usage (e.g., train tickets), or exercise (walking/cycling). "
                         "Provide measurements and numbers where visible."
                     )},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ]
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct",
            messages=messages,
            max_tokens=500
        )
        result = completion.choices[0].message.content.strip()
        return result
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# ------------------------------------------------------------------
# Helper to clean irrelevant information from Vision API output
def clean_vision_output(text):
    irrelevant_keywords = ["job postings", "implementation assistant", "junior consultant", "SRE"]
    lines = text.splitlines()
    cleaned = [line for line in lines if not any(keyword.lower() in line.lower() for keyword in irrelevant_keywords)]
    return "\n".join(cleaned)
# ------------------------------------------------------------------
# Function to calculate CO₂ savings based on extracted metrics
def calculate_co2_savings(activity_type, metrics):
    try:
        savings = 0
        atype = activity_type.lower()
        if "recycling" in atype:
            savings += metrics.get("pcr_percentage", 0) * 0.5
            savings += metrics.get("reusable_items", 0) * 500 * 0.033
        elif "public transport" in atype or "train" in atype or "ticket" in atype:
            distance = metrics.get("distance_km", 0)
            savings = distance * (0.2 - 0.04)
        elif "exercise" in atype or "walking" in atype or "cycling" in atype:
            distance = metrics.get("distance_km", 0)
            savings = distance * 0.2
        return max(savings, 0)
    except Exception as e:
        print(f"CO₂ calculation error: {str(e)}")
        return 0

# ------------------------------------------------------------------
# Main processing function using LLM for extraction and fallback regex as needed.
def process_with_langchain(vision_output):
    try:
        prompt = PromptTemplate(
            input_variables=["vision_output"],
            template="""
Analyze the following text and extract eco-friendly activity details.
Output one key-value pair per line in the following format:
Activity Type: [Recycling/Exercise/Public Transport/None]
Items Description: [detailed description or empty string]
PCR Percentage: [number or 0]
Reusable Items: [number or 0]
Single-Use Items Saved: [number or 0]
Distance (km): [number or 0]
Duration (min): [number or 0]
Source: [text or empty string]
Destination: [text or empty string]
Additional Notes: [text or empty string]

Image Description:
{vision_output}
            """
        )
        llm = HuggingFaceHub(
            repo_id="Qwen/Qwen2.5-72B-Instruct",
            huggingfacehub_api_token=HF_API_KEY
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(vision_output)
        print(f"LLM Response:\n{response}")

        activity_details = {
            "activity_type": "Not Available",
            "items_description": "",
            "pcr_percentage": 0,
            "reusable_items": 0,
            "single_use_saved": 0,
            "distance_km": 0,
            "time_duration": 0,
            "source": "",
            "destination": "",
            "additional_notes": "",
            "co2_savings": 0
        }

        # ... (rest of the processing logic remains the same)

        return activity_details

    except Exception as e:
        print(f"Processing Error: {str(e)}")
        return {
            "activity_type": "Not Available",
            "items_description": "",
            "pcr_percentage": 0,
            "reusable_items": 0,
            "single_use_saved": 0,
            "distance_km": 0,
            "time_duration": 0,
            "source": "",
            "destination": "",
            "additional_notes": "",
            "co2_savings": 0
        }
# ------------------------------------------------------------------
# Dashboard Display (with Firebase integration)
def show_dashboard():
    with st.sidebar:
        st.markdown("### Account")
        st.write(f"**Logged in as:** {st.session_state.user['email']}")
        if st.button("Sign Out"):
            st.session_state.logged_in = False
            st.session_state.user = None
            rerun_app()

    st.title("📊 CarbonWise")
    st.markdown("Upload an image to analyze eco-friendly activities and track your CO₂ savings.")

    # Rest of the dashboard display code remains the same...

# ------------------------------------------------------------------
# Main Function with Authentication and Dashboard routing
def main():
    st.set_page_config(page_title="CarbonWise", page_icon="🌱", layout="centered")
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 800px;
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user = None
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False

    if not initialize_firebase():
        st.error("Firebase initialization failed.")
        return

    # If logged in, show the dashboard
    if st.session_state.logged_in:
        show_dashboard()
        return

    # Authentication Page
    with st.container():
        st.markdown("<h1 style='text-align: center;'>CarbonWise</h1>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        if st.session_state.show_signup:
            # Signup form code...
            pass
        else:
            with col1:
                st.subheader("Login")
                with st.form("login_form", clear_on_submit=False):
                    login_email = st.text_input("Email", key="login_email")
                    login_password = st.text_input("Password", type="password", key="login_password")
                    submit_login = st.form_submit_button("Login")

                    if submit_login:
                        try:
                            # Use Firebase REST API to verify password
                            auth_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
                            data = {
                                "email": login_email,
                                "password": login_password,
                                "returnSecureToken": True
                            }
                            response = requests.post(auth_url, data=json.dumps(data))
                            result = response.json()

                            if 'error' in result:
                                st.error(f"Login failed: {result['error']['message']}")
                            else:
                                # Verify the ID token using Admin SDK
                                decoded_token = auth.verify_id_token(result['idToken'])
                                user = auth.get_user(decoded_token['uid'])

                                st.session_state.logged_in = True
                                st.session_state.user = {
                                    "email": user.email,
                                    "uid": user.uid
                                }
                                st.success("✅ Logged in successfully!")
                                rerun_app()

                        except Exception as e:
                            st.error(f"Login error: {str(e)}")

            with col2:
                st.markdown("### Welcome Back!")
                st.write("Log in to view your CarbonWise dashboard and check your CO₂ savings.")
                if st.button("Don't have an account? Sign Up", key="signup_button"):
                    st.session_state.show_signup = True

if __name__ == '__main__':
    main()
