import streamlit as st
import requests
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

# ------------------------------------------------------------------
# Configuration
FIREBASE_WEB_API_KEY = "AIzaSyC3aC_hW9he4VoG_lv3AFUWHVbJbRYNGq4"  # Replace with your Web API Key
FIREBASE_DATABASE_URL = "https://sample-project-050225-default-rtdb.firebaseio.com"
HF_API_KEY = "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"

# ------------------------------------------------------------------
# Firebase REST API Functions

def firebase_request(endpoint, method='GET', data=None, params=None):
    """Generic function for Firebase REST API requests"""
    url = f"{FIREBASE_DATABASE_URL}/{endpoint}.json"
    if params is None:
        params = {}
    if "auth" in st.session_state:
        params["auth"] = st.session_state.auth  # idToken from login
    try:
        if method == 'GET':
            response = requests.get(url, params=params)
        elif method == 'POST':
            response = requests.post(url, json=data, params=params)
        elif method == 'PUT':
            response = requests.put(url, json=data, params=params)
        else:
            raise ValueError("Unsupported HTTP method.")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print("Firebase request error:", e)
        return None

def firebase_sign_up(email, password):
    """Sign up a new user using Firebase Authentication REST API"""
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_WEB_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    try:
        res = requests.post(url, json=payload)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print("Sign up error:", e)
        return None

def firebase_sign_in(email, password):
    """Sign in an existing user using Firebase Authentication REST API"""
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    try:
        res = requests.post(url, json=payload)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print("Sign in error:", e)
        return None

# ------------------------------------------------------------------
# Function to fetch and aggregate CO₂ savings from Firebase for the logged-in user
def get_co2_summary(user_uid):
    summary = {"Today": 0, "This Week": 0, "This Month": 0, "This Year": 0, "Overall": 0}
    try:
        activities = firebase_request(f'users/{user_uid}/activities')
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
        allowed_keys = {
            "Activity Type",
            "Items Description",
            "PCR Percentage",
            "Reusable Items",
            "Single-Use Items Saved",
            "Distance (km)",
            "Duration (min)",
            "Source",
            "Destination",
            "Additional Notes"
        }

        for line in response.split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, value = [x.strip() for x in line.split(':', 1)]
            if key not in allowed_keys:
                continue
            if key == "Activity Type":
                activity_details["activity_type"] = value
            elif key == "Items Description":
                activity_details["items_description"] = value
            elif key == "PCR Percentage":
                try:
                    activity_details["pcr_percentage"] = float(value.split()[0])
                except:
                    activity_details["pcr_percentage"] = 0
            elif key == "Reusable Items":
                try:
                    activity_details["reusable_items"] = int(value.split()[0])
                except:
                    activity_details["reusable_items"] = 0
            elif key == "Single-Use Items Saved":
                try:
                    activity_details["single_use_saved"] = int(value.split()[0])
                except:
                    activity_details["single_use_saved"] = 0
            elif key == "Distance (km)":
                try:
                    activity_details["distance_km"] = float(value.split()[0])
                except:
                    activity_details["distance_km"] = 0
            elif key == "Duration (min)":
                try:
                    activity_details["time_duration"] = int(value.split()[0])
                except:
                    activity_details["time_duration"] = 0
            elif key == "Source":
                activity_details["source"] = value
            elif key == "Destination":
                activity_details["destination"] = value
            elif key == "Additional Notes":
                activity_details["additional_notes"] = value

        # --- Fallback Extraction from Raw Vision Output ---
        if activity_details["distance_km"] == 0:
            m = re.search(r"(?i)distance.*?(\d+\.?\d*)\s*kilometers", vision_output, re.DOTALL)
            if m:
                activity_details["distance_km"] = float(m.group(1))
                print("Fallback: extracted distance =", activity_details["distance_km"])

        if activity_details["activity_type"] in ["Not Available", "[Recycling/Exercise/Public Transport]"]:
            vision_lower = vision_output.lower()
            if "reusable" in vision_lower or "pcr" in vision_lower or "recycled" in vision_lower:
                activity_details["activity_type"] = "Recycling"
            elif "train" in vision_lower or "ticket" in vision_lower or "public transport" in vision_lower:
                activity_details["activity_type"] = "Public Transport"
            elif "walking" in vision_lower or "cycling" in vision_lower or "exercise" in vision_lower:
                activity_details["activity_type"] = "Exercise"
            else:
                activity_details["activity_type"] = "None"

        if activity_details["activity_type"] == "Recycling" and activity_details["pcr_percentage"] == 0:
            m = re.search(r"(\d+)%\s*(pcr)", vision_output.lower())
            if m:
                activity_details["pcr_percentage"] = float(m.group(1))
                print("Fallback: extracted PCR percentage =", activity_details["pcr_percentage"])

        metrics = {
            "distance_km": activity_details["distance_km"],
            "reusable_items": activity_details["reusable_items"],
            "pcr_percentage": activity_details["pcr_percentage"],
            "single_use_saved": activity_details["single_use_saved"]
        }
        activity_details["co2_savings"] = calculate_co2_savings(activity_details["activity_type"], metrics)
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
# Helper function to safely rerun the app
def rerun_app():
    try:
        st.experimental_rerun()
    except AttributeError:
        st.stop()

# ------------------------------------------------------------------
# Dashboard Display (with Firebase REST API integration)
def show_dashboard():
    with st.sidebar:
        st.markdown("### Account")
        st.write(f"**Logged in as:** {st.session_state.user['email']}")
        if st.button("Sign Out"):
            st.session_state.logged_in = False
            if "auth" in st.session_state:
                del st.session_state.auth
            st.session_state.user = None
            rerun_app()

    st.title("📊 CarbonWise")
    st.markdown("Upload an image to analyze eco-friendly activities and track your CO₂ savings.")

    # Display CO₂ savings summary.
    if st.session_state.user:
        summary = get_co2_summary(st.session_state.user["uid"])
        cols = st.columns(5)
        cols[0].metric("Today", f"{round(summary['Today'])} kg")
        cols[1].metric("This Week", f"{round(summary['This Week'])} kg")
        cols[2].metric("This Month", f"{round(summary['This Month'])} kg")
        cols[3].metric("This Year", f"{round(summary['This Year'])} kg")
        cols[4].metric("Overall", f"{round(summary['Overall'])} kg")
        trivia = generate_trivia(summary["Overall"])
        st.markdown(trivia)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        temp_image_path = "temp_image.jpg"
        image.save(temp_image_path)

        st.info("⏳ Analyzing image... Please wait.")
        vision_output = describe_image(temp_image_path)
        if "Error" in vision_output:
            st.error(vision_output)
            return

        # Clean the Vision API output.
        vision_output_clean = clean_vision_output(vision_output)

        st.markdown("### 📝 Image Description")
        st.write(vision_output_clean)

        activity_details = process_with_langchain(vision_output)
        st.markdown("### 📊 Activity Details")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Activity Type:** {activity_details['activity_type']}")
            if activity_details['activity_type'].lower() == 'recycling':
                if activity_details['items_description']:
                    st.write(f"**Items:** {activity_details['items_description']}")
                if activity_details['pcr_percentage'] > 0:
                    st.write(f"**PCR Content:** {activity_details['pcr_percentage']}%")
                if activity_details['reusable_items'] > 0:
                    st.write(f"**Reusable Items:** {activity_details['reusable_items']}")
                if activity_details['single_use_saved'] > 0:
                    st.write(f"**Single-use Items Saved:** {activity_details['single_use_saved']}")
        with col2:
            if activity_details['distance_km'] > 0:
                st.write(f"**Distance:** {activity_details['distance_km']} km")
            if activity_details['time_duration'] > 0:
                st.write(f"**Duration:** {activity_details['time_duration']} minutes")
            if activity_details['source'] and activity_details['destination']:
                st.write(f"**Route:** {activity_details['source']} to {activity_details['destination']}")

        # --- Key Change: Display CO₂ Savings with one decimal place ---
        st.metric("CO₂ Savings", f"{activity_details['co2_savings']:.1f} kg")

        if activity_details['additional_notes']:
            st.info(f"📌 **Additional Notes:** {activity_details['additional_notes']}")

        # Save activity to Firebase.
        if st.session_state.user:
            data = {
                "timestamp": datetime.now().isoformat(),
                "image_description": vision_output,
                "activity_details": activity_details
            }
            res = firebase_request(f'users/{st.session_state.user["uid"]}/activities', method='POST', data=data)
            if res:
                st.success("✅ Activity logged successfully!")
            else:
                st.error("Failed to log activity.")
            rerun_app()

        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

# ------------------------------------------------------------------
# Main Function with Authentication and Dashboard routing using Firebase REST API
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

    # Initialize session state.
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user = None
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False

    # If logged in, show the dashboard.
    if st.session_state.logged_in:
        show_dashboard()
        return

    # Authentication Page
    with st.container():
        st.markdown("<h1 style='text-align: center;'>CarbonWise</h1>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        # Sign Up Flow
        if st.session_state.show_signup:
            with col1:
                st.subheader("Sign Up")
                with st.form("signup_form", clear_on_submit=True):
                    signup_email = st.text_input("Email", key="signup_email")
                    signup_password = st.text_input("Password", type="password", key="signup_password")
                    submit_signup = st.form_submit_button("Create Account")
                    if submit_signup:
                        result = firebase_sign_up(signup_email, signup_password)
                        if result and "error" not in result:
                            st.success("✅ Account created successfully!")
                            st.info("Please login with your new account.")
                        else:
                            error_message = result.get("error", {}).get("message", "Unknown error")
                            st.error(f"Sign up failed: {error_message}")
                if st.button("Back to Login"):
                    st.session_state.show_signup = False
            with col2:
                st.markdown("### Welcome to CarbonWise")
                st.write("Join us to track your eco-friendly activities and monitor your CO₂ savings.")
        # Login Flow
        else:
            with col1:
                st.subheader("Login")
                with st.form("login_form", clear_on_submit=False):
                    login_email = st.text_input("Email", key="login_email")
                    login_password = st.text_input("Password", type="password", key="login_password")
                    submit_login = st.form_submit_button("Login")
                    if submit_login:
                        result = firebase_sign_in(login_email, login_password)
                        if result and "error" not in result:
                            st.session_state.logged_in = True
                            st.session_state.user = {
                                "email": login_email,
                                "uid": result["localId"]
                            }
                            st.session_state.auth = result["idToken"]
                            st.success("✅ Logged in successfully!")
                            rerun_app()
                        else:
                            error_message = result.get("error", {}).get("message", "Unknown error")
                            st.error(f"Login failed: {error_message}")
            with col2:
                st.markdown("### Welcome Back!")
                st.write("Log in to view your CarbonWise dashboard and check your CO₂ savings.")
                if st.button("Don't have an account? Sign Up", key="signup_button"):
                    st.session_state.show_signup = True

if __name__ == '__main__':
    main()
