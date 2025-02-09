import streamlit as st
import requests
import json
from PIL import Image
import base64
import os
from datetime import datetime
from huggingface_hub import InferenceClient
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import re  # Required for regex operations

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
    if params and 'auth' in st.session_state:
        params['auth'] = st.session_state.auth

    try:
        if method == 'GET':
            response = requests.get(url, params=params)
        elif method == 'POST':
            response = requests.post(url, json=data, params=params)
        elif method == 'PUT':
            response = requests.put(url, json=data, params=params)
        elif method == 'PATCH':
            response = requests.patch(url, json=data, params=params)

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Firebase API Error: {str(e)}")
        return None

def sign_in_with_email_password(email, password):
    """Sign in user with email and password"""
    auth_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }

    try:
        response = requests.post(auth_url, json=payload)
        data = response.json()

        if 'error' in data:
            return None, data['error']['message']

        return data, None
    except Exception as e:
        return None, str(e)

def sign_up_with_email_password(email, password):
    """Sign up new user with email and password"""
    auth_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_WEB_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }

    try:
        response = requests.post(auth_url, json=payload)
        data = response.json()

        if 'error' in data:
            return None, data['error']['message']

        return data, None
    except Exception as e:
        return None, str(e)

# ------------------------------------------------------------------
# Helper Functions
def rerun_app():
    """Helper function to safely rerun the Streamlit app"""
    try:
        st.experimental_rerun()
    except Exception:
        st.stop()

def get_co2_summary(user_id):
    """Fetch CO₂ savings summary using REST API"""
    summary = {"Today": 0, "This Week": 0, "This Month": 0, "This Year": 0, "Overall": 0}

    activities = firebase_request(f'users/{user_id}/activities',
                                  params={'auth': st.session_state.auth})

    if activities:
        now = datetime.now()
        for activity in activities.values():
            timestamp_str = activity.get("timestamp")
            if timestamp_str:
                try:
                    dt = datetime.fromisoformat(timestamp_str)
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
                except Exception:
                    continue

    return summary

def generate_trivia(overall_co2):
    """Generate trivia based on CO₂ savings"""
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
                    {
                        "type": "text",
                        "text": (
                            "Analyze this image in detail for eco-friendly activities. "
                            "Look for any evidence of recycling (e.g., reusable items or PCR content), "
                            "public transport usage (e.g., train tickets), or exercise (walking/cycling). "
                            "Provide measurements and numbers where visible."
                        )
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ]
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct",
            messages=messages,
            max_tokens=500
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# ------------------------------------------------------------------
# CO2 Calculation Functions
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

        lines = response.split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()

                if key in activity_details:
                    if key in ['pcr_percentage', 'reusable_items', 'single_use_saved', 'distance_km', 'time_duration']:
                        try:
                            activity_details[key] = float(re.findall(r'[\d.]+', value)[0])
                        except (IndexError, ValueError):
                            activity_details[key] = 0
                    else:
                        activity_details[key] = value

        activity_details["co2_savings"] = calculate_co2_savings(
            activity_details["activity_type"],
            activity_details
        )

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
# Main Function with Authentication
def main():
    st.set_page_config(page_title="CarbonWise", page_icon="🌱", layout="centered")

    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False
    if 'auth' not in st.session_state:
        st.session_state.auth = None
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None

    # Authentication UI
    if not st.session_state.logged_in:
        st.title("CarbonWise")

        if st.session_state.show_signup:
            # Signup Form
            with st.form("signup_form"):
                st.subheader("Sign Up")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Sign Up")

                if submit:
                    user_data, error = sign_up_with_email_password(email, password)
                    if error:
                        st.error(f"Signup failed: {error}")
                    else:
                        st.success("Account created! Please log in.")
                        st.session_state.show_signup = False
                        rerun_app()

            if st.button("Already have an account? Log In"):
                st.session_state.show_signup = False
                rerun_app()

        else:
            # Login Form
            with st.form("login_form"):
                st.subheader("Login")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")

                if submit:
                    user_data, error = sign_in_with_email_password(email, password)
                    if error:
                        st.error(f"Login failed: {error}")
                    else:
                        st.session_state.auth = user_data['idToken']
                        st.session_state.user_id = user_data['localId']
                        st.session_state.logged_in = True
                        st.success("Login successful!")
                        rerun_app()

            if st.button("Don't have an account? Sign Up"):
                st.session_state.show_signup = True
                rerun_app()

    else:
        # Main Dashboard
        st.title("📊 CarbonWise Dashboard")

        # Logout button
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.auth = None
            st.session_state.user_id = None
            rerun_app()

        # File upload and processing
        uploaded_file = st.file_uploader("Upload an image of your eco-friendly activity",
                                         type=['png', 'jpg', 'jpeg'])

        if uploaded_file:
            # Save uploaded file temporarily
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Display image
            image = Image.open("temp_image.jpg")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            with st.spinner("Analyzing image..."):
                # Process image
                vision_output = describe_image("temp_image.jpg")
                activity_details = process_with_langchain(vision_output)

                # Display results
                st.markdown("### Analysis Results")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Activity Type:**", activity_details["activity_type"])
                    st.write("**Description:**", activity_details["items_description"])
                    if activity_details["distance_km"] > 0:
                        st.write("**Distance:**", f"{activity_details['distance_km']:.1f} km")
                    if activity_details["time_duration"] > 0:
                        st.write("**Duration:**", f"{activity_details['time_duration']:.0f} minutes")

                with col2:
                    st.write("**CO₂ Savings:**", f"{activity_details['co2_savings']:.2f} kg")
                    if activity_details["pcr_percentage"] > 0:
                        st.write("**PCR Content:**", f"{activity_details['pcr_percentage']:.1f}%")
                    if activity_details["reusable_items"] > 0:
                        st.write("**Reusable Items:**", f"{activity_details['reusable_items']:.0f}")

                # Save to Firebase
                activity_data = {
                    "timestamp": datetime.now().isoformat(),
                    "activity_details": activity_details
                }

                result = firebase_request(
                    f'users/{st.session_state.user_id}/activities',
                    method='POST',
                    data=activity_data,
                    params={'auth': st.session_state.auth}
                )

                if result:
                    st.success("Activity saved successfully!")
                else:
                    st.error("Failed to save activity")

            # Clean up
            os.remove("temp_image.jpg")

        # Display CO2 summary
        summary = get_co2_summary(st.session_state.user_id)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Today", f"{summary['Today']:.1f} kg")
        with col2:
            st.metric("This Week", f"{summary['This Week']:.1f} kg")
        with col3:
            st.metric("This Month", f"{summary['This Month']:.1f} kg")
        with col4:
            st.metric("This Year", f"{summary['This Year']:.1f} kg")

if __name__ == "__main__":
    main()
