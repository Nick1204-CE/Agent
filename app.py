import streamlit as st
import pandas as pd
import base64
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# --- 1. PERSISTENCE LAYER (Real-World Applied) ---
DB_FILE = "my_expenses.csv"

def load_data():
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE)
    return pd.DataFrame(columns=["Date", "Amount", "Category"])

if 'ledger' not in st.session_state:
    st.session_state.ledger = load_data()

# --- 2. THE VISION ENGINE (Fixes the ValueError) ---
def process_with_vision(image_file, api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    
    encoded = base64.b64encode(image_file.getvalue()).decode()
    
    # This specific structure is required to avoid ValueErrors
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Identify the total payment amount and category. Return ONLY JSON like {'amount': 20.0, 'category': 'Food'}"},
            {
                "type": "image_url",
                "url": f"data:image/jpeg;base64,{encoded}", # Note: url key inside image_url
            },
        ]
    )
    
    response = llm.invoke([message])
    return response.content

# --- 3. THE INTERFACE ---
st.set_page_config(page_title="Pro Finance Agent", page_icon="💳")
st.title("💳 Real-World Finance Agent")

with st.sidebar:
    key = st.text_input("Gemini API Key", type="password")
    if not st.session_state.ledger.empty:
        total = st.session_state.ledger["Amount"].sum()
        st.metric("Total Monthly Spend", f"₹{total}")

uploaded_file = st.file_uploader("Upload Receipt", type=["png", "jpg", "jpeg"])

if uploaded_file and key:
    st.image(uploaded_file, width=250)
    
    if st.button("Analyze Receipt"):
        try:
            # Gemini Vision sees the '20' and ignores the '15' or '7895'
            raw_res = process_with_vision(uploaded_file, key)
            st.info(f"AI Detected: {raw_res}")
            
            # Form for user to verify (Essential for real-world accuracy)
            with st.form("confirmation"):
                # You can manually set these if the JSON parsing fails
                final_amt = st.number_input("Confirm Amount", step=1.0)
                final_cat = st.selectbox("Category", ["Food", "Transport", "Shopping", "Misc"])
                
                if st.form_submit_button("Save to Ledger"):
                    new_row = pd.DataFrame([{
                        "Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                        "Amount": final_amt,
                        "Category": final_cat
                    }])
                    st.session_state.ledger = pd.concat([st.session_state.ledger, new_row], ignore_index=True)
                    st.session_state.ledger.to_csv(DB_FILE, index=False)
                    st.success("Transaction Synced!")
                    st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# --- 4. DATA VISUALS ---
if not st.session_state.ledger.empty:
    st.divider()
    st.subheader("Your Spending History")
    st.dataframe(st.session_state.ledger, use_container_width=True)
