import streamlit as st
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# --- PERSISTENCE (Track A Requirement) ---
DB_FILE = "expense_database.csv"

def save_data(df):
    df.to_csv(DB_FILE, index=False)

if 'ledger' not in st.session_state:
    if os.path.exists(DB_FILE):
        st.session_state.ledger = pd.read_csv(DB_FILE)
    else:
        st.session_state.ledger = pd.DataFrame(columns=["Date", "Amount", "Category", "Source"])

# --- WEEK 3-4: GURU ADVICE ENGINE ---
def get_financial_advice(user_data, guru_principles=""):
    """
    Analyzes spending patterns based on financial philosophies (Track A Goal).
    [cite: 35, 99]
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=st.session_state.api_key)
    
    summary = user_data.groupby('Category')['Amount'].sum().to_dict()
    
    prompt = f"""
    As a financial advisor, analyze this spending: {summary}.
    Apply these principles: {guru_principles if guru_principles else '50/30/20 Rule'}.
    Give 2-3 specific, actionable tips for an Indian college student.
    """
    response = llm.invoke(prompt)
    return response.content

# --- UI LAYOUT ---
st.set_page_config(page_title="Personal Finance Agent", layout="wide")
st.title("🚀 Financial Advisor & Expense Manager")

# Sidebar for API Key and Guru Knowledge
with st.sidebar:
    st.session_state.api_key = st.text_input("Enter API Key", type="password")
    st.divider()
    st.subheader("📚 Guru Knowledge Base")
    guru_text = st.text_area("Paste Financial Principles (e.g., from Rich Dad Poor Dad)", 
                             help="Week 3-4: Multi-source content integration ")

# --- WEEK 3-4: MULTI-SOURCE TABS ---
tab1, tab2, tab3 = st.tabs(["📸 Screenshot", "⌨️ Manual Entry", "📊 Dashboard & Advice"])

with tab1:
    uploaded_file = st.file_uploader("Upload UPI Screenshot", type=["png", "jpg"])
    # (Insert your existing Vision processing logic here)

with tab2:
    st.subheader("Manual Expense Entry ")
    with st.form("manual_form"):
        col1, col2 = st.columns(2)
        m_amt = col1.number_input("Amount (₹)", min_value=0.0)
        m_cat = col2.selectbox("Category", ["Food", "Transport", "Fees", "Shopping", "Misc"])
        m_date = st.date_input("Transaction Date")
        if st.form_submit_button("Add Expense"):
            new_entry = pd.DataFrame([{"Date": m_date, "Amount": m_amt, "Category": m_cat, "Source": "Manual"}])
            st.session_state.ledger = pd.concat([st.session_state.ledger, new_entry], ignore_index=True)
            save_data(st.session_state.ledger)
            st.success("Manual entry saved!")

with tab3:
    if not st.session_state.ledger.empty:
        # --- WEEK 4: VISUALIZATION ---
        st.subheader("Spending Pattern Analysis ")
        chart_data = st.session_state.ledger.groupby("Category")["Amount"].sum()
        st.bar_chart(chart_data)
        
        # --- WEEK 4: ADVICE MILESTONE ---
        if st.button("Generate AI Financial Advice [cite: 103]"):
            with st.spinner("Analyzing your habits..."):
                advice = get_financial_advice(st.session_state.ledger, guru_text)
                st.markdown(f"### 💡 Personalized Advice\n{advice}")
    else:
        st.info("Add some expenses to see your financial health score.")
