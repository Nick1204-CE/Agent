import streamlit as st
import pytesseract
from PIL import Image, ImageOps
import re
import pandas as pd
import io

# --- MODERN 2026 IMPORTS ---
from langsmith import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool

# --- GLOBAL MEMORY INITIALIZATION ---
if 'expense_history' not in st.session_state:
    st.session_state.expense_history = []

# --- IMPROVED TOOLS ---

def ocr_tool(image_file):
    """Processes image for better OCR and extracts text."""
    img = Image.open(image_file)
    # Pre-processing: Grayscale + Contrast boost
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    
    # Whitelist numbers and common currency characters for better accuracy
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(img, config=custom_config)
    return text

def expense_tool(text):
    """Extracts amount and category, then saves to session memory."""
    # Improved Regex for UPI/Banking (matches ₹500, Rs. 500, Paid 500)
    amount_match = re.search(r'(?:₹|Rs\.?|Paid|Total)\s?([\d,]+\.?\d*)', text)
    
    if amount_match:
        amount = float(amount_match.group(1).replace(',', ''))
    else:
        # Fallback: Find the largest number (usually the amount)
        nums = [float(n.replace(',', '')) for n in re.findall(r'[\d,]+\.\d+', text)]
        amount = max(nums) if nums else 0.0

    text_lower = text.lower()
    if any(k in text_lower for k in ["swiggy", "zomato", "food", "blinkit", "restaurant"]):
        category = "Food"
    elif any(k in text_lower for k in ["uber", "ola", "rapido", "petrol", "fuel"]):
        category = "Transport"
    elif any(k in text_lower for k in ["amazon", "flipkart", "myntra", "mall", "shopping"]):
        category = "Shopping"
    else:
        category = "Miscellaneous"

    # Save to Session State if a valid amount was found
    if amount > 0:
        st.session_state.expense_history.append({"Amount": amount, "Category": category})
        return f"Success! Recorded ₹{amount} for {category}."
    
    return "Could not find a clear amount. Please try manual entry."

def budgeting_tool(query):
    """Calculates burn rate based on total history."""
    history = st.session_state.expense_history
    if not history:
        return "No spending data found yet."
    
    total = sum(item['Amount'] for item in history)
    if total > 10000:
        return f"⚠️ Total spend: ₹{total}. You are overspending!"
    elif total > 5000:
        return f"⚠️ Total spend: ₹{total}. You are near your budget limit."
    return f"✅ Total spend: ₹{total}. Your spending is under control."

def guru_advice_tool(query):
    """Provides dynamic advice based on the highest spending category."""
    history = st.session_state.expense_history
    if not history:
        return "Warren Buffett: 'Save before you spend.' Start by scanning a receipt!"

    df = pd.DataFrame(history)
    top_cat = df.groupby("Category")["Amount"].sum().idxmax()
    
    gurus = {
        "Food": "🍱 High food spend? Ramit Sethi says: 'Focus on Big Wins, but stop ghost-spending on apps.'",
        "Shopping": "🛍️ Naval Ravikant: 'Wealth is assets that earn while you sleep, not things that clutter your room.'",
        "Transport": "🚗 Keep your burn rate low to maintain your freedom.",
        "Miscellaneous": "💰 'Do not save what is left after spending...' — Warren Buffett"
    }
    return gurus.get(top_cat, "Invest wisely.")

# --- UI SETUP ---
st.set_page_config(page_title="AI Finance Agent", page_icon="💰", layout="wide")
st.title("💰 AI Personal Finance Agent")

with st.sidebar:
    st.header("Settings")
    gemini_key = st.text_input("Enter Gemini API Key", type="password")
    st.divider()
    
    # Financial Health Meter
    if st.session_state.expense_history:
        total_all = sum(item['Amount'] for item in st.session_state.expense_history)
        st.metric("Total Monthly Spend", f"₹{total_all}")
        score = max(0, 100 - int(total_all / 200))
        st.write(f"**Financial Health: {score}/100**")
        st.progress(score)

agent_executor = None

if gemini_key:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            google_api_key=gemini_key,
            temperature=0
        )

        tools = [
            Tool(name="OCR", func=ocr_tool, description="Extracts raw text from receipt images."),
            Tool(name="Analyzer", func=expense_tool, description="Extracts amount/category and SAVES to history."),
            Tool(name="Budgeting", func=budgeting_tool, description="Checks total spending history."),
            Tool(name="Guru", func=guru_advice_tool, description="Gives advice based on spending leaks.")
        ]

        client = Client()
        prompt = client.pull_prompt("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    except Exception as e:
        st.error(f"Setup Error: {e}")

# --- MAIN DASHBOARD ---
col1, col2 = st.columns([1, 1])

with col1:
    option = st.selectbox("Input Method", ["Screenshot", "Manual Entry"])

    if option == "Screenshot":
        uploaded_file = st.file_uploader("Upload Receipt", type=["jpg", "png", "jpeg"])
        if uploaded_file and agent_executor:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            if st.button("Process & Save"):
                with st.spinner("Agent is analyzing..."):
                    text = ocr_tool(uploaded_file)
                    # We pass clear instructions to the agent
                    res = agent_executor.invoke({"input": f"Analyze this text: '{text}'. Extract amount, save it, and then give me Guru advice."})
                    st.success(res["output"])

    else:
        user_input = st.text_input("Describe expense (e.g., 'Paid 500 for dinner')")
        if st.button("Add Entry") and agent_executor:
            res = agent_executor.invoke({"input": f"Analyze: '{user_input}'. Save it and check my budget."})
            st.success(res["output"])

with col2:
    st.subheader("📊 Spending Overview")
    if st.session_state.expense_history:
        df = pd.DataFrame(st.session_state.expense_history)
        chart_data = df.groupby("Category")["Amount"].sum().reset_index()
        st.bar_chart(chart_data.set_index("Category"))
        
        # Download Data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV Report", data=csv, file_name="expenses.csv", mime="text/csv")
        
        if st.button("Clear History"):
            st.session_state.expense_history = []
            st.rerun()
    else:
        st.info("No data yet. Upload a screenshot to see your breakdown.")

st.divider()
st.caption("Powered by Gemini 2.5 Flash-Lite & LangChain-Classic")
