import streamlit as st
import pytesseract
from PIL import Image, ImageOps
import re
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langsmith import Client
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool

# --- GLOBAL MEMORY ---
if 'expense_history' not in st.session_state:
    st.session_state.expense_history = []

# --- IMPROVED TOOLS ---

def ocr_tool(image_file):
    """Processes image for better OCR and extracts text."""
    img = Image.open(image_file)
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    custom_config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(img, config=custom_config)

def expense_tool(text):
    """Extracts amount/category. Robust against raw numbers and distractions."""
    # 1. Look for currency patterns first
    amount_match = re.search(r'(?:₹|Rs\.?|Paid|Total|Spent)\s?([\d,]+\.?\d*)', text, re.IGNORECASE)
    
    if amount_match:
        amount = float(amount_match.group(1).replace(',', ''))
    else:
        # 2. FALLBACK: Find the largest valid number (likely the transaction)
        nums = re.findall(r'[\d,]+\.?\d*', text)
        valid_nums = [float(n.replace(',', '')) for n in nums if 0 < float(n.replace(',', '')) < 1000000 and len(n.split('.')[0]) < 7]
        amount = max(valid_nums) if valid_nums else 0.0

    # 3. Categorization
    text_l = text.lower()
    category = "Miscellaneous"
    if any(k in text_l for k in ["swiggy", "zomato", "food", "dinner", "eat", "blinkit"]): category = "Food"
    elif any(k in text_l for k in ["uber", "ola", "petrol", "ride", "fuel"]): category = "Transport"
    elif any(k in text_l for k in ["amazon", "flipkart", "shop", "buy", "myntra"]): category = "Shopping"

    if amount > 0:
        st.session_state.expense_history.append({"Amount": amount, "Category": category})
        return f"Recorded ₹{amount} under {category}."
    return "Error: Could not identify a valid transaction amount."

def budgeting_tool(query):
    total = sum(item['Amount'] for item in st.session_state.expense_history)
    if total > 5000: return f"⚠️ Spending alert: ₹{total} used. Tighten the belt!"
    return f"✅ Budget OK: ₹{total} spent so far."

def guru_advice_tool(query):
    if not st.session_state.expense_history: return "Start tracking to get wisdom!"
    df = pd.DataFrame(st.session_state.expense_history)
    top_cat = df.groupby("Category")["Amount"].sum().idxmax()
    quotes = {
        "Food": "🍱 'Don't save $3 on lattes; focus on the Big Wins.' — Ramit Sethi",
        "Shopping": "🛍️ 'Wealth is assets that earn while you sleep.' — Naval Ravikant",
        "Transport": "🚗 Keep your fixed costs low for maximum freedom."
    }
    return quotes.get(top_cat, "Do not save what is left after spending. — Buffett")

# --- UI SETUP ---
st.set_page_config(page_title="AI Finance Agent", page_icon="💰", layout="wide")
st.title("💰 AI Personal Finance Agent")

with st.sidebar:
    gemini_key = st.text_input("Enter Gemini API Key", type="password")
    if st.session_state.expense_history:
        total_val = sum(item['Amount'] for item in st.session_state.expense_history)
        st.metric("Total Monthly Spend", f"₹{total_val:,.2f}")
        health_score = max(0, 100 - int(total_val / 100))
        st.write(f"**Financial Health: {health_score}/100**")
        st.progress(health_score)

agent_executor = None

if gemini_key:
    try:
        # RELAXED SAFETY SETTINGS to prevent ClientError
        safety_cfg = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", # Using stable 1.5-flash
            google_api_key=gemini_key,
            temperature=0,
            safety_settings=safety_cfg
        )

        tools = [
            Tool(name="OCR", func=ocr_tool, description="Extracts raw text from images."),
            Tool(name="Analyzer", func=expense_tool, description="Extracts amount/category and saves it."),
            Tool(name="Budgeting", func=budgeting_tool, description="Checks total spend history."),
            Tool(name="Guru", func=guru_advice_tool, description="Provides dynamic wealth advice.")
        ]

        client = Client()
        prompt = client.pull_prompt("hwchase17/react")
        agent_executor = AgentExecutor(
            agent=create_react_agent(llm, tools, prompt),
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
    except Exception as e:
        st.error(f"Initialization Error: {e}")

# --- DASHBOARD ---
col1, col2 = st.columns([1, 1])

with col1:
    mode = st.radio("Input Method", ["Manual Entry", "Screenshot"])
    
    if mode == "Manual Entry":
        user_input = st.text_input("Enter amount or detail (e.g. 20 for chai)")
        if st.button("Add Entry") and agent_executor:
            with st.spinner("Processing..."):
                res = agent_executor.invoke({"input": f"Analyze '{user_input}'. Extract amount, save it, and check my budget."})
                st.success(res["output"])
                st.rerun()

    else:
        file = st.file_uploader("Upload Receipt", type=["jpg", "png", "jpeg"])
        if file and st.button("Analyze Screenshot") and agent_executor:
            with st.spinner("Agent is reading image..."):
                text = ocr_tool(file)
                res = agent_executor.invoke({"input": f"Analyze this text: '{text}'. Save it and give Guru advice."})
                st.success(res["output"])
                st.rerun()

    if st.session_state.expense_history:
        st.subheader("📝 Recent Transactions")
        df_log = pd.DataFrame(st.session_state.expense_history).tail(5)
        st.table(df_log)

with col2:
    st.subheader("📊 Spending Trends")
    if st.session_state.expense_history:
        df = pd.DataFrame(st.session_state.expense_history)
        chart_data = df.groupby("Category")["Amount"].sum().reset_index()
        
        # Color chart red if total spending is high
        total_now = chart_data["Amount"].sum()
        color = "#FF4B4B" if total_now > 5000 else "#1F77B4"
        
        st.bar_chart(chart_data.set_index("Category"), color=color)
        
        if st.button("Clear History"):
            st.session_state.expense_history = []
            st.rerun()
    else:
        st.info("No data yet. Let's start tracking!")

st.divider()
st.caption("AI Finance Agent v2.0 | Built with LangChain & Gemini 1.5")
