import streamlit as st
import pytesseract
from PIL import Image, ImageOps
import re
import pandas as pd

# --- MODERN 2026 IMPORTS ---
from langsmith import Client
from langchain_google_genai import ChatGoogleGenerativeAI
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
    """Extracts amount/category. Now handles raw numbers like '4000'."""
    # 1. Look for currency patterns (₹500, Paid 500, etc.)
    amount_match = re.search(r'(?:₹|Rs\.?|Paid|Total|Spent)\s?([\d,]+\.?\d*)', text, re.IGNORECASE)
    
    if amount_match:
        amount = float(amount_match.group(1).replace(',', ''))
    else:
        # 2. FALLBACK: Grab the first valid number if no keywords exist
        nums = re.findall(r'[\d,]+\.?\d*', text)
        valid_nums = [float(n.replace(',', '')) for n in nums if len(n.replace(',', '').split('.')[0]) < 7]
        amount = valid_nums[0] if valid_nums else 0.0

    # 3. Flexible Categorization
    text_l = text.lower()
    category = "Miscellaneous"
    if any(k in text_l for k in ["swiggy", "zomato", "food", "dinner", "eat"]): category = "Food"
    elif any(k in text_l for k in ["uber", "ola", "petrol", "ride"]): category = "Transport"
    elif any(k in text_l for k in ["amazon", "flipkart", "shop", "buy"]): category = "Shopping"

    if amount > 0:
        st.session_state.expense_history.append({"Amount": amount, "Category": category})
        return f"Recorded ₹{amount} under {category}."
    
    return "Error: Could not find a numeric value."

def budgeting_tool(query):
    total = sum(item['Amount'] for item in st.session_state.expense_history)
    if total > 10000: return f"⚠️ Spending high: ₹{total}!"
    return f"✅ Budget OK: ₹{total} spent."

def guru_advice_tool(query):
    if not st.session_state.expense_history: return "Save first, spend later!"
    df = pd.DataFrame(st.session_state.expense_history)
    top_cat = df.groupby("Category")["Amount"].sum().idxmax()
    quotes = {"Food": "🍱 Limit the Swiggy orders!", "Shopping": "🛍️ Assets > Things.", "Transport": "🚗 Watch the fuel burn."}
    return quotes.get(top_cat, "Track every rupee.")

# --- UI SETUP ---
st.set_page_config(page_title="AI Finance Agent", page_icon="💰", layout="wide")
st.title("💰 AI Personal Finance Agent")

with st.sidebar:
    gemini_key = st.text_input("Enter Gemini API Key", type="password")
    if st.session_state.expense_history:
        total_val = sum(item['Amount'] for item in st.session_state.expense_history)
        st.metric("Total Spend", f"₹{total_val}")
        st.progress(min(int(total_val/200), 100))

if gemini_key:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=gemini_key, temperature=0)
        tools = [
            Tool(name="OCR", func=ocr_tool, description="Extracts text from images."),
            Tool(name="Analyzer", func=expense_tool, description="Extracts amount/category and SAVES to history."),
            Tool(name="Budgeting", func=budgeting_tool, description="Checks total spending."),
            Tool(name="Guru", func=guru_advice_tool, description="Gives financial advice.")
        ]
        client = Client()
        prompt = client.pull_prompt("hwchase17/react")
        agent_executor = AgentExecutor(agent=create_react_agent(llm, tools, prompt), tools=tools, verbose=True, handle_parsing_errors=True)
    except Exception as e:
        st.error(f"Error: {e}")

# --- DASHBOARD ---
col1, col2 = st.columns([1, 1])

with col1:
    mode = st.radio("Input Mode", ["Manual Entry", "Screenshot"])
    
    if mode == "Manual Entry":
        user_input = st.text_input("Enter amount or description (e.g. 4000)")
        if st.button("Add Entry"):
            # We tell the agent to be smart about raw numbers
            res = agent_executor.invoke({"input": f"Analyze '{user_input}'. If it's just a number, record it and ask me what it's for."})
            st.success(res["output"])
            st.rerun()

    else:
        file = st.file_uploader("Upload Receipt", type=["jpg", "png", "jpeg"])
        if file and st.button("Scan Receipt"):
            text = ocr_tool(file)
            res = agent_executor.invoke({"input": f"Analyze this text: '{text}'. Save it and give Guru advice."})
            st.write(res["output"])
            st.rerun()

    # --- RECENT LOG ---
    if st.session_state.expense_history:
        st.subheader("📝 Recent Transactions")
        st.table(pd.DataFrame(st.session_state.expense_history).tail(5))

with col2:
    st.subheader("📊 Spending Analysis")
    if st.session_state.expense_history:
        df = pd.DataFrame(st.session_state.expense_history)
        chart_data = df.groupby("Category")["Amount"].sum().reset_index()
        st.bar_chart(chart_data.set_index("Category"))
        
        if st.button("Clear All Data"):
            st.session_state.expense_history = []
            st.rerun()
    else:
        st.info("Start adding expenses to see the magic! ✨")

st.divider()
st.caption("2026 AI Finance Agent | Built with Streamlit & Gemini")
