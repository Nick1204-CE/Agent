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
    img = Image.open(image_file)
    # Convert to Grayscale
    img = ImageOps.grayscale(img)
    # INVERT colors (Black text on White background is better for OCR)
    img = ImageOps.invert(img)
    # Boost contrast to make the large "20" pop
    img = ImageOps.autocontrast(img)
    
    # Use PSM 6 (Assume a uniform block of text) or PSM 11 (Find sparse text)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(img, config=custom_config)
    return text
def expense_tool(text):
    # 1. CLEAN THE TEXT: Remove common date patterns (like 15 Mar 2026) 
    # so the '15' doesn't distract the code.
    clean_text = re.sub(r'\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}', '', text, flags=re.IGNORECASE)
    
    # 2. PRIORITY 1: Find number attached to Rupee symbol (e.g., ₹20)
    rupee_match = re.search(r'₹\s?([\d,]+\.?\d*)', clean_text)
    
    if rupee_match:
        amount = float(rupee_match.group(1).replace(',', ''))
    else:
        # 3. PRIORITY 2: If no ₹ symbol, find numbers but ignore "Year-like" numbers
        nums = re.findall(r'\b\d{1,4}(?:\.\d{1,2})?\b', clean_text)
        # Filter: Must be a number, but NOT 2024, 2025, or 2026
        valid_nums = [float(n) for n in nums if float(n) not in [2024, 2025, 2026]]
        # Take the largest remaining number (The Payment)
        amount = max(valid_nums) if valid_nums else 0.0

    # 4. CATEGORY: Look for "Swiggy" specifically for your screenshot
    text_l = text.lower()
    category = "Food" if "swiggy" in text_l else "Miscellaneous"

    if amount > 0:
        st.session_state.expense_history.append({"Amount": amount, "Category": category})
        return f"Fixed! Recorded ₹{amount} for {category}."
    return "Could not find amount."

def budgeting_tool(query):
    total = sum(item['Amount'] for item in st.session_state.expense_history)
    return f"Total spending is ₹{total}."

def guru_advice_tool(query):
    return "Save before you spend! High spending in any category should be reviewed monthly."

# --- UI SETUP ---
st.set_page_config(page_title="AI Finance Agent", page_icon="💰", layout="wide")
st.title("💰 AI Personal Finance Agent")

with st.sidebar:
    gemini_key = st.text_input("Enter Gemini API Key", type="password")
    if st.session_state.expense_history:
        total_val = sum(item['Amount'] for item in st.session_state.expense_history)
        st.metric("Total Spend", f"₹{total_val:,.2f}")

agent_executor = None

if gemini_key:
    try:
        # STRICT SAFETY OVERRIDE
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            google_api_key=gemini_key,
            temperature=0,
            safety_settings=safety_settings
        )

        tools = [
            Tool(name="Analyzer", func=expense_tool, description="Extracts amount and category."),
            Tool(name="Budgeting", func=budgeting_tool, description="Checks total spend."),
            Tool(name="Guru", func=guru_advice_tool, description="Provides advice.")
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
        st.error(f"Setup Error: {e}")

# --- DASHBOARD ---
col1, col2 = st.columns([1, 1])

with col1:
    mode = st.radio("Input Method", ["Manual Entry", "Screenshot"])
    
    if mode == "Manual Entry":
        user_input = st.text_input("Enter expense (e.g. 500 for pizza)")
        if st.button("Add Entry") and agent_executor:
            try:
                res = agent_executor.invoke({"input": f"Use Analyzer to save this: {user_input}"})
                st.success(res["output"])
                st.rerun()
            except Exception as e:
                st.error(f"Agent Error: {e}")

    else:
        file = st.file_uploader("Upload Receipt", type=["jpg", "png", "jpeg"])
       # --- IN YOUR BUTTON LOGIC ---
if st.button("Analyze Screenshot") and agent_executor:
    with st.spinner("Checking local cache..."):
        raw_text = ocr_tool(file)
        
        # 1. TRY LOCAL EXTRACTION FIRST (No API Cost)
        local_result = expense_tool(raw_text)
        
        if "Recorded" in local_result:
            st.success(f"✅ Local Logic: {local_result}")
            # Only call Gemini for the 'Guru Advice' part to save quota
            try:
                # Optional: Only call Gemini if you really want advice
                # res = agent_executor.invoke({"input": "Give me one line of guru advice."})
                # st.info(res["output"])
                st.rerun()
            except Exception:
                st.warning("Guru is sleeping (Quota full), but your expense was saved locally!")
        else:
            # 2. ONLY CALL GEMINI IF LOCAL LOGIC FAILS
            try:
                res = agent_executor.invoke({"input": f"Use Analyzer on: {raw_text}"})
                st.success(res["output"])
                st.rerun()
            except Exception as e:
                st.error("API Limit Reached. Please try Manual Entry for now.")

with col2:
    st.subheader("📊 Spending Analysis")
    if st.session_state.expense_history:
        df = pd.DataFrame(st.session_state.expense_history)
        chart_data = df.groupby("Category")["Amount"].sum().reset_index()
        st.bar_chart(chart_data.set_index("Category"))
        if st.button("Clear All"):
            st.session_state.expense_history = []
            st.rerun()
    else:
        st.info("Upload a screenshot to see your spending breakdown.")
