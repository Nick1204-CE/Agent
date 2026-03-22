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
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    # Basic config to help Tesseract focus
    return pytesseract.image_to_string(img, config='--oem 3 --psm 6')

def expense_tool(text):
    # Regex for currency/amount patterns
    amount_match = re.search(r'(?:₹|Rs\.?|Paid|Total|Spent)\s?([\d,]+\.?\d*)', text, re.IGNORECASE)
    
    if amount_match:
        amount = float(amount_match.group(1).replace(',', ''))
    else:
        # Fallback: Find the largest number
        nums = re.findall(r'[\d,]+\.?\d*', text)
        valid_nums = [float(n.replace(',', '')) for n in nums if 0 < float(n.replace(',', '')) < 1000000 and len(n.split('.')[0]) < 7]
        amount = max(valid_nums) if valid_nums else 0.0

    text_l = text.lower()
    category = "Miscellaneous"
    if any(k in text_l for k in ["swiggy", "zomato", "food", "blinkit"]): category = "Food"
    elif any(k in text_l for k in ["uber", "ola", "petrol", "fuel"]): category = "Transport"
    elif any(k in text_l for k in ["amazon", "flipkart", "shop", "myntra"]): category = "Shopping"

    if amount > 0:
        st.session_state.expense_history.append({"Amount": amount, "Category": category})
        return f"Stored ₹{amount} in {category}."
    return "Error: Could not extract amount."

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
        if file and st.button("Analyze Screenshot") and agent_executor:
            try:
                raw_text = ocr_tool(file)
                # Clean text of weird characters before sending to Gemini
                clean_text = "".join(i for i in raw_text if ord(i) < 128)
                res = agent_executor.invoke({"input": f"Use Analyzer on this text: {clean_text}"})
                st.success(res["output"])
                st.rerun()
            except Exception as e:
                st.error(f"Analysis Error: {e}")

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
