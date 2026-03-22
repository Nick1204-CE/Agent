import streamlit as st
import pytesseract
from PIL import Image, ImageOps
import re
import pandas as pd

# --- GLOBAL MEMORY ---
if 'expense_history' not in st.session_state:
    st.session_state.expense_history = []

def ocr_tool(image_file):
    """Enhanced Pre-processing for Dark-Mode Screenshots (Google Pay)."""
    img = Image.open(image_file)
    img = ImageOps.grayscale(img)
    # Invert helps Tesseract read white text on dark backgrounds
    img = ImageOps.invert(img) 
    img = ImageOps.autocontrast(img)
    return pytesseract.image_to_string(img, config='--oem 3 --psm 6')

def expense_tool(text):
    # 1. THE RUPEE SNIPER: Focus only on digits immediately next to a ₹ symbol
    # This ignores dates, times, and account numbers.
    rupee_pattern = re.search(r'₹\s?(\d+)', text)
    
    if rupee_pattern:
        amount = float(rupee_pattern.group(1))
    else:
        # 2. THE CLEANER: If no ₹, remove times (11:13) and years (2026) manually
        # This removes HH:MM patterns
        clean_text = re.sub(r'\d{1,2}:\d{2}', '', text)
        # This removes 4-digit bank/year numbers
        clean_text = re.sub(r'\b\d{4}\b', '', clean_text)
        
        # 3. Grab the first remaining number
        nums = re.findall(r'\b\d{1,3}\b', clean_text)
        amount = float(nums[0]) if nums else 0.0

    # 4. CATEGORY LOGIC
    category = "Food" if "swiggy" in text.lower() else "Miscellaneous"

    if amount > 0:
        return amount, category
    return 0.0, "Unknown"

# --- UI ---
st.set_page_config(page_title="AI Finance Agent", layout="wide")
st.title("💰 AI Personal Finance Agent")

# Sidebar for manual corrections
with st.sidebar:
    st.header("Ledger Summary")
    if st.session_state.expense_history:
        df = pd.DataFrame(st.session_state.expense_history)
        st.metric("Total Spent", f"₹{df['Amount'].sum()}")
    if st.button("🗑️ Clear All Data"):
        st.session_state.expense_history = []
        st.rerun()

# Main Interface
file = st.file_uploader("Upload Google Pay/UPI Screenshot", type=["png", "jpg", "jpeg"])

if file:
    col1, col2 = st.columns(2)
    with col1:
        st.image(file, caption="Receipt", use_container_width=True)
    
    with col2:
        with st.spinner("Analyzing..."):
            raw_text = ocr_tool(file)
            amt, cat = expense_tool(raw_text)
            
            st.subheader("Verify Details")
            # We let the user edit it, just in case OCR still sees 7895
            correct_amt = st.number_input("Amount (₹)", value=float(amt))
            correct_cat = st.selectbox("Category", ["Food", "Transport", "Shopping", "Bills", "Misc"], 
                                      index=0 if cat=="Food" else 4)
            
            if st.button("✅ Confirm & Save"):
                st.session_state.expense_history.append({"Amount": correct_amt, "Category": correct_cat})
                st.success(f"Saved ₹{correct_amt} to {correct_cat}!")
                st.rerun()

# Visuals
if st.session_state.expense_history:
    st.divider()
    df_viz = pd.DataFrame(st.session_state.expense_history)
    st.bar_chart(df_viz.groupby("Category")["Amount"].sum())
    st.table(df_viz.tail(5))
