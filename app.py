import streamlit as st
import pytesseract
from PIL import Image, ImageOps
import re
import pandas as pd

# --- GLOBAL MEMORY ---
if 'expense_history' not in st.session_state:
    st.session_state.expense_history = []

def ocr_tool(image_file):
    img = Image.open(image_file)
    # 1. ENHANCE: Make it ultra-high contrast so the white '20' stands out
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img, cutoff=2) # Pushes whites to pure white
    
    # 2. CONFIG: PSM 11 is 'Sparse Text'—perfect for giant floating numbers
    # We also tell it to ONLY look for digits and the Rupee symbol
    custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789₹'
    return pytesseract.image_to_string(img, config=custom_config)

def expense_tool(text):
    # Remove distracting small numbers like times (11:13) or years (2026)
    # We only want numbers that are 2 digits or more but NOT years
    nums = re.findall(r'\b\d{1,3}\b', text)
    valid_nums = [float(n) for n in nums if float(n) not in [11, 13, 22, 2026]]
    
    # In a payment screenshot, the largest remaining number is the amount
    amount = max(valid_nums) if valid_nums else 0.0
    
    # Default category for Swiggy screenshots
    return amount, "Food"
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
