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
    width, height = img.size
    
    # CROP the image to only the top 40% 
    # This removes the bank details, dates, and times at the bottom!
    top_half = img.crop((0, 0, width, int(height * 0.4)))
    
    # Pre-process the cropped area
    top_half = ImageOps.grayscale(top_half)
    top_half = ImageOps.invert(top_half)
    top_half = ImageOps.autocontrast(top_half)
    
    return pytesseract.image_to_string(top_half, config='--oem 3 --psm 6')

def expense_tool(text):
    # Now that we've cropped the '15' out of the image, 
    # we just need to find the number next to the ₹
    match = re.search(r'(?:₹|Rs\.?)\s?(\d+)', text)
    
    if match:
        amount = float(match.group(1))
    else:
        # Fallback for raw numbers in the top section
        nums = re.findall(r'\b\d{1,4}\b', text)
        amount = float(nums[0]) if nums else 0.0
        
    category = "Food" if "swiggy" in text.lower() else "Miscellaneous"
    return amount, category
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
