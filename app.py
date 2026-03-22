import streamlit as st
import pandas as pd
import os
import base64
import time
from datetime import date
from io import BytesIO
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import google.generativeai as genai

# ─────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────
DB_FILE = "expense_database.csv"
BUDGET_FILE = "budgets.csv"

CATEGORIES = ["Food", "Transport", "Fees", "Shopping", "Entertainment", "Health", "Misc"]


def save_expenses(df):
    df.to_csv(DB_FILE, index=False)


def load_expenses():
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE)
    return pd.DataFrame(columns=["Date", "Amount", "Category", "Source"])


def save_budgets(budgets: dict):
    pd.DataFrame(list(budgets.items()), columns=["Category", "Limit"]).to_csv(BUDGET_FILE, index=False)


def load_budgets() -> dict:
    if os.path.exists(BUDGET_FILE):
        df = pd.read_csv(BUDGET_FILE)
        return dict(zip(df["Category"], df["Limit"]))
    return {}


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "ledger" not in st.session_state:
    st.session_state.ledger = load_expenses()

if "budgets" not in st.session_state:
    st.session_state.budgets = load_budgets()

if "api_key" not in st.session_state:
    st.session_state.api_key = ""


# ─────────────────────────────────────────────
# HELPER: get LLM
# ─────────────────────────────────────────────
def get_llm():
    if not st.session_state.api_key:
        st.error("⚠️ Please enter your Gemini API key in the sidebar.")
        st.stop()
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=st.session_state.api_key
    )


# ─────────────────────────────────────────────
# FEATURE 1 — SCREENSHOT VISION EXTRACTION
# ─────────────────────────────────────────────
def resize_image(image_bytes: bytes, max_size: int = 1024) -> tuple[bytes, str]:
    """Resize image so longest side <= max_size to stay within token limits."""
    img = Image.open(BytesIO(image_bytes))
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO()
    fmt = img.format or "JPEG"
    if fmt not in ("JPEG", "PNG"):
        fmt = "JPEG"
    img.save(buf, format=fmt)
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return buf.getvalue(), mime


def validate_api_key(api_key: str) -> bool:
    """Quick text-only ping to confirm the key works before sending image."""
    try:
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel("gemini-2.0-flash-lite")
        m.generate_content("Say OK")
        return True
    except Exception:
        return False


def extract_expenses_from_screenshot(image_bytes: bytes, mime_type: str) -> list[dict]:
    """
    Sends UPI / bank screenshot to Gemini Vision using the native SDK.
    Retries up to 3 times with backoff on quota errors.
    Returns a list of dicts: {Date, Amount, Category, Source}
    """
    if not st.session_state.api_key:
        st.error("⚠️ Please enter your Gemini API key in the sidebar.")
        st.stop()

    # Resize image to avoid token limit issues
    image_bytes, mime_type = resize_image(image_bytes)

    genai.configure(api_key=st.session_state.api_key)
    model = genai.GenerativeModel("gemini-2.0-flash-lite")

    prompt = """
    You are an expense extraction assistant. Look at this payment/UPI/bank screenshot.
    Extract ALL transactions you can see.
    Return ONLY a Python list of dicts, no explanation, no markdown fences.
    Each dict must have exactly these keys:
      "Date"     : "YYYY-MM-DD" (use today if not visible)
      "Amount"   : float (INR, positive number)
      "Category" : one of [Food, Transport, Fees, Shopping, Entertainment, Health, Misc]
      "Source"   : "Screenshot"

    Example output (no other text):
    [{"Date": "2024-06-01", "Amount": 120.0, "Category": "Food", "Source": "Screenshot"}]
    """

    image_part = {"mime_type": mime_type, "data": image_bytes}

    # Retry up to 3 times with exponential backoff on quota errors
    for attempt in range(3):
        try:
            response = model.generate_content([image_part, prompt])
            raw = response.text.strip()
            break  # success — exit retry loop
        except Exception as e:
            err = str(e)
            if "ResourceExhausted" in err or "429" in err:
                if attempt < 2:
                    wait = 15 * (attempt + 1)  # 15s, 30s
                    st.warning(f"⏳ Rate limit hit — retrying in {wait} seconds... (attempt {attempt + 1}/3)")
                    time.sleep(wait)
                    continue
                else:
                    st.error(
                        "⚠️ Quota exhausted after 3 retries.\n\n"
                        "**Likely cause:** Too many requests today on this API key.\n\n"
                        "**Fix options:**\n"
                        "1. Wait 24 hours for quota reset\n"
                        "2. Create a **new Google account** → new API key at aistudio.google.com\n"
                        "3. Enable billing on your Google Cloud project for higher limits"
                    )
                    return []
            elif "API_KEY_INVALID" in err or "401" in err:
                st.error("⚠️ Invalid API key. Check the key in the sidebar.")
                return []
            elif "NotFound" in err or "404" in err:
                st.error("⚠️ Model not found. Check your Google Cloud region or API key permissions.")
                return []
            else:
                st.error(f"⚠️ Unexpected error: {err}")
                return []

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("python"):
            raw = raw[6:]
    raw = raw.strip()

    try:
        extracted = eval(raw)
        if isinstance(extracted, list):
            return extracted
    except Exception:
        pass
    return []


# ─────────────────────────────────────────────
# FEATURE 2 — SPLITWISE CSV INTEGRATION
# ─────────────────────────────────────────────
def parse_splitwise_csv(uploaded_csv) -> pd.DataFrame:
    """
    Parses a Splitwise export CSV into our standard ledger format.
    Splitwise columns: Date, Description, Category, Cost, Currency, ...
    Handles both the old and new Splitwise export formats.
    """
    df = pd.read_csv(uploaded_csv)
    df.columns = [c.strip() for c in df.columns]

    # Normalise column names (Splitwise sometimes uses different casing)
    col_map = {c.lower(): c for c in df.columns}

    date_col   = col_map.get("date")
    amount_col = col_map.get("cost") or col_map.get("amount")
    cat_col    = col_map.get("category")

    if not date_col or not amount_col:
        return pd.DataFrame()

    result = pd.DataFrame()
    result["Date"]   = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    result["Amount"] = pd.to_numeric(df[amount_col], errors="coerce").abs()

    # Map Splitwise categories → our categories
    splitwise_cat_map = {
        "food and drink": "Food",
        "groceries": "Food",
        "restaurant": "Food",
        "transportation": "Transport",
        "taxi": "Transport",
        "entertainment": "Entertainment",
        "movies": "Entertainment",
        "utilities": "Fees",
        "rent": "Fees",
        "medical": "Health",
        "healthcare": "Health",
        "shopping": "Shopping",
        "general": "Misc",
    }

    if cat_col:
        result["Category"] = df[cat_col].str.lower().map(
            lambda x: next((v for k, v in splitwise_cat_map.items() if k in str(x)), "Misc")
        )
    else:
        result["Category"] = "Misc"

    result["Source"] = "Splitwise"
    return result.dropna(subset=["Amount"])


# ─────────────────────────────────────────────
# FEATURE 3 — GURU ADVICE ENGINE
# ─────────────────────────────────────────────
def get_financial_advice(user_data: pd.DataFrame, guru_principles: str = "") -> str:
    llm = get_llm()
    summary = user_data.groupby("Category")["Amount"].sum().to_dict()
    total   = sum(summary.values())

    budget_status = ""
    if st.session_state.budgets:
        lines = []
        for cat, limit in st.session_state.budgets.items():
            spent = summary.get(cat, 0)
            pct   = (spent / limit * 100) if limit > 0 else 0
            lines.append(f"  {cat}: spent ₹{spent:.0f} / budget ₹{limit:.0f} ({pct:.0f}%)")
        budget_status = "Budget status:\n" + "\n".join(lines)

    prompt = f"""
You are a financial advisor for an Indian college student.
Total spending: ₹{total:.0f}
Spending by category: {summary}
{budget_status}

Apply these financial principles: {guru_principles if guru_principles else '50/30/20 Rule'}.

Give 3 specific, actionable, encouraging tips. 
Mention categories where the student is over-budget if any.
Keep it concise and friendly.
"""
    response = llm.invoke(prompt)
    return response.content


# ─────────────────────────────────────────────
# FEATURE 4 — BUDGET WARNING HELPER
# ─────────────────────────────────────────────
def show_budget_warnings(ledger: pd.DataFrame, budgets: dict):
    if not budgets or ledger.empty:
        return
    spending = ledger.groupby("Category")["Amount"].sum()
    warnings = []
    for cat, limit in budgets.items():
        spent = spending.get(cat, 0)
        if limit > 0:
            pct = spent / limit * 100
            if pct >= 100:
                warnings.append(("🔴", cat, spent, limit, pct))
            elif pct >= 80:
                warnings.append(("🟡", cat, spent, limit, pct))
    if warnings:
        st.subheader("⚠️ Budget Alerts")
        for icon, cat, spent, limit, pct in warnings:
            label = "OVER BUDGET" if pct >= 100 else f"{pct:.0f}% used"
            st.warning(f"{icon} **{cat}**: ₹{spent:.0f} / ₹{limit:.0f} — {label}")


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Personal Finance Agent", layout="wide", page_icon="💰")
st.title("💰 Financial Advisor & Expense Manager")
st.caption("Week 3–4 build — multi-source expenses · guru advice · budget tracking")

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.api_key = st.text_input("Gemini API Key", type="password",
                                              value=st.session_state.api_key)
    st.divider()

    st.subheader("📚 Guru Knowledge Base")
    guru_text = st.text_area(
        "Paste financial principles (e.g. Rich Dad Poor Dad excerpts)",
        height=160,
        help="Week 3-4: multi-source content integration"
    )

    st.divider()

    # ── FEATURE 4: Budget Settings ────────────
    st.subheader("🎯 Monthly Budgets (₹)")
    updated_budgets = {}
    for cat in CATEGORIES:
        default = st.session_state.budgets.get(cat, 0)
        val = st.number_input(cat, min_value=0, value=int(default), step=100, key=f"budget_{cat}")
        if val > 0:
            updated_budgets[cat] = val

    if st.button("💾 Save Budgets"):
        st.session_state.budgets = updated_budgets
        save_budgets(updated_budgets)
        st.success("Budgets saved!")

    st.divider()
    if st.button("🗑️ Clear All Expenses"):
        st.session_state.ledger = pd.DataFrame(columns=["Date", "Amount", "Category", "Source"])
        save_expenses(st.session_state.ledger)
        st.success("Cleared.")

# ── Budget warnings banner ────────────────────
show_budget_warnings(st.session_state.ledger, st.session_state.budgets)

# ── Tabs ──────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Screenshot",
    "⌨️ Manual Entry",
    "📂 Splitwise Import",
    "📊 Dashboard & Advice"
])

# ─────────────────────────────────────────────
# TAB 1 — SCREENSHOT VISION (Feature 1)
# ─────────────────────────────────────────────
with tab1:
    st.subheader("Extract Expenses from UPI / Bank Screenshot")
    uploaded_img = st.file_uploader("Upload screenshot", type=["png", "jpg", "jpeg"])

    if uploaded_img:
        st.image(uploaded_img, caption="Uploaded screenshot", use_column_width=True)
        if st.button("🔍 Extract Expenses with AI"):
            with st.spinner("Analysing screenshot with Gemini Vision..."):
                mime = "image/png" if uploaded_img.name.endswith(".png") else "image/jpeg"
                extracted = extract_expenses_from_screenshot(uploaded_img.read(), mime)

            if extracted:
                df_extracted = pd.DataFrame(extracted)
                st.success(f"✅ Found {len(df_extracted)} transaction(s)!")
                st.dataframe(df_extracted, use_container_width=True)

                if st.button("➕ Add These to Ledger"):
                    st.session_state.ledger = pd.concat(
                        [st.session_state.ledger, df_extracted], ignore_index=True
                    )
                    save_expenses(st.session_state.ledger)
                    st.success("Added to ledger!")
            else:
                st.warning("Could not extract any transactions. Try a clearer screenshot.")

# ─────────────────────────────────────────────
# TAB 2 — MANUAL ENTRY
# ─────────────────────────────────────────────
with tab2:
    st.subheader("Manual Expense Entry")
    with st.form("manual_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        m_amt  = col1.number_input("Amount (₹)", min_value=0.0, step=10.0)
        m_cat  = col2.selectbox("Category", CATEGORIES)
        m_date = st.date_input("Transaction Date", value=date.today())
        m_note = st.text_input("Note (optional)")
        submitted = st.form_submit_button("➕ Add Expense")

    if submitted:
        new_row = pd.DataFrame([{
            "Date": str(m_date),
            "Amount": m_amt,
            "Category": m_cat,
            "Source": "Manual",
            "Note": m_note
        }])
        st.session_state.ledger = pd.concat(
            [st.session_state.ledger, new_row], ignore_index=True
        )
        save_expenses(st.session_state.ledger)
        st.success(f"✅ ₹{m_amt:.0f} added under {m_cat}.")

# ─────────────────────────────────────────────
# TAB 3 — SPLITWISE CSV IMPORT (Feature 2)
# ─────────────────────────────────────────────
with tab3:
    st.subheader("Import Splitwise Group Expenses")
    st.info("Export your Splitwise data: **Account → Export to CSV** and upload here.")

    sw_file = st.file_uploader("Upload Splitwise CSV", type=["csv"])
    if sw_file:
        parsed = parse_splitwise_csv(sw_file)
        if parsed.empty:
            st.error("Could not parse this CSV. Make sure it's a standard Splitwise export.")
        else:
            st.success(f"✅ Parsed {len(parsed)} transactions from Splitwise.")
            st.dataframe(parsed, use_container_width=True)

            if st.button("➕ Add Splitwise Data to Ledger"):
                st.session_state.ledger = pd.concat(
                    [st.session_state.ledger, parsed], ignore_index=True
                )
                save_expenses(st.session_state.ledger)
                st.success("Splitwise expenses added to ledger!")

    st.divider()
    st.subheader("📥 Or Upload Any Expense CSV")
    st.caption("Columns required: Date, Amount, Category  |  Optional: Source, Note")
    generic_csv = st.file_uploader("Upload generic expense CSV", type=["csv"], key="generic_csv")
    if generic_csv:
        try:
            df_generic = pd.read_csv(generic_csv)
            df_generic.columns = [c.strip() for c in df_generic.columns]
            if "Source" not in df_generic.columns:
                df_generic["Source"] = "CSV Upload"
            st.dataframe(df_generic.head(10), use_container_width=True)
            if st.button("➕ Add CSV Data to Ledger"):
                st.session_state.ledger = pd.concat(
                    [st.session_state.ledger, df_generic], ignore_index=True
                )
                save_expenses(st.session_state.ledger)
                st.success("CSV data added!")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# ─────────────────────────────────────────────
# TAB 4 — DASHBOARD & ADVICE (Features 3 + 4)
# ─────────────────────────────────────────────
with tab4:
    if st.session_state.ledger.empty:
        st.info("Add some expenses first to see your dashboard.")
    else:
        ledger = st.session_state.ledger.copy()
        ledger["Amount"] = pd.to_numeric(ledger["Amount"], errors="coerce").fillna(0)

        # ── KPI row ──────────────────────────
        total_spent = ledger["Amount"].sum()
        num_txns    = len(ledger)
        sources     = ledger["Source"].nunique()

        k1, k2, k3 = st.columns(3)
        k1.metric("💸 Total Spent", f"₹{total_spent:,.0f}")
        k2.metric("🧾 Transactions", num_txns)
        k3.metric("📡 Sources", sources)

        st.divider()

        # ── Charts ───────────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Spending by Category")
            by_cat = ledger.groupby("Category")["Amount"].sum().sort_values(ascending=False)
            st.bar_chart(by_cat)

        with col_b:
            st.subheader("Spending by Source")
            by_src = ledger.groupby("Source")["Amount"].sum()
            st.bar_chart(by_src)

        # ── Budget progress bars ──────────────
        if st.session_state.budgets:
            st.divider()
            st.subheader("🎯 Budget Progress")
            spending_map = ledger.groupby("Category")["Amount"].sum()
            for cat, limit in st.session_state.budgets.items():
                spent = spending_map.get(cat, 0)
                pct   = min(spent / limit, 1.0) if limit > 0 else 0
                color = "🔴" if pct >= 1 else ("🟡" if pct >= 0.8 else "🟢")
                st.write(f"{color} **{cat}**: ₹{spent:.0f} / ₹{limit:.0f}")
                st.progress(pct)

        # ── Transactions table ────────────────
        st.divider()
        st.subheader("📋 All Transactions")
        display_cols = [c for c in ["Date", "Amount", "Category", "Source", "Note"] if c in ledger.columns]
        st.dataframe(
            ledger[display_cols].sort_values("Date", ascending=False),
            use_container_width=True
        )

        # ── AI Advice ────────────────────────
        st.divider()
        st.subheader("🤖 AI Financial Advice")
        if st.button("✨ Generate Personalised Advice"):
            with st.spinner("Analysing your spending habits..."):
                advice = get_financial_advice(ledger, guru_text)
            st.markdown(f"### 💡 Personalised Advice\n\n{advice}")
