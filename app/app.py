import html
import streamlit as st
from src.agent.support_agent import AmericanAirSupportAgent


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AmericanAir AI Support Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>

    /* ---------- Global ---------- */
    .stApp {
        background: #f5f7fb;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* ---------- Header ---------- */
    .hero {
        background: linear-gradient(135deg, #0b1f4d 0%, #173b8f 100%);
        padding: 28px 32px;
        border-radius: 18px;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
    }

    .hero-title {
        font-size: 30px;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 15px;
        opacity: 0.88;
        line-height: 1.5;
    }

    /* ---------- Section ---------- */
    .section-title {
        font-size: 20px;
        font-weight: 750;
        color: #111827;
        margin: 22px 0 10px 0;
    }

    .section-caption {
        color: #6b7280;
        font-size: 13px;
        margin-bottom: 12px;
    }

    /* ---------- Input Card ---------- */
    .input-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 3px 14px rgba(15, 23, 42, 0.05);
    }

    /* ---------- Metric Cards ---------- */
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px;
        min-height: 115px;
        box-shadow: 0 3px 14px rgba(15, 23, 42, 0.04);
    }

    .metric-label {
        color: #6b7280;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #111827;
        font-size: 20px;
        font-weight: 800;
        line-height: 1.3;
    }

    /* ---------- Decision ---------- */
    .decision-auto {
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        color: #166534;
        padding: 10px 14px;
        border-radius: 10px;
        display: inline-block;
        font-weight: 800;
        font-size: 14px;
    }

    .decision-escalate {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
        padding: 10px 14px;
        border-radius: 10px;
        display: inline-block;
        font-weight: 800;
        font-size: 14px;
    }

    /* ---------- Reply ---------- */
    .reply-card {
        background: white;
        border: 1px solid #dbeafe;
        border-left: 6px solid #2563eb;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.07);
    }

    .reply-text {
        color: #1e3a8a;
        font-size: 18px;
        line-height: 1.65;
        font-weight: 500;
    }

    /* ---------- Reason ---------- */
    .reason-card {
        background: #eff6ff;
        border: 1px solid #dbeafe;
        border-radius: 12px;
        padding: 16px;
        color: #1e40af;
        line-height: 1.55;
    }

    /* ---------- Evidence ---------- */
    .evidence-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
    }

    .evidence-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }

    .evidence-title {
        color: #111827;
        font-weight: 800;
        font-size: 14px;
    }

    .similarity {
        background: #eff6ff;
        color: #1d4ed8;
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 12px;
        font-weight: 800;
    }

    .evidence-label {
        color: #6b7280;
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 8px;
        margin-bottom: 5px;
    }

    .evidence-text {
        color: #374151;
        font-size: 14px;
        line-height: 1.55;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: #0f172a;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .sidebar-box {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 14px;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD AGENT
# =========================================================
@st.cache_resource
def load_agent():
    return AmericanAirSupportAgent(
        "data/american_air_support_pairs.csv"
    )


agent = load_agent()


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">✈️ AmericanAir AI Support Agent</div>
        <div class="hero-subtitle">
            AI customer-support agent grounded in historical
            AmericanAir support conversations
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("## 🤖 Agent")

    st.markdown(
        """
        <div class="sidebar-box">
            <b>Pipeline</b><br><br>
            1. Intent classification<br>
            2. Historical retrieval<br>
            3. Grounded reply generation<br>
            4. Escalation decision
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-box">
            <b>Components</b><br><br>
            • Hybrid classifier<br>
            • Sentence embeddings<br>
            • Historical evidence retrieval<br>
            • GPT-OSS 20B<br>
            • Rule-based escalation
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Hiver SDE Intern Take-Home Assignment")


# =========================================================
# CUSTOMER MESSAGE
# =========================================================
st.markdown(
    '<div class="section-title">💬 Customer Message</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-caption">Enter a customer support request to analyze.</div>',
    unsafe_allow_html=True,
)

customer_message = st.text_area(
    "Customer message",
    placeholder="Example: My flight was cancelled and I need help getting rebooked.",
    height=125,
    label_visibility="collapsed",
)

# =========================================================
# EXAMPLES
# =========================================================
st.markdown(
    '<div class="section-caption">Quick examples</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    flight_example = st.button(
        "✈️ Flight",
        use_container_width=True,
    )

with c2:
    baggage_example = st.button(
        "🧳 Baggage",
        use_container_width=True,
    )

with c3:
    refund_example = st.button(
        "💰 Refund",
        use_container_width=True,
    )

with c4:
    loyalty_example = st.button(
        "⭐ Loyalty",
        use_container_width=True,
    )


if flight_example:
    customer_message = (
        "My flight was cancelled and I need help getting rebooked."
    )

if baggage_example:
    customer_message = (
        "My suitcase was lost and I need help finding it."
    )

if refund_example:
    customer_message = (
        "I need a refund for my cancelled flight."
    )

if loyalty_example:
    customer_message = (
        "I just reached Executive Platinum and need help with my benefits."
    )


# =========================================================
# ANALYZE BUTTON
# =========================================================
st.markdown("")

analyze = st.button(
    "🔍 Analyze Customer Message",
    type="primary",
    use_container_width=True,
)


# =========================================================
# RESULTS
# =========================================================
if analyze:

    if not customer_message.strip():
        st.warning("Please enter a customer message.")

    else:

        with st.spinner("Analyzing customer message..."):
            result = agent.respond(customer_message.strip())

        st.divider()

        # -------------------------------------------------
        # ANALYSIS
        # -------------------------------------------------
        st.markdown(
            '<div class="section-title">📊 AI Analysis</div>',
            unsafe_allow_html=True,
        )

        cases = result.get("cases")
        evidence_count = len(cases) if cases is not None else 0

        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Intent</div>
                    <div class="metric-value">
                        {html.escape(result["intent"])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m2:
            decision = result["decision"]

            if decision == "AUTO_HANDLE":
                badge = (
                    '<span class="decision-auto">✅ AUTO_HANDLE</span>'
                )
            else:
                badge = (
                    '<span class="decision-escalate">⚠️ ESCALATE</span>'
                )

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Action</div>
                    <div style="margin-top:10px;">
                        {badge}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Historical Evidence</div>
                    <div class="metric-value">
                        {evidence_count} cases
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -------------------------------------------------
        # DECISION REASON
        # -------------------------------------------------
        st.markdown(
            '<div class="section-title">🧾 Decision Reason</div>',
            unsafe_allow_html=True,
        )

        reason = html.escape(result["reason"])

        st.markdown(
            f"""
            <div class="reason-card">
                {reason}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # GENERATED REPLY
        # -------------------------------------------------
        st.markdown(
            '<div class="section-title">💡 Generated Reply</div>',
            unsafe_allow_html=True,
        )

        reply = result.get("reply")

        if reply:

            reply = html.escape(reply)

            st.markdown(
                f"""
                <div class="reply-card">
                    <div class="reply-text">
                        {reply}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            st.warning("No automated reply was generated.")

        # -------------------------------------------------
        # HISTORICAL EVIDENCE
        # -------------------------------------------------
        st.markdown(
            '<div class="section-title">📚 Historical Evidence</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-caption">
                Historical AmericanAir conversations retrieved to
                ground the generated response.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if cases is not None and not cases.empty:

            for i, (_, row) in enumerate(
                cases.iterrows(),
                start=1,
            ):

                similarity = float(row["similarity"])

                customer_text = html.escape(
                    str(row["customer_text"])
                )

                historical_response = html.escape(
                    str(row["americanair_response"])
                )

                with st.container(border=True):
                    col1, col2 = st.columns([6, 1])

                    with col1:
                        st.markdown(f"### Evidence {i}")

                    with col2:
                        st.metric("Similarity", f"{similarity:.3f}")

                    st.markdown("**Customer**")
                    st.write(customer_text)

                    st.markdown("**Historical AmericanAir Response**")
                    st.write(historical_response)

        else:

            st.warning(
                "No sufficiently strong historical evidence found."
            )