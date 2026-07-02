import streamlit as st

# ✅ ALWAYS INITIALIZE SAFELY
def init_session():
    if "role" not in st.session_state:
        st.session_state["role"] = None


def login():

    # ✅ Ensure session is initialized first
    init_session()

    st.sidebar.title("🔐 Login")

    role = st.sidebar.selectbox("Select Role", ["User", "Admin"])

    # -------------------------
    # ✅ ADMIN LOGIN
    # -------------------------
    if role == "Admin":

        password = st.sidebar.text_input("Enter Admin Password", type="password")

        if st.sidebar.button("Login"):

            if password == "admin123":
                st.session_state["role"] = "ADMIN"
                st.sidebar.success("✅ Admin logged in")

            else:
                st.sidebar.error("❌ Invalid password")

    # -------------------------
    # ✅ USER LOGIN
    # -------------------------
    else:
        if st.sidebar.button("Continue as User"):
            st.session_state["role"] = "USER"

    # -------------------------
    # ✅ SAFE ROLE ACCESS (FIX)
    # -------------------------
    if st.session_state.get("role"):
        if st.sidebar.button("Logout"):
            st.session_state["role"] = None
            st.sidebar.success("✅ Logged out")

    return st.session_state.get("role")
 