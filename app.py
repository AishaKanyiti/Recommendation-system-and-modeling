# --- app.py : Recommender System Streamlit App ---

import streamlit as st
import pickle
import scipy.sparse as sp
import polars as pl
import os

# --- Setup Paths ---
BASE_PATH = os.getcwd()  # works for Streamlit Cloud and local
st.write("📂 Files in current folder:", os.listdir(BASE_PATH))  # Debugging check

# --- Load Data & Model ---
@st.cache_resource
def load_data():
    # Load sparse user-item matrix
    UI_csr = sp.load_npz(os.path.join(BASE_PATH, "UI_csr.npz"))

    # Load user mappings
    with open(os.path.join(BASE_PATH, "user_mappings.pkl"), "rb") as f:
        user_id_to_idx, idx_to_user_id = pickle.load(f)

    # Load item mappings
    with open(os.path.join(BASE_PATH, "item_mappings.pkl"), "rb") as f:
        item_id_to_idx, idx_to_item_id = pickle.load(f)

    return UI_csr, user_id_to_idx, idx_to_user_id, item_id_to_idx, idx_to_item_id


@st.cache_resource
def load_model():
    with open(os.path.join(BASE_PATH, "als_model.pkl"), "rb") as f:
        als = pickle.load(f)
    return als


UI_csr, user_id_to_idx, idx_to_user_id, item_id_to_idx, idx_to_item_id = load_data()
als = load_model()

st.success("✅ Model, data, and mappings loaded successfully!")


# --- Recommend Function ---
def recommend_cf(user_raw_id, N=10):
    """Generate top-N recommendations for a given user ID"""
    if user_raw_id not in user_id_to_idx:
        st.warning(f"⚠️ User {user_raw_id} not found in mappings.")
        return pl.DataFrame({"itemid": [], "score": []})

    uidx = user_id_to_idx[user_raw_id]

    try:
        ids, scores = als.recommend(uidx, UI_csr[uidx], N=N, recalculate_user=True)
    except IndexError:
        st.warning("⚠️ No recommendations found (IndexError).")
        return pl.DataFrame({"itemid": [], "score": []})

    recs = pl.DataFrame({
        "itemid": [idx_to_item_id[i] for i in ids],
        "score": scores
    })
    return recs


# --- Streamlit UI ---
st.title("🎯 Recommendation System")

user_input = st.text_input("Enter a User ID to get recommendations:")

if st.button("Get Recommendations"):
    if user_input.strip():
        recs = recommend_cf(user_input.strip(), N=10)
        if recs.shape[0] > 0:
            st.write("Top Recommendations:")
            st.dataframe(recs)
        else:
            st.write("No recommendations found for this user.")
    else:
        st.warning("⚠️ Please enter a valid User ID.")
