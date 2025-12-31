import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Page Configuration
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

st.title("🛍️ Mall Customer Segmentation Dashboard")
st.markdown("""
Is app ke zariye aap Customers ko unki **Income** aur **Spending Score** ki bunyaad par groups mein baant sakte hain.
""")

# ==========================================
# 1. LOAD DATASET
# ==========================================
@st.cache_data
def load_data():
    # Kaggle ya Local path
    try:
        df = pd.read_csv('Mall_Customers.csv')
    except:
        # Agar file nahi milti to synthetic data bana lein demo ke liye
        np.random.seed(42)
        data = {
            "CustomerID": range(1, 201),
            "Annual Income (k$)": np.random.randint(15, 140, 200),
            "Spending Score (1-100)": np.random.randint(1, 100, 200)
        }
        df = pd.DataFrame(data)
    return df

df = load_data()

# Sidebar for Settings
st.sidebar.header("Model Settings")
algorithm = st.sidebar.selectbox("Select Algorithm", ["K-Means", "DBSCAN"])

# Data Selection
X = df.iloc[:, [df.columns.get_loc("Annual Income (k$)"), 
                df.columns.get_loc("Spending Score (1-100)")]].values

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 2. CLUSTERING LOGIC
# ==========================================
if algorithm == "K-Means":
    k_value = st.sidebar.slider("Number of Clusters (k)", 2, 10, 5)
    model = KMeans(n_clusters=k_value, init='k-means++', n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    
else: # DBSCAN
    eps_value = st.sidebar.slider("Epsilon (eps)", 0.1, 1.0, 0.3, 0.1)
    min_samples = st.sidebar.slider("Min Samples", 1, 10, 5)
    model = DBSCAN(eps=eps_value, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    # Silhouette score handle karna agar sirf 1 cluster bane
    if len(np.unique(labels)) > 1:
        score = silhouette_score(X_scaled, labels)
    else:
        score = 0

# ==========================================
# 3. VISUALIZATION
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"{algorithm} Clustering Result")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis', s=100, ax=ax, edgecolor='black')
    
    if algorithm == "K-Means":
        # Centroids ko show karna
        centers = scaler.inverse_transform(model.cluster_centers_)
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')
    
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    st.pyplot(fig)

with col2:
    st.metric("Silhouette Score", f"{score:.4f}")
    st.write("### Data Preview")
    st.dataframe(df.head(10))

    # Download Results
    df['Cluster'] = labels
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Clustered Data", csv, "customers_segmented.csv", "text/csv")

st.info("**Tip:** K-Means ke liye 'k=5' aur DBSCAN ke liye 'eps=0.3' is dataset par behtareen results dete hain.")