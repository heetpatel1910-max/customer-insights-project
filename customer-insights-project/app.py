import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Insights Dashboard",
    page_icon="🛍️",
    layout="wide"
)

#Load Data and keep them in the memory ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    rfm = pd.read_csv('outputs/rfm_final.csv')
    basket = pd.read_csv('outputs/basket_binary.csv', index_col=0)
    return rfm, basket

rfm, basket_binary = load_data()

#Measures how similar two customers are based on their purchase history. ───────────────────────────────────
@st.cache_data
def compute_similarity(basket):
    sim = cosine_similarity(basket)
    return pd.DataFrame(sim, index=basket.index, columns=basket.index)

customer_similarity_df = compute_similarity(basket_binary)

#
def get_recommendations(customer_id, n=5):
    customer_id = int(customer_id)
    if customer_id not in customer_similarity_df.index:
        return []
    similar = customer_similarity_df[customer_id]\
        .sort_values(ascending=False)[1:6].index
    customer_products = set(
        basket_binary.loc[customer_id][basket_binary.loc[customer_id] > 0].index
    )
    recommendations = {}
    for s in similar:
        similar_products = set(
            basket_binary.loc[s][basket_binary.loc[s] > 0].index
        )
        for p in similar_products - customer_products:
            recommendations[p] = recommendations.get(p, 0) + 1
    top = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n]
    return [p for p, _ in top]

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("🛍️ Customer Insights")
st.sidebar.markdown("AI-Powered Analytics Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🗂️ Segments", "⚠️ Churn Risk", "🛍️ Recommendations"]
)

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Customer Insights Overview")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(rfm):,}")
    col2.metric("Avg Order Frequency", f"{rfm['Frequency'].mean():.1f}")
    col3.metric("Avg Spend per Customer", f"${rfm['Monetary'].mean():,.0f}")
    col4.metric("Churn Rate", f"{rfm['Churned'].mean()*100:.1f}%")

    st.markdown("---")

    # Revenue by Segment
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Revenue by Segment")
        rev = rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#2a9d8f', '#457b9d', '#f4a261', '#e63946']
        rev.plot(kind='bar', ax=ax, color=colors)
        ax.set_xlabel("")
        ax.set_ylabel("Total Revenue ($)")
        ax.set_title("Revenue by Segment")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("👥 Customers by Segment")
        counts = rfm['Segment'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        wedges, _, autotexts = ax.pie(
            counts,
            labels=None,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            pctdistance=1.25,
            radius=0.75,
        )
        for t in autotexts:
            t.set_fontsize(9)
            t.set_fontweight('bold')

        # Nudge labels that are too close together
        positions = [np.array(t.get_position()) for t in autotexts]
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < 0.25:
                    direction = diff / (dist + 1e-9)
                    shift = (0.25 - dist) / 2 + 0.06
                    positions[i] += direction * shift
                    positions[j] -= direction * shift
                    autotexts[i].set_position(tuple(positions[i]))
                    autotexts[j].set_position(tuple(positions[j]))

        ax.legend(
            wedges, counts.index,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
            frameon=False
        )
        ax.set_title("Customer Distribution")
        plt.tight_layout()
        st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — SEGMENTS
# ═══════════════════════════════════════════════════════════════
elif page == "🗂️ Segments":
    st.title("🗂️ Customer Segments")

    # Segment filter
    selected = st.multiselect(
        "Filter by Segment",
        options=rfm['Segment'].unique().tolist(),
        default=rfm['Segment'].unique().tolist()
    )
    filtered = rfm[rfm['Segment'].isin(selected)]

    # Scatter plot
    st.subheader("Recency vs Monetary by Segment")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_map = {
        'VIP': '#2a9d8f',
        'Loyal': '#457b9d',
        'Regular': '#f4a261',
        'Dormant': '#e63946'
    }
    for segment, color in colors_map.items():
        subset = filtered[filtered['Segment'] == segment]
        ax.scatter(subset['Recency'], subset['Monetary'],
                   label=segment, color=color, alpha=0.6, s=40)
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Monetary ($)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    # Segment summary table
    st.subheader("Segment Summary")
    summary = rfm.groupby('Segment').agg(
        Customers=('CustomerID', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean'),
        Total_Revenue=('Monetary', 'sum')
    ).round(1)
    st.dataframe(summary, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — CHURN RISK
# ═══════════════════════════════════════════════════════════════
elif page == "⚠️ Churn Risk":
    st.title("⚠️ Churn Risk Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Rate by Segment")
        churn = rfm.groupby('Segment')['Churned'].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        churn.sort_values(ascending=False).plot(
            kind='bar', ax=ax,
            color=['#e63946', '#f4a261', '#457b9d', '#2a9d8f']
        )
        ax.set_ylabel("Churn Rate (%)")
        ax.set_xlabel("")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Churn Probability Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(rfm['Churn_Probability'], bins=20,
                color='#e63946', edgecolor='white')
        ax.set_xlabel("Churn Probability (%)")
        ax.set_ylabel("Number of Customers")
        plt.tight_layout()
        st.pyplot(fig)

    # High risk table
    st.subheader("🔴 High Risk Customers — Worth Saving")
    threshold = st.slider("Churn Probability Threshold (%)", 50, 99, 80)
    high_risk = rfm[rfm['Churn_Probability'] >= threshold]\
        .sort_values('Monetary', ascending=False)\
        [['CustomerID', 'Segment', 'Recency',
          'Frequency', 'Monetary', 'Churn_Probability']]
    st.write(f"**{len(high_risk)} customers** above {threshold}% churn risk")
    st.dataframe(high_risk, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 4 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════
elif page == "🛍️ Recommendations":
    st.title("🛍️ Product Recommendations")

    customer_id = st.number_input(
        "Enter Customer ID",
        min_value=int(rfm['CustomerID'].min()),
        max_value=int(rfm['CustomerID'].max()),
        value=12346
    )

    if st.button("Get Recommendations"):
        # Customer info
        customer_info = rfm[rfm['CustomerID'] == customer_id]

        if customer_info.empty:
            st.error("Customer not found!")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Segment", customer_info['Segment'].values[0])
            col2.metric("Recency", f"{customer_info['Recency'].values[0]} days")
            col3.metric("Total Spend", f"${customer_info['Monetary'].values[0]:,.0f}")
            col4.metric("Churn Risk", f"{customer_info['Churn_Probability'].values[0]}%")

            st.markdown("---")
            st.subheader("🛍️ Recommended Products")

            recs = get_recommendations(customer_id)
            if recs:
                for i, product in enumerate(recs, 1):
                    st.markdown(f"**{i}.** {product}")
            else:
                st.warning("Not enough data to generate recommendations.")