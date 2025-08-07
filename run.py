import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Mapping numérique ↔ alphabétique pour les ratings
dict_num_to_alpha = {
    21: 'AAA', 20: 'AA+', 19: 'AA', 18: 'AA-', 17: 'A+', 16: 'A', 15: 'A-',
    14: 'BBB+', 13: 'BBB', 12: 'BBB-', 11: 'BB+', 10: 'BB', 9: 'BB-',
    8: 'B+', 7: 'B', 6: 'B-', 5: 'CCC+', 4: 'CCC', 3: 'CCC-', 2: 'CC', 1: 'SD'
}
alpha_to_num = {v: k for k, v in dict_num_to_alpha.items()}

# Configuration de la page
st.set_page_config(page_title="Analyse des Spreads Obligataires", layout="wide")

@st.cache_data
# Chargement et normalisation des données
def load_data():
    df = pd.read_excel('obligations.xlsx', usecols='B:F', names=[
        'secteur', 'spread', 'emission_annee', 'fourchette_annee', 'rating'
    ])
    # Conversion si rating numérique
    if df['rating'].dtype.kind in 'iufc':
        df['rating'] = df['rating'].map(dict_num_to_alpha)
    # Code numérique pour les axes
    df['rating_num'] = df['rating'].map(alpha_to_num)
    return df

# Chargement des données
df = load_data()

# Interface
st.markdown("# Analyse des Spreads Obligataires 📊")

# Onglets
tab1, tab2, tab3 = st.tabs(["Graphiques", "Tableaux", "Recherche"])

with tab1:
    st.header("Visualisations Interactives")
    # Filtres
    cols = st.columns(3)
    secteur_sel = cols[0].multiselect("Secteur", df['secteur'].unique(), default=df['secteur'].unique())
    echeance_sel = cols[1].multiselect("Échéance", df['fourchette_annee'].unique(), default=df['fourchette_annee'].unique())
    rating_min, rating_max = df['rating_num'].min(), df['rating_num'].max()
    rating_sel = cols[2].slider("Rating", int(rating_min), int(rating_max), (int(rating_min), int(rating_max)))

    df_f = df[
        df['secteur'].isin(secteur_sel) &
        df['fourchette_annee'].isin(echeance_sel) &
        df['rating_num'].between(rating_sel[0], rating_sel[1])
    ]

    # Heatmap animée
    fig1 = px.density_heatmap(
        df_f,
        x='fourchette_annee', y='rating_num', z='spread', histfunc='avg',
        animation_frame='secteur',
        labels={'fourchette_annee':'Échéance', 'rating_num':'Rating', 'spread':'Spread moyen'},
        title='Heatmap des spreads moyens par secteur'
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Bubble chart global
    grouped = (
        df_f.groupby(['secteur', 'fourchette_annee', 'rating_num'], as_index=False)['spread']
        .mean()
    )
    # Transformation d'échéance en code pour l'axe X
    grouped['bucket_code'] = grouped['fourchette_annee'].astype('category').cat.codes
    fig2 = px.scatter(
        grouped,
        x='bucket_code', y='rating_num', size='spread', opacity=0.6,
        labels={'bucket_code':'Échéance code', 'rating_num':'Rating', 'spread':'Spread moyen'},
        title='Bubble Chart: Spread moyen par bucket et rating'
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Graphique 3D secteur
    st.subheader("Spread moyen par secteur (3D)")
    mean_sec = df.groupby('secteur')['spread'].mean().reset_index()
    x_vals = list(range(len(mean_sec)))
    z_vals = mean_sec['spread'].tolist()
    fig3 = go.Figure()
    for xi, zi, sect in zip(x_vals, z_vals, mean_sec['secteur']):
        # Ligne verticale
        fig3.add_trace(go.Scatter3d(
            x=[xi, xi], y=[0, 0], z=[0, zi], mode='lines', line=dict(width=8)
        ))
        # Marqueur
        fig3.add_trace(go.Scatter3d(
            x=[xi], y=[0], z=[zi], mode='markers+text', text=[f"{zi:.2f}"],
            textposition='top center'
        ))
    fig3.update_layout(
        scene=dict(
            xaxis=dict(tickmode='array', tickvals=x_vals, ticktext=mean_sec['secteur']),
            yaxis=dict(title=''), zaxis=dict(title='Spread moyen')
        ), title='Surface 3D: Spread moyen par secteur'
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.header("Statistiques & Matrice")
    st.write(df['spread'].describe())
    pivot = df_f.pivot_table(index='rating', columns='fourchette_annee', values='spread', aggfunc='mean').fillna(0)
    st.dataframe(pivot)

with tab3:
    st.header("Recherche d'un spread")
n    secteur_q = st.selectbox("Secteur", df['secteur'].unique())
n    four_q = st.selectbox("Échéance", df['fourchette_annee'].unique())
n    rating_q = st.selectbox("Rating", df['rating'].unique())
    sub = df[(df['secteur']==secteur_q)&(df['fourchette_annee']==four_q)&(df['rating']==rating_q)]
    if not sub.empty:
        st.metric("Spread moyen estimé", f"{sub['spread'].mean():.2f}")
    else:
        st.warning("Pas de données exactes")

st.markdown("---")
st.markdown("*Application Streamlit - visualisation des spreads obligataires.*")
