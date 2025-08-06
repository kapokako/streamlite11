# Pour lancer correctement cette application sur Streamlit Cloud, nommez le fichier app.py
import streamlit as st

# Config must occur before any other Streamlit calls
st.set_page_config(page_title="Analyse des Spreads Obligataires", layout="wide")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Chargement des données (cache pour performance)
@st.cache_data
# Utiliser st.cache_data pour stocker en cache le chargement du fichier
# si vous avez une connexion lente ou de gros fichiers
def load_data():
    df = pd.read_excel(
        'obligations.xlsx',
        usecols='B:F',
        names=['secteur', 'spread', 'emission_annee', 'fourchette_annee', 'rating']
    )
    df['rating'] = df['rating'].astype(int)
    df['fourchette_annee'] = df['fourchette_annee'].astype(str)
    return df

# Sidebar: bouton de rechargement
def sidebar_controls():
    st.sidebar.title("Contrôles")
    if st.sidebar.button("🔄 Recharger les données"):
        load_data.clear()
        st.experimental_rerun()

sidebar_controls()

# Charger les données
df = load_data()

# Page d'accueil stylée
st.markdown(
    "<div style='text-align:center;padding:20px;background-color:#f0f8ff;'>"
    "<h1 style='color:#2E86AB;'>Analyse des Spreads Obligataires</h1>"
    "<p style='font-size:18px;'>Explorez, filtrez et estimez les spreads<br>par secteur, échéance & rating</p>"
    "</div>",
    unsafe_allow_html=True
)
st.markdown("---")

# Onglets principaux
tab_graphs, tab_tables, tab_search = st.tabs(["📈 Graphiques", "📊 Tables", "🔍 Recherche Spread"])

# Données globales pour filtres
secteurs = df['secteur'].unique().tolist()
echeances = df['fourchette_annee'].unique().tolist()
rating_min, rating_max = int(df['rating'].min()), int(df['rating'].max())

with tab_graphs:
    st.header("Visualisations Interactives")
    # Filtres
    col1, col2, col3 = st.columns(3)
    secteur_filter = col1.multiselect("Secteurs", options=secteurs, default=secteurs)
    echeance_filter = col2.multiselect("Échéances", options=echeances, default=echeances)
    rating_filter = col3.slider("Rating (min-max)", rating_min, rating_max, (rating_min, rating_max), step=1)

    df_filt = df[
        df['secteur'].isin(secteur_filter) &
        df['fourchette_annee'].isin(echeance_filter) &
        df['rating'].between(rating_filter[0], rating_filter[1])
    ]

    # Heatmap animée par secteur
    fig1 = px.density_heatmap(
        df_filt,
        x='fourchette_annee', y='rating', z='spread', histfunc='avg',
        animation_frame='secteur',
        labels={'fourchette_annee':'Échéance','rating':'Rating','spread':'Spread moyen'},
        title="Heatmap des spreads moyens par secteur"
    )
    fig1.update_layout(coloraxis_colorbar_title='Spread')
    st.plotly_chart(fig1, use_container_width=True)

    # Bubble chart
    grouped = df_filt.groupby(['secteur','fourchette_annee','rating'], as_index=False)['spread'].mean()
    fig2 = px.scatter(
        grouped,
        x='fourchette_annee', y='rating', size='spread', color='secteur',
        title='Bubble Chart : spread moyen',
        labels={'fourchette_annee':'Échéance','rating':'Rating','spread':'Spread moyen'},
        hover_data={'spread':':.2f'}
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Graphique 3D interactif : spread par rating et échéance, coloré par secteur
    st.subheader("Spread moyen par Rating et Échéance (3D)")
    df_3d = df_filt.groupby(['secteur','rating','fourchette_annee'], as_index=False)['spread'].mean()
    df_3d['bucket_code'] = df_3d['fourchette_annee'].astype('category').cat.codes

    fig3 = px.scatter_3d(
        df_3d,
        x='rating', y='bucket_code', z='spread',
        color='secteur',
        labels={
            'rating':'Rating',
            'bucket_code':'Échéance (code)',
            'spread':'Spread moyen',
            'secteur':'Secteur'
        },
        title='3D Scatter : Spread moyen par Rating & Échéance',
        hover_data={'fourchette_annee':True, 'spread':':.2f'}
    )
    fig3.update_layout(legend_title_text='Secteur')
    st.plotly_chart(fig3, use_container_width=True)

with tab_tables:
    st.header("Tableaux de Données")
    st.subheader("Statistiques descriptives")
    st.write(df_filt['spread'].describe())

    st.subheader("Matrice Pivot Rating × Échéance")
    pivot = df_filt.pivot_table(
        index='rating',
        columns='fourchette_annee',
        values='spread',
        aggfunc='mean'
    ).fillna(0)
    st.dataframe(pivot)

with tab_search:
    st.header("Rechercher le Spread Moyen d'une Obligation")
    secteur_sel = st.selectbox("Secteur", options=secteurs)
    echeance_sel = st.selectbox("Échéance (Bucket)", options=echeances)
    rating_sel = st.slider("Rating (exact)", rating_min, rating_max, (rating_min+rating_max)//2, step=1)

    df_sel = df[(df['secteur']==secteur_sel)&(df['fourchette_annee']==echeance_sel)&(df['rating']==rating_sel)]
    if not df_sel.empty:
        st.metric(label="Spread Moyen Estimé", value=f"{df_sel['spread'].mean():.2f}")
    else:
        st.warning("Aucune donnée exacte. Voici les plus proches :")
        temp = df[(df['secteur']==secteur_sel)&(df['fourchette_annee']==echeance_sel)].assign(diff=abs(df['rating']-rating_sel))
        voisins = temp.sort_values('diff').head(5).drop(columns='diff')
        st.table(voisins[['rating','spread']])

# Footer
st.markdown("---")
st.markdown("*Application Streamlit pour l'analyse des spreads obligataires.*")
