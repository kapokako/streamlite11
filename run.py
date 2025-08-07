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

# Configuration Streamlit
st.set_page_config(page_title="Analyse des Spreads Obligataires", layout="wide")

# Chargement des données (colonnes B à E)
df = pd.read_excel(
    'obligations.xlsx', usecols='B:E',
    names=['secteur','spread','fourchette_annee','rating']
)
# Conversion ratings numériques -> alphabétiques si nécessaire
if df['rating'].dtype.kind in 'iufc':
    df['rating'] = df['rating'].astype(int).map(dict_num_to_alpha)
# Ajouter code numérique pour calculs
df['rating_num'] = df['rating'].map(alpha_to_num)

# Accueil
st.markdown(
    "<div style='text-align:center;padding:20px;background:#f0f8ff;'>"
    "<h1 style='color:#2E86AB;'>Analyse des Spreads Obligataires</h1>"
    "<p style='font-size:18px;'>Explorez, filtrez et estimez les spreads<br>par secteur, échéance & rating</p>"
    "</div>", unsafe_allow_html=True
)
st.markdown("---")

# Filtres globaux
secteurs = df['secteur'].unique().tolist()
echeances = df['fourchette_annee'].unique().tolist()
ratings_alpha = list(dict_num_to_alpha.values())

# Onglets
tab_graphs, tab_tables, tab_search = st.tabs(["📈 Graphiques","📊 Tables","🔍 Recherche"])

with tab_graphs:
    st.header("Visualisations Interactives")
    c1, c2, c3 = st.columns(3)
    sel_secteurs = c1.multiselect("Secteurs", options=secteurs, default=secteurs)
    sel_echeances = c2.multiselect("Échéances", options=echeances, default=echeances)
    sel_ratings = c3.multiselect("Rating", options=ratings_alpha, default=ratings_alpha)

    filt = df[
        df['secteur'].isin(sel_secteurs) &
        df['fourchette_annee'].isin(sel_echeances) &
        df['rating'].isin(sel_ratings)
    ]

    # Heatmap animée
    fig1 = px.density_heatmap(
        filt,
        x='fourchette_annee', y='rating_num', z='spread', histfunc='avg',
        animation_frame='secteur',
        labels={'fourchette_annee':'Échéance','rating_num':'Rating','spread':'Spread moyen'},
        title='Heatmap des spreads moyens par secteur'
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Bubble chart simplifié avec codes numériques pour l'échéance
    grp = filt.groupby(['secteur','fourchette_annee','rating_num'], as_index=False)['spread'].mean()
    # Convertir fourchette_annee en code numérique
    grp['bucket_code'] = grp['fourchette_annee'].astype('category').cat.codes
    # Assurer que spread est float
    grp['spread'] = grp['spread'].astype(float)
    # Tracé
    fig2 = px.scatter(
        grp,
        x='bucket_code',
        y='rating_num',
        size='spread',
        color='secteur',
        labels={
            'bucket_code':'Échéance (code)',
            'rating_num':'Rating',
            'spread':'Spread moyen'
        },
        title='Bubble Chart : spread moyen',
        hover_data={'fourchette_annee':True, 'spread':':.2f'}
    )
    # Remettre les labels d'échéance
    fig2.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=grp['bucket_code'],
            ticktext=grp['fourchette_annee'].unique()
        )
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 3D Scatter
    st.subheader('Spread moyen par Rating & Échéance (3D)')
    df3 = grp.copy()
    df3['bucket_code'] = df3['fourchette_annee'].astype('category').cat.codes
    fig3 = px.scatter_3d(
        df3,
        x='rating_num', y='bucket_code', z='spread', color='secteur',
        labels={'rating_num':'Rating','bucket_code':'Échéance code','spread':'Spread moyen'},
        title='3D Scatter : Spread moyen by Rating & Échéance'
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab_tables:
    st.header("Statistiques & Matrices")
    st.subheader("Descriptives Spread")
    st.write(df['spread'].describe())

    st.subheader("Pivot Rating×Échéance")
    pivot = df.pivot_table(
        index='rating', columns='fourchette_annee', values='spread', aggfunc='mean'
    ).fillna(0)
    st.dataframe(pivot)

with tab_search:
    st.header("Recherche Spread")
    sel_s = st.selectbox("Secteur", secteurs)
    sel_e = st.selectbox("Échéance", echeances)
    sel_r = st.selectbox("Rating", ratings_alpha)
    sub = df[(df['secteur']==sel_s)&(df['fourchette_annee']==sel_e)&(df['rating']==sel_r)]
    if not sub.empty:
        st.metric("Spread moyen estimé", f"{sub['spread'].mean():.2f}")
    else:
        st.warning("Pas de donnée exacte, spreads proches:")
        temp = df[(df['secteur']==sel_s)&(df['fourchette_annee']==sel_e)].assign(
            diff=abs(df['rating_num'] - alpha_to_num[sel_r])
        )
        vo = temp.sort_values('diff').head(5)
        st.table(vo[['rating','spread']])

st.markdown("---")
st.markdown("*Application Streamlit - analyse des spreads obligataires.*")
