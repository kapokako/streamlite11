import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuration Streamlit
st.set_page_config(page_title="Analyse des Spreads Obligataires", layout="wide")

# Chargement des données avec cache
@st.cache_data
# Utilise st.cache_data pour memoire
def load_data():
    df = pd.read_excel(
        'obligations.xlsx',
        usecols='B:F',
        names=['secteur','spread','emission_annee','fourchette_annee','rating']
    )
    df['rating'] = df['rating'].astype(int)
    df['fourchette_annee'] = df['fourchette_annee'].astype(str)
    return df

# Charger données
df = load_data()

# Page d'accueil
st.markdown(
    "<div style='text-align:center;padding:20px;background:#f0f8ff;'>"
    "<h1 style='color:#2E86AB;'>Analyse des Spreads Obligataires</h1>"
    "<p style='font-size:18px;'>Explorez, filtrez et estimez les spreads<br>par secteur, échéance & rating</p>"
    "</div>", unsafe_allow_html=True
)
st.markdown("---")

# Préparer filtres globaux
secteurs = df['secteur'].unique().tolist()
echeances = df['fourchette_annee'].unique().tolist()
rating_min, rating_max = int(df['rating'].min()), int(df['rating'].max())

# Onglets
tab_graphs, tab_tables, tab_search = st.tabs(["📈 Graphiques","📊 Tables","🔍 Recherche"])

with tab_graphs:
    st.header("Visualisations Interactives")
    c1,c2,c3 = st.columns(3)
    filter_secteurs = c1.multiselect("Secteurs",secteurs,default=secteurs)
    filter_echeances = c2.multiselect("Échéances",echeances,default=echeances)
    filter_rating = c3.slider("Rating",rating_min,rating_max,(rating_min,rating_max),step=1)

    df_f = df[
        df['secteur'].isin(filter_secteurs)&
        df['fourchette_annee'].isin(filter_echeances)&
        df['rating'].between(filter_rating[0],filter_rating[1])
    ]

    # Heatmap animée
    fig1=px.density_heatmap(
        df_f,x='fourchette_annee',y='rating',z='spread',histfunc='avg',animation_frame='secteur',
        labels={'fourchette_annee':'Échéance','rating':'Rating','spread':'Spread moyen'},
        title='Heatmap des spreads moyens par secteur'
    )
    fig1.update_layout(coloraxis_colorbar_title='Spread')
    st.plotly_chart(fig1,use_container_width=True)

    # Bubble chart
    grp=df_f.groupby(['secteur','fourchette_annee','rating'],as_index=False)['spread'].mean()
    fig2=px.scatter(
        grp,x='fourchette_annee',y='rating',size='spread',color='secteur',
        labels={'fourchette_annee':'Échéance','rating':'Rating','spread':'Spread moyen'},
        title='Bubble Chart : spread moyen',hover_data={'spread':':.2f'}
    )
    st.plotly_chart(fig2,use_container_width=True)

    # Scatter 3D
    st.subheader('Spread moyen par Rating & Échéance (3D)')
    df3=df_f.groupby(['secteur','rating','fourchette_annee'],as_index=False)['spread'].mean()
    df3['bucket_code']=df3['fourchette_annee'].astype('category').cat.codes
    fig3=px.scatter_3d(
        df3,x='rating',y='bucket_code',z='spread',color='secteur',
        labels={'rating':'Rating','bucket_code':'Échéance code','spread':'Spread moyen','secteur':'Secteur'},
        title='3D Scatter : Spread moyen by Rating & Échéance',
        hover_data={'fourchette_annee':True,'spread':':.2f'}
    )
    fig3.update_layout(legend_title_text='Secteur')
    st.plotly_chart(fig3,use_container_width=True)

with tab_tables:
    st.header("Statistiques & Matrices")
    st.subheader("Descriptives Spread")
    st.write(df_f['spread'].describe())
    st.subheader("Pivot Rating×Échéance")
    pivot=df_f.pivot_table(index='rating',columns='fourchette_annee',values='spread',aggfunc='mean').fillna(0)
    st.dataframe(pivot)

with tab_search:
    st.header("Recherche Spread")
    sel_s=st.selectbox("Secteur",secteurs)
    sel_e=st.selectbox("Échéance",echeances)
    sel_r=st.slider("Rating",rating_min,rating_max,(rating_min+rating_max)//2,step=1)
    sub=df[(df['secteur']==sel_s)&(df['fourchette_annee']==sel_e)&(df['rating']==sel_r)]
    if not sub.empty:
        st.metric("Spread moyen estimé",f"{sub['spread'].mean():.2f}")
    else:
        st.warning("Pas de donnée exacte, spreads proches:")
        t=df[(df['secteur']==sel_s)&(df['fourchette_annee']==sel_e)].assign(diff=abs(df['rating']-sel_r))
        vo=t.sort_values('diff').head(5).drop(columns='diff')
        st.table(vo[['rating','spread']])

st.markdown("---")
st.markdown("*Application Streamlit - analyse des spreads obligataires.*")
