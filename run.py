import streamlit as st
import pandas as pd
import plotly.express as px

# Mapping numérique ↔ alphabétique
rating_map = {21:'AAA',20:'AA+',19:'AA',18:'AA-',17:'A+',16:'A',15:'A-',14:'BBB+',13:'BBB',12:'BBB-',11:'BB+',10:'BB',9:'BB-',8:'B+',7:'B',6:'B-',5:'CCC+',4:'CCC',3:'CCC-',2:'CC',1:'SD'}
alpha_to_num = {v:k for k,v in rating_map.items()}

st.set_page_config(page_title="Analyse des Spreads Obligataires", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel('obligations.xlsx', usecols='B:E', names=['secteur','spread','fourchette_annee','rating'])
    if df['rating'].dtype.kind in 'iufc':
        df['rating'] = df['rating'].map(rating_map)
    df['rating_num'] = df['rating'].map(alpha_to_num)
    return df

df = load_data()

st.markdown("<div style='text-align:center;padding:20px;background:#f0f8ff;'><h1 style='color:#2E86AB;'>Analyse des Spreads Obligataires</h1><p style='font-size:18px;'>Explorez, filtrez et estimez les spreads par secteur, échéance & rating</p></div>", unsafe_allow_html=True)
st.markdown("---")

secteurs = df['secteur'].unique().tolist()
echeances = df['fourchette_annee'].unique().tolist()
ratings = list(rating_map.values())

tab1, tab2, tab3 = st.tabs(["📈 Graphiques","📊 Tables","🔍 Recherche"])

with tab1:
    c1, c2, c3 = st.columns(3)
    sel_s = c1.multiselect("Secteurs", secteurs, default=secteurs)
    sel_e = c2.multiselect("Échéances", echeances, default=echeances)
    sel_r = c3.multiselect("Ratings", ratings, default=ratings)
    df_f = df[df['secteur'].isin(sel_s)&df['fourchette_annee'].isin(sel_e)&df['rating'].isin(sel_r)]
    fig1 = px.density_heatmap(df_f, x='fourchette_annee', y='rating_num', z='spread', histfunc='avg', animation_frame='secteur', labels={'fourchette_annee':'Échéance','rating_num':'Rating','spread':'Spread moyen'})
    st.plotly_chart(fig1, use_container_width=True)
    grp = df_f.groupby(['secteur','fourchette_annee','rating_num'], as_index=False)['spread'].mean()
    grp['bucket'] = grp['fourchette_annee'].astype('category').cat.codes
    fig2 = px.scatter(grp, x='bucket', y='rating_num', size=grp['spread'].clip(0), color='secteur', labels={'bucket':'Échéance code','rating_num':'Rating'})
    st.plotly_chart(fig2, use_container_width=True)
    fig3 = px.scatter_3d(grp, x='rating_num', y='bucket', z='spread', color='secteur', labels={'rating_num':'Rating','bucket':'Échéance code'})
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.write(df['spread'].describe())
    pivot = df_f.pivot_table(index='rating', columns='fourchette_annee', values='spread', aggfunc='mean').fillna(0)
    st.dataframe(pivot)

with tab3:
    ss = st.selectbox("Secteur", secteurs)
    se = st.selectbox("Échéance", echeances)
    sr = st.selectbox("Rating", ratings)
    sub = df[(df['secteur']==ss)&(df['fourchette_annee']==se)&(df['rating']==sr)]
    if not sub.empty:
        st.metric("Spread moyen estimé", f"{sub['spread'].mean():.2f}")
    else:
        st.warning("Aucune donnée exacte. Affichage des plus proches résultats:")
        temp = df[(df['secteur']==ss)&(df['fourchette_annee']==se)].assign(diff=abs(df['rating_num']-alpha_to_num[sr]))
        vo = temp.sort_values('diff').head(5)
        st.write(f"{len(vo)} spreads trouvés")
        st.table(vo[['rating','spread']])
        st.bar_chart(vo.set_index('rating')['spread'])

st.markdown("---")
st.markdown("*App Streamlit - analyse des spreads obligataires.*")
