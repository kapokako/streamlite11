import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

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

tab1, tab2, tab3, tab4 = st.tabs(["📈 Graphiques", "📊 Tables", "🔍 Recherche", "📋 Analyse Détaillée"])

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

with tab4:
    st.markdown("## 📋 Analyse Détaillée des Spreads")
    st.markdown("*Cette page présente une analyse complète des spreads par secteur, rating et échéance*")
    
    # Calcul des moyennes par dimension
    spread_by_sector = df.groupby('secteur')['spread'].agg(['mean', 'std', 'count']).round(2)
    spread_by_rating = df.groupby('rating')['spread'].agg(['mean', 'std', 'count']).round(2)
    spread_by_maturity = df.groupby('fourchette_annee')['spread'].agg(['mean', 'std', 'count']).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏭 Spreads par Secteur")
        
        # Graphique en barres horizontal pour les secteurs
        fig_sector = px.bar(
            x=spread_by_sector['mean'], 
            y=spread_by_sector.index,
            orientation='h',
            title="Spread Moyen par Secteur",
            labels={'x': 'Spread Moyen (bps)', 'y': 'Secteur'},
            color=spread_by_sector['mean'],
            color_continuous_scale='RdYlBu_r'
        )
        fig_sector.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_sector, use_container_width=True)
        
        # Box plot pour la distribution par secteur
        fig_box_sector = px.box(df, x='secteur', y='spread', title="Distribution des Spreads par Secteur")
        fig_box_sector.update_xaxes(tickangle=45)
        fig_box_sector.update_layout(height=400)
        st.plotly_chart(fig_box_sector, use_container_width=True)
    
    with col2:
        st.subheader("⭐ Spreads par Rating")
        
        # Graphique en barres pour les ratings
        # Trier par rating_num pour avoir un ordre logique
        rating_data = spread_by_rating.copy()
        rating_data['rating_num'] = rating_data.index.map(alpha_to_num)
        rating_data = rating_data.sort_values('rating_num', ascending=False)
        
        fig_rating = px.bar(
            x=rating_data.index,
            y=rating_data['mean'],
            title="Spread Moyen par Rating",
            labels={'x': 'Rating', 'y': 'Spread Moyen (bps)'},
            color=rating_data['mean'],
            color_continuous_scale='RdYlBu_r'
        )
        fig_rating.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_rating, use_container_width=True)
        
        # Graphique linéaire montrant la courbe de spread par rating
        fig_curve = px.line(
            x=rating_data['rating_num'],
            y=rating_data['mean'],
            title="Courbe des Spreads par Qualité de Crédit",
            labels={'x': 'Rating (échelle numérique)', 'y': 'Spread Moyen (bps)'},
            markers=True
        )
        fig_curve.update_layout(height=400)
        st.plotly_chart(fig_curve, use_container_width=True)
    
    st.subheader("📅 Spreads par Échéance")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Graphique des spreads par maturité
        fig_maturity = px.bar(
            x=spread_by_maturity.index,
            y=spread_by_maturity['mean'],
            title="Spread Moyen par Échéance",
            labels={'x': 'Échéance', 'y': 'Spread Moyen (bps)'},
            color=spread_by_maturity['mean'],
            color_continuous_scale='viridis'
        )
        fig_maturity.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_maturity, use_container_width=True)
    
    with col4:
        # Violin plot pour montrer la distribution par échéance
        fig_violin = px.violin(df, x='fourchette_annee', y='spread', 
                              title="Distribution des Spreads par Échéance",
                              box=True)
        fig_violin.update_layout(height=400)
        st.plotly_chart(fig_violin, use_container_width=True)
    
    st.subheader("🎯 Analyse Multidimensionnelle")
    
    # Heatmap croisée Rating vs Échéance
    pivot_heatmap = df.pivot_table(
        index='rating', 
        columns='fourchette_annee', 
        values='spread', 
        aggfunc='mean'
    )
    
    # Réordonner les ratings
    rating_order = sorted(pivot_heatmap.index, key=lambda x: alpha_to_num[x], reverse=True)
    pivot_heatmap = pivot_heatmap.reindex(rating_order)
    
    fig_heatmap = px.imshow(
        pivot_heatmap.values,
        x=pivot_heatmap.columns,
        y=pivot_heatmap.index,
        title="Heatmap des Spreads : Rating vs Échéance",
        labels={'x': 'Échéance', 'y': 'Rating', 'color': 'Spread Moyen (bps)'},
        color_continuous_scale='RdYlBu_r',
        aspect='auto'
    )
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Graphique radar par secteur
    st.subheader("🕸️ Profil Radar des Secteurs")
    
    # Calculer les moyennes par secteur et rating
    sector_rating = df.groupby(['secteur', 'rating'])['spread'].mean().unstack(fill_value=0)
    
    # Sélectionner quelques ratings clés pour le radar
    key_ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B']
    available_ratings = [r for r in key_ratings if r in sector_rating.columns]
    
    if available_ratings:
        fig_radar = go.Figure()
        
        for sector in sector_rating.index[:5]:  # Limiter à 5 secteurs pour la lisibilité
            fig_radar.add_trace(go.Scatterpolar(
                r=sector_rating.loc[sector, available_ratings],
                theta=available_ratings,
                fill='toself',
                name=sector
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, sector_rating[available_ratings].max().max()]
                )),
            showlegend=True,
            title="Profil des Spreads par Secteur et Rating"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Tableaux de synthèse
    st.subheader("📊 Tableaux de Synthèse")
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.markdown("**Top 5 Secteurs - Spreads les plus élevés**")
        top_sectors = spread_by_sector.sort_values('mean', ascending=False).head()
        st.dataframe(top_sectors[['mean', 'count']])
    
    with col6:
        st.markdown("**Ratings - Vue d'ensemble**")
        rating_summary = spread_by_rating.sort_values('mean', ascending=False)
        st.dataframe(rating_summary[['mean', 'count']])
    
    with col7:
        st.markdown("**Échéances - Vue d'ensemble**")
        maturity_summary = spread_by_maturity.sort_values('mean', ascending=False)
        st.dataframe(maturity_summary[['mean', 'count']])
    
    # Insights automatiques
    st.subheader("💡 Points Clés")
    
    max_spread_sector = spread_by_sector['mean'].idxmax()
    min_spread_sector = spread_by_sector['mean'].idxmin()
    max_spread_rating = spread_by_rating['mean'].idxmax()
    min_spread_rating = spread_by_rating['mean'].idxmin()
    
    insights = f"""
    **Observations principales :**
    
    • **Secteur le plus risqué** : {max_spread_sector} ({spread_by_sector.loc[max_spread_sector, 'mean']:.0f} bps en moyenne)
    • **Secteur le moins risqué** : {min_spread_sector} ({spread_by_sector.loc[min_spread_sector, 'mean']:.0f} bps en moyenne)
    • **Rating le plus pénalisé** : {max_spread_rating} ({spread_by_rating.loc[max_spread_rating, 'mean']:.0f} bps en moyenne)
    • **Rating le mieux traité** : {min_spread_rating} ({spread_by_rating.loc[min_spread_rating, 'mean']:.0f} bps en moyenne)
    • **Nombre total d'obligations** : {len(df):,}
    • **Spread moyen global** : {df['spread'].mean():.0f} bps
    """
    
    st.markdown(insights)

st.markdown("---")
st.markdown("*App Streamlit - analyse des spreads obligataires.*")
