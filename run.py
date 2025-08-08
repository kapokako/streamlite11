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

tab1, tab2, tab3, tab4 = st.tabs(["📋 Analyse Détaillée", "📈 Graphiques Interactifs", "📊 Tables & Données", "🔍 Recherche Avancée"])

with tab1:
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
        st.plotly_chart(fig_maturity, use_container_width=True

with tab2:
    st.markdown("## 📈 Graphiques Interactifs")
    st.markdown("*Explorez les données avec des graphiques personnalisables et des filtres interactifs*")
    
    # Filtres améliorés
    st.markdown("### 🎛️ Filtres de Données")
    c1, c2, c3, c4 = st.columns(4)
    sel_s = c1.multiselect("🏭 Secteurs", secteurs, default=secteurs)
    sel_e = c2.multiselect("📅 Échéances", echeances, default=echeances)
    sel_r = c3.multiselect("⭐ Ratings", ratings, default=ratings)
    
    # Nouveau filtre par spread
    spread_range = c4.slider(
        "💰 Fourchette Spread (bps)", 
        min_value=int(df['spread'].min()), 
        max_value=int(df['spread'].max()), 
        value=(int(df['spread'].min()), int(df['spread'].max()))
    )
    
    # Application des filtres
    df_f = df[
        df['secteur'].isin(sel_s) &
        df['fourchette_annee'].isin(sel_e) &
        df['rating'].isin(sel_r) &
        (df['spread'] >= spread_range[0]) &
        (df['spread'] <= spread_range[1])
    ]
    
    # Metrics en haut
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("📊 Obligations sélectionnées", f"{len(df_f):,}")
    col_m2.metric("📈 Spread moyen", f"{df_f['spread'].mean():.0f} bps")
    col_m3.metric("📉 Spread médian", f"{df_f['spread'].median():.0f} bps")
    col_m4.metric("📏 Écart-type", f"{df_f['spread'].std():.0f} bps")
    
    st.markdown("---")
    
    # Graphiques améliorés
    if not df_f.empty:
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("### 🔥 Heatmap Animée par Secteur")
            fig1 = px.density_heatmap(
                df_f, 
                x='fourchette_annee', 
                y='rating_num', 
                z='spread', 
                histfunc='avg', 
                animation_frame='secteur',
                labels={'fourchette_annee':'Échéance','rating_num':'Rating','spread':'Spread moyen (bps)'},
                color_continuous_scale='RdYlBu_r'
            )
            fig1.update_layout(height=450)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col_g2:
            st.markdown("### 📊 Histogramme des Spreads")
            fig_hist = px.histogram(
                df_f, 
                x='spread', 
                color='secteur',
                title="Distribution des Spreads par Secteur",
                labels={'spread': 'Spread (bps)', 'count': 'Nombre d\'obligations'},
                marginal="rug"
            )
            fig_hist.update_layout(height=450)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("### 🎯 Graphiques de Corrélation")
        col_g3, col_g4 = st.columns(2)
        
        with col_g3:
            # Scatter plot amélioré
            grp = df_f.groupby(['secteur','fourchette_annee','rating_num'], as_index=False)['spread'].mean()
            grp['bucket'] = grp['fourchette_annee'].astype('category').cat.codes
            
            fig2 = px.scatter(
                grp, 
                x='bucket', 
                y='rating_num', 
                size='spread',
                color='secteur',
                hover_data={'spread': ':.1f'},
                title="Relation Rating-Échéance-Spread",
                labels={'bucket':'Code Échéance','rating_num':'Rating (numérique)', 'spread': 'Spread (bps)'}
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col_g4:
            # Sunburst chart
            if len(df_f) > 0:
                fig_sun = px.sunburst(
                    df_f,
                    path=['secteur', 'fourchette_annee', 'rating'],
                    values='spread',
                    title="Répartition Hiérarchique des Spreads"
                )
                fig_sun.update_layout(height=400)
                st.plotly_chart(fig_sun, use_container_width=True)
        
        st.markdown("### 🌐 Vue 3D Interactive")
        grp = df_f.groupby(['secteur','fourchette_annee','rating_num'], as_index=False)['spread'].mean()
        grp['bucket'] = grp['fourchette_annee'].astype('category').cat.codes
        
        fig3 = px.scatter_3d(
            grp, 
            x='rating_num', 
            y='bucket', 
            z='spread', 
            color='secteur',
            size='spread',
            title="Vue 3D : Rating × Échéance × Spread",
            labels={'rating_num':'Rating','bucket':'Échéance','spread':'Spread (bps)'}
        )
        fig3.update_layout(height=600)
        st.plotly_chart(fig3, use_container_width=True)
    
    else:
        st.warning("⚠️ Aucune donnée ne correspond aux filtres sélectionnés")

with tab3:
    st.markdown("## 📊 Tables & Données")
    st.markdown("*Consultez les données détaillées et les statistiques descriptives*")
    
    # Statistiques globales améliorées
    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        st.markdown("### 📈 Statistiques Descriptives Globales")
        desc_stats = df['spread'].describe()
        desc_df = pd.DataFrame({
            'Statistique': ['Nombre', 'Moyenne', 'Écart-type', 'Minimum', '25%', 'Médiane', '75%', 'Maximum'],
            'Valeur (bps)': [f"{desc_stats['count']:.0f}", f"{desc_stats['mean']:.1f}", 
                           f"{desc_stats['std']:.1f}", f"{desc_stats['min']:.1f}",
                           f"{desc_stats['25%']:.1f}", f"{desc_stats['50%']:.1f}",
                           f"{desc_stats['75%']:.1f}", f"{desc_stats['max']:.1f}"]
        })
        st.dataframe(desc_df, use_container_width=True)
    
    with col_stat2:
        st.markdown("### 🎯 Répartition par Catégorie")
        
        # Comptages par catégorie
        sector_count = df['secteur'].value_counts()
        rating_count = df['rating'].value_counts()
        maturity_count = df['fourchette_annee'].value_counts()
        
        st.markdown("**Nombre d'obligations par secteur (Top 5):**")
        st.dataframe(sector_count.head().to_frame('Nombre'), use_container_width=True)
    
    # Table pivot interactive améliorée
    st.markdown("### 🏗️ Table Pivot Interactive")
    
    pivot_options = st.radio(
        "Choisissez la vue :",
        ["Rating × Échéance", "Secteur × Échéance", "Secteur × Rating"],
        horizontal=True
    )
    
    if pivot_options == "Rating × Échéance":
        pivot = df.pivot_table(
            index='rating', 
            columns='fourchette_annee', 
            values='spread', 
            aggfunc=['mean', 'count'],
            fill_value=0
        ).round(1)
    elif pivot_options == "Secteur × Échéance":
        pivot = df.pivot_table(
            index='secteur', 
            columns='fourchette_annee', 
            values='spread', 
            aggfunc=['mean', 'count'],
            fill_value=0
        ).round(1)
    else:  # Secteur × Rating
        pivot = df.pivot_table(
            index='secteur', 
            columns='rating', 
            values='spread', 
            aggfunc=['mean', 'count'],
            fill_value=0
        ).round(1)
    
    st.dataframe(pivot, use_container_width=True)
    
    # Données brutes avec filtres
    st.markdown("### 🔍 Données Brutes (échantillon)")
    
    # Filtres pour les données brutes
    col_f1, col_f2 = st.columns(2)
    filter_sector = col_f1.selectbox("Filtrer par secteur", ["Tous"] + secteurs)
    filter_rating = col_f2.selectbox("Filtrer par rating", ["Tous"] + ratings)
    
    # Application des filtres
    display_df = df.copy()
    if filter_sector != "Tous":
        display_df = display_df[display_df['secteur'] == filter_sector]
    if filter_rating != "Tous":
        display_df = display_df[display_df['rating'] == filter_rating]
    
    # Affichage avec tri
    sort_by = st.selectbox("Trier par", ["spread", "secteur", "rating", "fourchette_annee"])
    ascending = st.checkbox("Ordre croissant", value=True)
    
    display_df_sorted = display_df.sort_values(sort_by, ascending=ascending)
    
    st.markdown(f"**Affichage de {len(display_df_sorted)} obligations** (sur {len(df)} au total)")
    st.dataframe(
        display_df_sorted[['secteur', 'fourchette_annee', 'rating', 'spread']], 
        use_container_width=True
    )
    
    # Bouton de téléchargement
    csv = display_df_sorted.to_csv(index=False)
    st.download_button(
        label="💾 Télécharger les données filtrées (CSV)",
        data=csv,
        file_name="spreads_obligataires_filtered.csv",
        mime="text/csv"
    )

with tab4:
    st.markdown("## 🔍 Recherche Avancée")
    st.markdown("*Trouvez et estimez des spreads avec des outils de recherche sophistiqués*")
    
    # Mode de recherche
    search_mode = st.radio(
        "Mode de recherche :",
        ["🎯 Recherche Exacte", "🔍 Recherche Approximative", "📊 Comparaison Multiple"],
        horizontal=True
    )
    
    if search_mode == "🎯 Recherche Exacte":
        st.markdown("### Recherche par critères précis")
        col_s1, col_s2, col_s3 = st.columns(3)
        ss = col_s1.selectbox("Secteur", secteurs)
        se = col_s2.selectbox("Échéance", echeances)
        sr = col_s3.selectbox("Rating", ratings)
        
        sub = df[(df['secteur']==ss)&(df['fourchette_annee']==se)&(df['rating']==sr)]
        
        if not sub.empty:
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("📊 Spread moyen", f"{sub['spread'].mean():.2f} bps")
            col_m2.metric("📏 Écart-type", f"{sub['spread'].std():.2f} bps" if len(sub) > 1 else "N/A")
            col_m3.metric("🔢 Nombre d'obligations", len(sub))
            
            st.markdown("**Détail des obligations trouvées :**")
            st.dataframe(sub[['secteur', 'fourchette_annee', 'rating', 'spread']], use_container_width=True)
            
            if len(sub) > 1:
                fig_detail = px.histogram(sub, x='spread', title=f"Distribution - {ss} | {se} | {sr}")
                st.plotly_chart(fig_detail, use_container_width=True)
        else:
            st.warning("❌ Aucune donnée exacte trouvée")
            st.markdown("**Suggestions alternatives :**")
            
            # Recherche par secteur et échéance
            temp = df[(df['secteur']==ss)&(df['fourchette_annee']==se)]
            if not temp.empty:
                temp['diff'] = abs(temp['rating_num'] - alpha_to_num[sr])
                closest = temp.nsmallest(5, 'diff')
                
                st.markdown(f"**Même secteur ({ss}) et échéance ({se}), ratings proches :**")
                st.dataframe(closest[['rating', 'spread', 'diff']], use_container_width=True)
                
                fig_alt = px.bar(closest, x='rating', y='spread', 
                               title=f"Spreads alternatifs - {ss} | {se}")
                st.plotly_chart(fig_alt, use_container_width=True)
    
    elif search_mode == "🔍 Recherche Approximative":
        st.markdown("### Recherche par fourchettes")
        
        # Sélecteurs multiples pour recherche approximative
        col_a1, col_a2 = st.columns(2)
        selected_sectors = col_a1.multiselect("Secteurs d'intérêt", secteurs, default=secteurs[:3])
        selected_ratings = col_a2.multiselect("Ratings d'intérêt", ratings, default=ratings[:5])
        
        # Fourchette de spread recherchée
        spread_target = st.slider("Fourchette de spread recherchée (bps)", 
                                 int(df['spread'].min()), int(df['spread'].max()), 
                                 (100, 300))
        
        # Recherche
        approx_results = df[
            df['secteur'].isin(selected_sectors) &
            df['rating'].isin(selected_ratings) &
            (df['spread'] >= spread_target[0]) &
            (df['spread'] <= spread_target[1])
        ]
        
        if not approx_results.empty:
            st.success(f"✅ {len(approx_results)} obligations trouvées")
            
            # Statistiques des résultats
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            col_r1.metric("📊 Spread moyen", f"{approx_results['spread'].mean():.1f} bps")
            col_r2.metric("📉 Spread min", f"{approx_results['spread'].min():.1f} bps")
            col_r3.metric("📈 Spread max", f"{approx_results['spread'].max():.1f} bps")
            col_r4.metric("📏 Écart-type", f"{approx_results['spread'].std():.1f} bps")
            
            # Graphiques des résultats
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig_sector_approx = px.box(approx_results, x='secteur', y='spread', 
                                         title="Distribution par secteur")
                fig_sector_approx.update_xaxes(tickangle=45)
                st.plotly_chart(fig_sector_approx, use_container_width=True)
            
            with col_g2:
                fig_rating_approx = px.box(approx_results, x='rating', y='spread', 
                                         title="Distribution par rating")
                fig_rating_approx.update_xaxes(tickangle=45)
                st.plotly_chart(fig_rating_approx, use_container_width=True)
            
            # Table des résultats
            st.dataframe(approx_results.sample(min(20, len(approx_results))), use_container_width=True)
        else:
            st.error("❌ Aucun résultat trouvé avec ces critères")
    
    else:  # Comparaison Multiple
        st.markdown("### Comparaison de plusieurs profils")
        
        st.markdown("**Définissez jusqu'à 3 profils à comparer :**")
        
        profiles = []
        for i in range(3):
            st.markdown(f"**Profil {i+1} :**")
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            
            p_sector = col_p1.selectbox(f"Secteur {i+1}", [""] + secteurs, key=f"sector_{i}")
            p_maturity = col_p2.selectbox(f"Échéance {i+1}", [""] + echeances, key=f"maturity_{i}")
            p_rating = col_p3.selectbox(f"Rating {i+1}", [""] + ratings, key=f"rating_{i}")
            p_name = col_p4.text_input(f"Nom du profil {i+1}", value=f"Profil {i+1}", key=f"name_{i}")
            
            if p_sector and p_maturity and p_rating:
                profiles.append({
                    'name': p_name,
                    'secteur': p_sector,
                    'fourchette_annee': p_maturity,
                    'rating': p_rating
                })
        
        if profiles:
            st.markdown("### 📊 Résultats de la Comparaison")
            
            comparison_data = []
            for profile in profiles:
                result = df[
                    (df['secteur'] == profile['secteur']) &
                    (df['fourchette_annee'] == profile['fourchette_annee']) &
                    (df['rating'] == profile['rating'])
                ]
                
                if not result.empty:
                    comparison_data.append({
                        'Profil': profile['name'],
                        'Secteur': profile['secteur'],
                        'Échéance': profile['fourchette_annee'],
                        'Rating': profile['rating'],
                        'Spread Moyen (bps)': round(result['spread'].mean(), 2),
                        'Nb Obligations': len(result),
                        'Spread Min (bps)': round(result['spread'].min(), 2),
                        'Spread Max (bps)': round(result['spread'].max(), 2)
                    })
                else:
                    comparison_data.append({
                        'Profil': profile['name'],
                        'Secteur': profile['secteur'],
                        'Échéance': profile['fourchette_annee'],
                        'Rating': profile['rating'],
                        'Spread Moyen (bps)': "N/A",
                        'Nb Obligations': 0,
                        'Spread Min (bps)': "N/A",
                        'Spread Max (bps)': "N/A"
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Graphique de comparaison
                valid_data = [d for d in comparison_data if d['Nb Obligations'] > 0]
                if len(valid_data) > 1:
                    fig_comp = px.bar(
                        pd.DataFrame(valid_data), 
                        x='Profil', 
                        y='Spread Moyen (bps)',
                        title="Comparaison des Spreads Moyens",
                        color='Profil'
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)

 labels={'x': yen (bps)', 'y': 'Secteur'},
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
