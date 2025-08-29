import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io

# Configuration de la page
st.set_page_config(
    page_title="Bond Spread Analytics", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mapping numérique ↔ alphabétique
rating_map = {21:'AAA',20:'AA+',19:'AA',18:'AA-',17:'A+',16:'A',15:'A-',14:'BBB+',13:'BBB',12:'BBB-',11:'BB+',10:'BB',9:'BB-',8:'B+',7:'B',6:'B-',5:'CCC+',4:'CCC',3:'CCC-',2:'CC',1:'SD'}
alpha_to_num = {v:k for k,v in rating_map.items()}

# Styles CSS pour un look professionnel
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f1f3f4;
        border-radius: 8px;
        padding: 0 24px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2a5298;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path='obligations.xlsx'):
    """Charge les données depuis le fichier Excel"""
    try:
        df = pd.read_excel(file_path, usecols='A:E', names=['nom_obligation','secteur','spread','fourchette_annee','rating'])
        df = df.dropna()
        
        # Convertir les ratings numériques en alphabétiques si nécessaire
        if df['rating'].dtype.kind in 'iufc':
            df['rating'] = df['rating'].map(rating_map)
        
        # Nettoyer et valider les données
        df = df[df['rating'].notna()]
        df = df[df['rating'].isin(alpha_to_num.keys())]
        df['rating_num'] = df['rating'].map(alpha_to_num)
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {str(e)}")
        return pd.DataFrame()

def update_data(existing_df, new_df):
    """Met à jour les données existantes avec les nouvelles"""
    if existing_df.empty:
        return new_df
    
    # Identifier les obligations communes par nom
    common_bonds = new_df[new_df['nom_obligation'].isin(existing_df['nom_obligation'])]['nom_obligation']
    new_bonds = new_df[~new_df['nom_obligation'].isin(existing_df['nom_obligation'])]
    
    # Mettre à jour les obligations existantes
    updated_df = existing_df.copy()
    for bond_name in common_bonds:
        mask = updated_df['nom_obligation'] == bond_name
        new_data = new_df[new_df['nom_obligation'] == bond_name].iloc[0]
        updated_df.loc[mask, ['secteur', 'spread', 'fourchette_annee', 'rating', 'rating_num']] = [
            new_data['secteur'], new_data['spread'], new_data['fourchette_annee'], 
            new_data['rating'], new_data['rating_num']
        ]
    
    # Ajouter les nouvelles obligations
    final_df = pd.concat([updated_df, new_bonds], ignore_index=True)
    
    return final_df

# Chargement initial des données
if 'df_main' not in st.session_state:
    st.session_state.df_main = load_data()

df = st.session_state.df_main

# Header principal
st.markdown("""
<div class="main-header">
    <h1>🏦 Bond Spread Analytics Pro</h1>
    <p style="font-size: 18px; margin-top: 1rem;">Analyse professionnelle des spreads obligataires avec visualisation avancée</p>
</div>
""", unsafe_allow_html=True)

# Sidebar pour l'import et les contrôles
with st.sidebar:
    st.header("📊 Contrôles & Import")
    
    # Section import de données
    st.subheader("📥 Import de Données")
    uploaded_file = st.file_uploader(
        "Charger un fichier Excel",
        type=['xlsx', 'xls'],
        help="Le fichier doit contenir les colonnes : Nom, Secteur, Spread, Échéance, Rating"
    )
    
    if uploaded_file is not None:
        try:
            new_df = pd.read_excel(uploaded_file, usecols='A:E', names=['nom_obligation','secteur','spread','fourchette_annee','rating'])
            new_df = new_df.dropna()
            
            if new_df['rating'].dtype.kind in 'iufc':
                new_df['rating'] = new_df['rating'].map(rating_map)
            
            new_df = new_df[new_df['rating'].notna()]
            new_df = new_df[new_df['rating'].isin(alpha_to_num.keys())]
            new_df['rating_num'] = new_df['rating'].map(alpha_to_num)
            
            if st.button("🔄 Mettre à jour les données"):
                st.session_state.df_main = update_data(st.session_state.df_main, new_df)
                df = st.session_state.df_main
                st.success(f"✅ Données mises à jour ! {len(new_df)} obligations traitées")
                st.rerun()
                
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {str(e)}")
    
    # Statistiques globales dans la sidebar
    if not df.empty:
        st.subheader("📈 Vue d'ensemble")
        st.metric("Nombre d'obligations", f"{len(df):,}")
        st.metric("Spread moyen", f"{df['spread'].mean():.0f} bps")
        st.metric("Nombre de secteurs", f"{df['secteur'].nunique()}")
        st.metric("Plage de ratings", f"{df['rating'].nunique()} ratings")

# Vérification des données
if df.empty:
    st.error("⚠️ Aucune donnée disponible. Veuillez vérifier le fichier obligations.xlsx")
    st.stop()

# Préparation des listes pour les filtres
secteurs = sorted(df['secteur'].unique().tolist())
echeances = sorted(df['fourchette_annee'].unique().tolist())
ratings_available = sorted([r for r in rating_map.values() if r in df['rating'].unique()], 
                          key=lambda x: alpha_to_num[x], reverse=True)

# Tabs principales
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Analyse par Secteur", "📊 Dashboard Global", "🔍 Recherche Détaillée", "📋 Base de Données"])

with tab1:
    st.header("🎯 Analyse Sectorielle Approfondie")
    
    # Sélection du secteur
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_sector = st.selectbox("🏭 Choisissez un secteur à analyser", secteurs, key="sector_analysis")
    with col2:
        show_comparison = st.checkbox("📊 Comparer avec la moyenne du marché", value=True)
    
    sector_data = df[df['secteur'] == selected_sector].copy()
    
    if not sector_data.empty:
        # Métriques du secteur
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        sector_mean = sector_data['spread'].mean()
        market_mean = df['spread'].mean()
        sector_count = len(sector_data)
        sector_std = sector_data['spread'].std()
        
        with col_m1:
            delta = f"{sector_mean - market_mean:+.0f} vs marché" if show_comparison else None
            st.metric("🎯 Spread Moyen Secteur", f"{sector_mean:.0f} bps", delta=delta)
        
        with col_m2:
            st.metric("📊 Nombre d'Obligations", f"{sector_count}")
        
        with col_m3:
            st.metric("📏 Volatilité (σ)", f"{sector_std:.0f} bps")
        
        with col_m4:
            percentile_rank = (df['spread'].rank(pct=True) * 100)[df['secteur'] == selected_sector].mean()
            st.metric("📈 Rang Percentile", f"{percentile_rank:.0f}%")
        
        # Graphiques sectoriels
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            # Graphique par rating avec comparaison
            sector_rating_stats = sector_data.groupby('rating')['spread'].agg(['mean', 'count']).reset_index()
            
            if show_comparison:
                market_rating_stats = df.groupby('rating')['spread'].mean().reset_index()
                market_rating_stats.columns = ['rating', 'market_mean']
                sector_rating_stats = sector_rating_stats.merge(market_rating_stats, on='rating', how='left')
                
                fig_rating = go.Figure()
                fig_rating.add_trace(go.Bar(
                    name=f'{selected_sector}',
                    x=sector_rating_stats['rating'],
                    y=sector_rating_stats['mean'],
                    marker_color='#2a5298'
                ))
                fig_rating.add_trace(go.Bar(
                    name='Marché',
                    x=sector_rating_stats['rating'],
                    y=sector_rating_stats['market_mean'],
                    marker_color='#ffa726',
                    opacity=0.7
                ))
                fig_rating.update_layout(
                    title=f"Spreads par Rating - {selected_sector} vs Marché",
                    xaxis_title="Rating",
                    yaxis_title="Spread (bps)",
                    barmode='group',
                    height=400
                )
            else:
                fig_rating = px.bar(
                    sector_rating_stats, 
                    x='rating', 
                    y='mean',
                    title=f"Spreads Moyens par Rating - {selected_sector}",
                    color='mean',
                    color_continuous_scale='RdYlBu_r'
                )
                fig_rating.update_layout(height=400)
            
            st.plotly_chart(fig_rating, use_container_width=True)
        
        with col_g2:
            # Distribution des spreads avec box plot
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Box(
                y=sector_data['spread'],
                name=selected_sector,
                boxpoints='outliers',
                marker_color='#2a5298'
            ))
            
            if show_comparison:
                fig_dist.add_trace(go.Box(
                    y=df['spread'],
                    name='Marché Global',
                    boxpoints=False,
                    marker_color='#ffa726',
                    opacity=0.7
                ))
            
            fig_dist.update_layout(
                title=f"Distribution des Spreads - {selected_sector}",
                yaxis_title="Spread (bps)",
                height=400
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Graphique par échéance
        st.subheader("📅 Analyse par Échéance")
        
        maturity_stats = sector_data.groupby('fourchette_annee')['spread'].agg(['mean', 'std', 'count']).reset_index()
        
        fig_maturity = px.line(
            maturity_stats, 
            x='fourchette_annee', 
            y='mean',
            title=f"Courbe des Spreads par Échéance - {selected_sector}",
            markers=True,
            line_shape='spline'
        )
        
        # Ajouter les barres d'erreur
        fig_maturity.add_trace(go.Scatter(
            x=maturity_stats['fourchette_annee'],
            y=maturity_stats['mean'] + maturity_stats['std'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig_maturity.add_trace(go.Scatter(
            x=maturity_stats['fourchette_annee'],
            y=maturity_stats['mean'] - maturity_stats['std'],
            mode='lines',
            fill='tonexty',
            line=dict(width=0),
            name='±1σ',
            fillcolor='rgba(42, 82, 152, 0.2)'
        ))
        
        fig_maturity.update_layout(height=400)
        st.plotly_chart(fig_maturity, use_container_width=True)
        
        # Table des obligations du secteur
        st.subheader(f"💼 Obligations du secteur {selected_sector}")
        
        display_sector = sector_data[['nom_obligation', 'fourchette_annee', 'rating', 'spread']].copy()
        display_sector = display_sector.sort_values('spread', ascending=False)
        
        st.dataframe(
            display_sector,
            column_config={
                "nom_obligation": "Nom de l'Obligation",
                "fourchette_annee": "Échéance",
                "rating": "Rating",
                "spread": st.column_config.NumberColumn("Spread (bps)", format="%.1f")
            },
            hide_index=True,
            use_container_width=True
        )

with tab2:
    st.header("📊 Dashboard Global")
    
    # Filtres globaux
    st.subheader("🎛️ Filtres Globaux")
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    with col_f1:
        selected_sectors = st.multiselect("🏭 Secteurs", secteurs, default=secteurs[:5])
    with col_f2:
        selected_ratings = st.multiselect("⭐ Ratings", ratings_available, default=ratings_available[:6])
    with col_f3:
        selected_maturities = st.multiselect("📅 Échéances", echeances, default=echeances)
    with col_f4:
        spread_range = st.slider(
            "💰 Fourchette Spread (bps)",
            min_value=int(df['spread'].min()),
            max_value=int(df['spread'].max()),
            value=(int(df['spread'].min()), int(df['spread'].max()))
        )
    
    # Application des filtres
    df_filtered = df[
        df['secteur'].isin(selected_sectors) &
        df['rating'].isin(selected_ratings) &
        df['fourchette_annee'].isin(selected_maturities) &
        (df['spread'] >= spread_range[0]) &
        (df['spread'] <= spread_range[1])
    ]
    
    if not df_filtered.empty:
        # Métriques filtrées
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("📊 Obligations", f"{len(df_filtered):,}")
        col_m2.metric("📈 Spread Moyen", f"{df_filtered['spread'].mean():.0f} bps")
        col_m3.metric("📉 Spread Médian", f"{df_filtered['spread'].median():.0f} bps")
        col_m4.metric("📏 Écart-type", f"{df_filtered['spread'].std():.0f} bps")
        
        # Graphiques principaux
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            # Heatmap Rating vs Échéance
            pivot_data = df_filtered.pivot_table(
                index='rating',
                columns='fourchette_annee',
                values='spread',
                aggfunc='mean'
            )
            
            # Trier les ratings
            rating_order = sorted([r for r in pivot_data.index if r in alpha_to_num], 
                                key=lambda x: alpha_to_num[x], reverse=True)
            pivot_data = pivot_data.reindex(rating_order)
            
            fig_heatmap = px.imshow(
                pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                title="Heatmap des Spreads : Rating × Échéance",
                color_continuous_scale='RdYlBu_r',
                aspect='auto'
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with col_g2:
            # Graphique 3D amélioré
            if len(df_filtered) > 0:
                grp_3d = df_filtered.groupby(['secteur', 'fourchette_annee', 'rating_num'], as_index=False)['spread'].mean()
                grp_3d = grp_3d.dropna()
                grp_3d = grp_3d[grp_3d['spread'] > 0]
                
                if len(grp_3d) > 0:
                    grp_3d['maturity_code'] = pd.Categorical(grp_3d['fourchette_annee']).codes
                    
                    fig_3d = px.scatter_3d(
                        grp_3d,
                        x='rating_num',
                        y='maturity_code',
                        z='spread',
                        color='secteur',
                        size='spread',
                        title="Vue 3D : Rating × Échéance × Spread",
                        labels={
                            'rating_num': 'Rating (numérique)',
                            'maturity_code': 'Échéance (code)',
                            'spread': 'Spread (bps)'
                        },
                        height=400
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.info("Pas assez de données pour la vue 3D")
        
        # Analyse comparative par secteur
        st.subheader("🏭 Comparaison Inter-Sectorielle")
        
        sector_comparison = df_filtered.groupby('secteur')['spread'].agg(['mean', 'std', 'count']).reset_index()
        sector_comparison = sector_comparison.sort_values('mean', ascending=True)
        
        fig_sectors = px.bar(
            sector_comparison,
            x='mean',
            y='secteur',
            orientation='h',
            title="Spread Moyen par Secteur",
            color='mean',
            color_continuous_scale='RdYlBu_r'
        )
        fig_sectors.update_layout(height=400)
        st.plotly_chart(fig_sectors, use_container_width=True)
        
    else:
        st.warning("⚠️ Aucune donnée ne correspond aux filtres sélectionnés")

with tab3:
    st.header("🔍 Recherche Détaillée")
    
    # Modes de recherche
    search_mode = st.radio(
        "Mode de recherche :",
        ["🎯 Recherche par Nom", "🔍 Recherche Multi-Critères", "📊 Comparateur"],
        horizontal=True
    )
    
    if search_mode == "🎯 Recherche par Nom":
        st.subheader("Recherche par nom d'obligation")
        
        search_term = st.text_input("🔍 Tapez le nom ou une partie du nom de l'obligation :")
        
        if search_term:
            results = df[df['nom_obligation'].str.contains(search_term, case=False, na=False)]
            
            if not results.empty:
                st.success(f"✅ {len(results)} obligation(s) trouvée(s)")
                
                for idx, row in results.iterrows():
                    with st.expander(f"📋 {row['nom_obligation']}", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("🏭 Secteur", row['secteur'])
                        col2.metric("📅 Échéance", row['fourchette_annee'])
                        col3.metric("⭐ Rating", row['rating'])
                        col4.metric("💰 Spread", f"{row['spread']:.1f} bps")
                        
                        # Comparaison avec les pairs
                        peers = df[(df['secteur'] == row['secteur']) & 
                                  (df['rating'] == row['rating']) & 
                                  (df['nom_obligation'] != row['nom_obligation'])]
                        
                        if not peers.empty:
                            peer_mean = peers['spread'].mean()
                            deviation = row['spread'] - peer_mean
                            st.info(f"📊 Comparaison avec les pairs du secteur {row['secteur']} en rating {row['rating']} : "
                                   f"Moyenne = {peer_mean:.1f} bps | Écart = {deviation:+.1f} bps")
            else:
                st.warning("❌ Aucune obligation trouvée")
    
    elif search_mode == "🔍 Recherche Multi-Critères":
        st.subheader("Recherche avancée par critères")
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            criteria_sectors = st.multiselect("🏭 Secteurs", secteurs)
        with col_s2:
            criteria_ratings = st.multiselect("⭐ Ratings", ratings_available)
        with col_s3:
            criteria_maturities = st.multiselect("📅 Échéances", echeances)
        
        criteria_spread = st.slider(
            "💰 Fourchette de spread recherchée",
            min_value=int(df['spread'].min()),
            max_value=int(df['spread'].max()),
            value=(int(df['spread'].quantile(0.25)), int(df['spread'].quantile(0.75)))
        )
        
        if st.button("🔍 Lancer la recherche"):
            filtered_results = df.copy()
            
            if criteria_sectors:
                filtered_results = filtered_results[filtered_results['secteur'].isin(criteria_sectors)]
            if criteria_ratings:
                filtered_results = filtered_results[filtered_results['rating'].isin(criteria_ratings)]
            if criteria_maturities:
                filtered_results = filtered_results[filtered_results['fourchette_annee'].isin(criteria_maturities)]
            
            filtered_results = filtered_results[
                (filtered_results['spread'] >= criteria_spread[0]) &
                (filtered_results['spread'] <= criteria_spread[1])
            ]
            
            if not filtered_results.empty:
                st.success(f"✅ {len(filtered_results)} obligation(s) correspondent aux critères")
                
                # Statistiques des résultats
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("📈 Spread moyen", f"{filtered_results['spread'].mean():.1f} bps")
                col_r2.metric("📉 Spread médian", f"{filtered_results['spread'].median():.1f} bps")
                col_r3.metric("📏 Écart-type", f"{filtered_results['spread'].std():.1f} bps")
                
                # Affichage des résultats
                st.dataframe(
                    filtered_results[['nom_obligation', 'secteur', 'fourchette_annee', 'rating', 'spread']],
                    column_config={
                        "nom_obligation": "Nom de l'Obligation",
                        "secteur": "Secteur",
                        "fourchette_annee": "Échéance",
                        "rating": "Rating",
                        "spread": st.column_config.NumberColumn("Spread (bps)", format="%.1f")
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.warning("❌ Aucune obligation ne correspond aux critères")
    
    else:  # Comparateur
        st.subheader("📊 Comparateur d'Obligations")
        
        st.markdown("**Sélectionnez jusqu'à 5 obligations à comparer :**")
        
        bond_names = df['nom_obligation'].tolist()
        selected_bonds = st.multiselect(
            "💼 Choisissez les obligations",
            bond_names,
            max_selections=5
        )
        
        if selected_bonds:
            comparison_data = df[df['nom_obligation'].isin(selected_bonds)]
            
            # Tableau comparatif
            st.dataframe(
                comparison_data[['nom_obligation', 'secteur', 'fourchette_annee', 'rating', 'spread']],
                column_config={
                    "nom_obligation": "Obligation",
                    "secteur": "Secteur",
                    "fourchette_annee": "Échéance",
                    "rating": "Rating",
                    "spread": st.column_config.NumberColumn("Spread (bps)", format="%.1f")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Graphique comparatif
            if len(selected_bonds) > 1:
                fig_comp = px.bar(
                    comparison_data,
                    x='nom_obligation',
                    y='spread',
                    color='secteur',
                    title="Comparaison des Spreads",
                    text='spread'
                )
                fig_comp.update_traces(texttemplate='%{text:.1f} bps', textposition='outside')
                fig_comp.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_comp, use_container_width=True)

with tab4:
    st.header("📋 Base de Données Complète")
    
    # Outils de gestion
    col_t1, col_t2, col_t3 = st.columns(3)
    
    with col_t1:
        sort_column = st.selectbox(
            "📊 Trier par",
            ['spread', 'nom_obligation', 'secteur', 'rating', 'fourchette_annee']
        )
    
    with col_t2:
        sort_ascending = st.radio("Ordre", ["Croissant", "Décroissant"], horizontal=True) == "Croissant"
    
    with col_t3:
        items_per_page = st.selectbox("Éléments par page", [25, 50, 100, 200])
    
    # Filtres de la base de données
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        db_sectors = st.multiselect("🏭 Filtrer par secteurs", secteurs, key="db_sectors")
    with col_f2:
        db_ratings = st.multiselect("⭐ Filtrer par ratings", ratings_available, key="db_ratings")
    with col_f3:
        db_maturities = st.multiselect("📅 Filtrer par échéances", echeances, key="db_maturities")
    
    # Application des filtres
    db_filtered = df.copy()
    if db_sectors:
        db_filtered = db_filtered[db_filtered['secteur'].isin(db_sectors)]
    if db_ratings:
        db_filtered = db_filtered[db_filtered['rating'].isin(db_ratings)]
    if db_maturities:
        db_filtered = db_filtered[db_filtered['fourchette_annee'].isin(db_maturities)]
    
    # Tri
    db_filtered = db_filtered.sort_values(sort_column, ascending=sort_ascending)
    
    # Pagination
    total_items = len(db_filtered)
    total_pages = (total_items - 1) // items_per_page + 1
    
    if total_pages > 1:
        page = st.selectbox(f"Page (Total: {total_pages})", list(range(1, total_pages + 1)))
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        db_display = db_filtered.iloc[start_idx:end_idx]
        st.info(f"Affichage des éléments {start_idx + 1} à {end_idx} sur {total_items}")
    else:
        db_display = db_filtered
        st.info(f"Affichage de {total_items} obligations")
    
    # Affichage de la base de données
    st.dataframe(
        db_display[['nom_obligation', 'secteur', 'fourchette_annee', 'rating', 'spread']],
        column_config={
            "nom_obligation": st.column_config.TextColumn("Nom de l'Obligation", width="large"),
            "secteur": st.column_config.TextColumn("Secteur", width="medium"),
            "fourchette_annee": st.column_config.TextColumn("Échéance", width="small"),
            "rating": st.column_config.TextColumn("Rating", width="small"),
            "spread": st.column_config.NumberColumn("Spread (bps)", format="%.1f", width="small")
        },
        hide_index=True,
        use_container_width=True,
        height=600
    )
    
    # Export des données
    st.subheader("📤 Export des Données")
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        # Export CSV
        csv_data = db_filtered.to_csv(index=False)
        st.download_button(
            label="💾 Télécharger en CSV",
            data=csv_data,
            file_name=f"bond_spreads_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col_e2:
        # Export Excel
        try:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                db_filtered.to_excel(writer, sheet_name='Bond_Spreads', index=False)
            
            st.download_button(
                label="📊 Télécharger en Excel",
                data=buffer.getvalue(),
                file_name=f"bond_spreads_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except ImportError:
            st.warning("⚠️ Export Excel indisponible. Utilisez l'export CSV.")
            st.info("💡 Pour activer l'export Excel, installez : `pip install openpyxl`")

# Footer avec informations système
st.markdown("---")
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.info(f"📊 **{len(df):,}** obligations dans la base")

with col_info2:
    st.info(f"🏭 **{df['secteur'].nunique()}** secteurs couverts")

with col_info3:
    last_update = pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')
    st.info(f"🕐 Dernière mise à jour : {last_update}")

# Notifications et alertes
if not df.empty:
    # Détection d'outliers
    Q1 = df['spread'].quantile(0.25)
    Q3 = df['spread'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['spread'] < lower_bound) | (df['spread'] > upper_bound)]
    
    if not outliers.empty and len(outliers) <= 10:
        with st.expander(f"⚠️ Alertes : {len(outliers)} obligation(s) avec des spreads atypiques"):
            for _, outlier in outliers.iterrows():
                if outlier['spread'] > upper_bound:
                    st.error(f"🔴 **{outlier['nom_obligation']}** - Spread élevé : {outlier['spread']:.1f} bps "
                           f"({outlier['secteur']}, {outlier['rating']})")
                else:
                    st.success(f"🟢 **{outlier['nom_obligation']}** - Spread très bas : {outlier['spread']:.1f} bps "
                            f"({outlier['secteur']}, {outlier['rating']})")

# CSS supplémentaire pour les métriques
st.markdown("""
<style>
    [data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
    }
    
    [data-testid="metric-container"] > div {
        width: fit-content;
    }
    
    [data-testid="metric-container"] > div > div {
        width: fit-content;
    }
    
    .stExpander {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)
