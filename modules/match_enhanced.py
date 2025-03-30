# match_enhancement.py
import pandas as pd


def enhance_match_data(df_matches, df_stars, df_rankings=None):
    # Étape 1 : Obtenir les top stars par date
    df_top_stars = df_stars[df_stars['is_star']].copy()
    df_top_stars['star_rank'] = df_top_stars.groupby('tourney_date')['cumulative_points'].rank(ascending=False,
                                                                                               method='first')
    df_top_stars = df_top_stars[df_top_stars['star_rank'] <= 5]
    df_top_stars = df_top_stars.sort_values(['tourney_date', 'star_rank'])

    # Étape 2 : Obtenir les participants de chaque tournoi
    participants = df_matches.groupby('tourney_id').apply(
        lambda x: set(x['winner_id'].unique()).union(set(x['loser_id'].unique()))
    ).reset_index()
    participants.columns = ['tourney_id', 'participants']

    # Ajouter 'tourney_date' aux participants
    tourney_dates = df_matches[['tourney_id', 'tourney_date']].drop_duplicates()
    participants = participants.merge(tourney_dates, on='tourney_id', how='left')

    # Étape 3 : Associer les stars aux tournois
    merged = participants.merge(df_top_stars, on='tourney_date', how='left')
    # Supprimer les lignes où 'player_id' est NaN (pas de stars à cette date)
    merged = merged.dropna(subset=['player_id'])
    # Vérifier si la star est présente dans le tournoi
    merged['star_present'] = merged.apply(lambda row: row['player_id'] in row['participants'], axis=1)
    # Conserver uniquement les stars présentes
    merged = merged[merged['star_present']]
    # Obtenir 'tourney_id', 'star_rank', 'player_name' des stars présentes
    tournament_stars = merged[['tourney_id', 'star_rank', 'player_name']].copy()

    # Créer les colonnes 'star_top_1' à 'star_top_5'
    tournament_stars_pivot = tournament_stars.pivot_table(index='tourney_id', columns='star_rank', values='player_name',
                                                          aggfunc='first')
    tournament_stars_pivot = tournament_stars_pivot.rename(columns=lambda x: f'star_top_{int(x)}')
    tournament_stars_pivot = tournament_stars_pivot.reset_index()

    # Compter le nombre de stars présentes dans chaque tournoi
    star_counts = tournament_stars.groupby('tourney_id').size().reset_index(name='num_stars_in_tournament')

    # Fusionner avec df_matches
    df_matches = df_matches.merge(tournament_stars_pivot, on='tourney_id', how='left')
    df_matches = df_matches.merge(star_counts, on='tourney_id', how='left')

    # Remplacer les NaN dans 'num_stars_in_tournament' par 0
    df_matches['num_stars_in_tournament'] = df_matches['num_stars_in_tournament'].fillna(0).astype(int)

    # Remplir les colonnes 'star_top_n' avec None si NaN
    for i in range(1, 6):
        star_col = f'star_top_{i}'
        if star_col not in df_matches.columns:
            df_matches[star_col] = None
        else:
            df_matches[star_col] = df_matches[star_col].fillna('')

    # Vérifier si le gagnant a un classement plus élevé que le perdant
    # Si les classements manquent, essayer de les obtenir à partir de df_rankings à la même date
    if df_rankings is not None:
        # S'assurer que les dates sont au format datetime
        df_matches['tourney_date'] = pd.to_datetime(df_matches['tourney_date'])
        df_rankings['ranking_date'] = pd.to_datetime(df_rankings['ranking_date'])

        # Fusionner les classements du gagnant
        df_matches = df_matches.merge(
            df_rankings[['ranking_date', 'player', 'rank']].rename(
                columns={'player': 'winner_id', 'rank': 'winner_rank_from_rankings'}),
            left_on=['tourney_date', 'winner_id'], right_on=['ranking_date', 'winner_id'], how='left'
        )
        # Fusionner les classements du perdant
        df_matches = df_matches.merge(
            df_rankings[['ranking_date', 'player', 'rank']].rename(
                columns={'player': 'loser_id', 'rank': 'loser_rank_from_rankings'}),
            left_on=['tourney_date', 'loser_id'], right_on=['ranking_date', 'loser_id'], how='left'
        )
        # Remplir les classements manquants
        df_matches['winner_rank'] = df_matches['winner_rank'].fillna(df_matches['winner_rank_from_rankings'])
        df_matches['loser_rank'] = df_matches['loser_rank'].fillna(df_matches['loser_rank_from_rankings'])
        # Supprimer les colonnes supplémentaires
        df_matches = df_matches.drop(
            columns=['winner_rank_from_rankings', 'loser_rank_from_rankings', 'ranking_date_x', 'ranking_date_y'])

    df_matches['winner_lower_rank'] = df_matches.apply(
        lambda row: (row['winner_rank'] > row['loser_rank']) if pd.notnull(row['winner_rank']) and pd.notnull(
            row['loser_rank']) else None,
        axis=1
    )

    return df_matches


def ensure_type_conversion(df, cat_codes=False):
    """
    Convertit les colonnes en types optimisés : 'category' pour les colonnes catégorielles,
    et 'int' pour les binaires compatibles.
    """
    for col in df.columns:
        # Si la colonne est catégorielle ou a un petit nombre de valeurs uniques
        if df[col].dtype == 'object' or df[col].dtype.name == 'category' :  # or df[col].nunique() <= 10
            # Convertir en 'category' pour optimiser la mémoire et simplifier les transformations
            df[col] = df[col].astype('category')
            # Si cat_codes est False, on code les catégories en numéros
            if not cat_codes:
                df[col] = df[col].cat.codes

        # Si la colonne est binaire (exactement 2 valeurs uniques)
        elif df[col].nunique() == 2:
            unique_vals = df[col].dropna().unique()
            # Convertir en 0/1 si les valeurs sont déjà sous forme binaire
            if set(unique_vals) == {0, 1} or set(unique_vals) == {True, False}:
                df[col] = df[col].astype(int)
            else:
                # Si non binaire, on transforme en catégorie et on utilise les codes
                df[col] = df[col].astype('category').cat.codes

    return df
