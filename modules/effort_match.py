#effort measurement
import pandas as pd
import re

def enhance_match_data(df_matches, df_stars):
    """
    Enrichit les données de matchs avec les informations sur les stars.
    """
    # Mapper le nombre de stars présentes dans chaque tournoi
    star_counts = df_stars.set_index('tourney_date')['number_of_stars']
    df_matches['num_stars'] = df_matches['tourney_date'].map(star_counts)

    # Ajouter les noms des stars dans les colonnes correspondantes
    for rank in range(1, 6):
        star_names = df_stars.set_index('tourney_date')[f'top_{rank}_star']
        df_matches[f'star_{rank}_name'] = df_matches['tourney_date'].map(star_names)

    # Indiquer si les stars sont présentes dans le tournoi
    df_matches['stars_present'] = df_matches['num_stars'] > 0

    # Comparer les classements des joueurs
    df_matches['winner_lower_rank'] = df_matches['winner_rank'] > df_matches['loser_rank']

    # Calculer le nombre de sets gagnés et déterminer si la victoire est dominante
    def parse_score(score):
        sets_won = 0
        dominant_set_count = 0
        sets = re.findall(r'(\d+)-(\d+)(?:\(\d+\))?', str(score))

        for winner_set_score, loser_set_score in sets:
            winner_set_score = int(winner_set_score)
            loser_set_score = int(loser_set_score)

            if winner_set_score > loser_set_score:
                sets_won += 1
                if winner_set_score >= 6 and loser_set_score <= 2:
                    dominant_set_count += 1

        dominant_victory = dominant_set_count >= 2
        return sets_won, dominant_victory

    df_matches[['sets_won', 'victory_dominant']] = df_matches['score'].apply(lambda x: pd.Series(parse_score(x)))

    # Réinitialiser l'index
    df_matches.reset_index(drop=True, inplace=True)

    return df_matches
