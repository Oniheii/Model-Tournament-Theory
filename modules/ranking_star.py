# ranking_star/star_atp_table.py
import pandas as pd

# Système de points pour les tournois de type G (Grand Chelem) et M (Masters 1000)
points_system = {
    'G': {'W': 2000, 'F': 1200, 'SF': 720, 'QF': 360, 'R16': 180, 'R32': 90, 'R64': 45, 'R128': 10},
    'M': {'W': 1000, 'F': 600, 'SF': 360, 'QF': 180, 'R16': 90, 'R32': 45, 'R64': 25, 'R128': 10}
}


# Fonction pour obtenir les points selon le round atteint
def get_round_points(tourney_level, round_name):
    return points_system.get(tourney_level, {}).get(round_name, 0)


def build_empty_table(df_matches):
    """
    Builds a table with all tournament dates and names for each player who participated in them.
    """
    # Convert 'tourney_date' to datetime format
    df_matches['tourney_date'] = pd.to_datetime(df_matches['tourney_date'], format='%d/%m/%Y')
    # Prepare player participation data
    # For winners
    df_winners = df_matches[['tourney_date', 'tourney_name', 'winner_id', 'winner_name']].rename(
        columns={'winner_id': 'player_id', 'winner_name': 'player_name'}
    )
    # For losers
    df_losers = df_matches[['tourney_date', 'tourney_name', 'loser_id', 'loser_name']].rename(
        columns={'loser_id': 'player_id', 'loser_name': 'player_name'}
    )
    # Combine winners and losers
    df_players = pd.concat([df_winners, df_losers], ignore_index=True)
    # Remove duplicates to get unique player-tournament combinations
    df_players = df_players.drop_duplicates(subset=['tourney_date', 'tourney_name', 'player_id'])
    # Ensure 'player_name' is not null
    df_players = df_players.dropna(subset=['player_name'])

    return df_players


def calculate_points_for_empty_table(df_empty, df_matches):
    """
    Calcule les points obtenus par chaque joueur à chaque date de tournoi.
    Si un joueur n'a pas participé à un tournoi, ses points sont zéro.
    """
    # Préparer les données des matchs avec 'player_id', 'tourney_date', 'tourney_level', 'round_achieved', et 'points'
    df_matches_winners = df_matches[
        ['tourney_id', 'tourney_date', 'tourney_name', 'tourney_level', 'winner_id', 'winner_name', 'round']].copy()
    df_matches_winners['player_id'] = df_matches_winners['winner_id']
    df_matches_winners['player_name'] = df_matches_winners['winner_name']
    df_matches_winners['round_achieved'] = df_matches_winners['round']
    df_matches_winners['result'] = 'win'

    df_matches_losers = df_matches[
        ['tourney_id', 'tourney_date', 'tourney_name', 'tourney_level', 'loser_id', 'loser_name', 'round']].copy()
    df_matches_losers['player_id'] = df_matches_losers['loser_id']
    df_matches_losers['player_name'] = df_matches_losers['loser_name']
    df_matches_losers['round_achieved'] = df_matches_losers['round']
    df_matches_losers['result'] = 'lose'

    df_players = pd.concat([df_matches_winners, df_matches_losers], ignore_index=True)
    df_players['points'] = df_players.apply(
        lambda row: get_round_points(row['tourney_level'], row['round_achieved']),
        axis=1
    )

    # Garder uniquement les colonnes nécessaires
    df_players = df_players[['tourney_date', 'tourney_name', 'player_id', 'player_name', 'points']]

    # Fusionner 'df_empty' avec 'df_players' pour obtenir les points
    df_empty = df_empty.merge(df_players, on=['tourney_date', 'tourney_name', 'player_id', 'player_name'], how='left')

    # Remplir les points manquants par zéro
    df_empty['points'] = df_empty['points'].fillna(0)

    # Trier la table résultante par 'player_id' et 'tourney_date'
    df_empty = df_empty.sort_values(by=['player_id', 'tourney_date'])

    return df_empty


def calculate_cumulative_points(df, df_matches):
    """
    Calcule le cumul de points pour chaque joueur sur les 13 derniers tournois.
    """

    # Assign a unique number to each tournament, sorted by date
    tournament_dates = df[['tourney_date']].drop_duplicates().sort_values('tourney_date')
    tournament_dates = tournament_dates.reset_index(drop=True)
    tournament_dates['tournament_number'] = tournament_dates.index + 1  # Start at 1

    # Merge tournament numbers with 'df'
    df = df.merge(tournament_dates, on='tourney_date', how='left')

    # Prepare a pivot table of player points by tournament_number
    df_pivot = df.pivot_table(index='player_id', columns='tournament_number',
                              values='points', fill_value=0)

    # Calculate the rolling cumulative sum of points over the last 13 tournaments
    cumulative_points_list = []
    for tn in tournament_dates['tournament_number']:
        last_13 = [n for n in range(tn - 12, tn + 1) if n > 0]
        cumulative_points = df_pivot[last_13].sum(axis=1)
        df_cp = pd.DataFrame({
            'player_id': cumulative_points.index,
            'tournament_number': tn,
            'cumulative_points': cumulative_points.values
        })
        cumulative_points_list.append(df_cp)

    df_cumulative = pd.concat(cumulative_points_list, ignore_index=True)

    # Merge cumulative points with 'df'
    df = df.merge(df_cumulative, on=['player_id', 'tournament_number'], how='left')

    # Calculate total points available for each tournament_number
    total_points_period = df_cumulative.groupby('tournament_number')['cumulative_points'].sum().reset_index()
    total_points_period = total_points_period.rename(columns={'cumulative_points': 'total_points_period'})

    # Merge total points with 'df'
    df = df.merge(total_points_period, on='tournament_number', how='left')

    # Calculate the share of points earned by each player
    df['part_points_earned'] = (df['cumulative_points'] / df['total_points_period']) * 100

    # Assign rank to each player for each tournament_number, inversely proportional to cumulative_points
    df['rank'] = df.groupby('tournament_number')['cumulative_points'].rank(ascending=False, method='min')

    # Méthode 1 : Moyenne + écart-type
    stats = df.groupby('tournament_number')['cumulative_points'].agg(['mean', 'std']).reset_index()
    stats = stats.rename(columns={'mean': 'mean_cp', 'std': 'std_cp'})
    df = df.merge(stats, on='tournament_number', how='left')
    df['star_method_1'] = df['cumulative_points'] > (df['mean_cp'] + df['std_cp'])

    # Méthode 2 : Valeurs aberrantes (Q3 + 1.5 * IQR)
    quartiles = df.groupby('tournament_number')['cumulative_points'].quantile([0.25, 0.75]).unstack().reset_index()
    quartiles = quartiles.rename(columns={0.25: 'Q1', 0.75: 'Q3'})
    quartiles['IQR'] = quartiles['Q3'] - quartiles['Q1']
    quartiles['outlier_threshold'] = quartiles['Q3'] + 1.5 * quartiles['IQR']
    df = df.merge(quartiles[['tournament_number', 'outlier_threshold']], on='tournament_number', how='left')
    df['star_method_2'] = df['cumulative_points'] > df['outlier_threshold']

    # Méthode 3 : 99e percentile de la distribution des parts de points
    percentiles = df.groupby('tournament_number')['part_points_earned'].quantile(0.99).reset_index()
    percentiles = percentiles.rename(columns={'part_points_earned': 'percentile_99'})
    df = df.merge(percentiles, on='tournament_number', how='left')
    df['star_method_3'] = df['part_points_earned'] > df['percentile_99']

    # Flag the player as a star if they satisfy at least two methods
    df['is_star'] = df[['star_method_1', 'star_method_2', 'star_method_3']].sum(axis=1) >= 2

    # Add quartile classification within each tournament
    df['quartile'] = df.groupby(['tourney_date', 'tourney_name'])[
        'cumulative_points'].transform(
        lambda x: pd.qcut(x.rank(method='first', ascending=False), 4,
                          labels=['75-100%', '50-75%', '25-50%', '0-25%']))

    # Obtenir le round maximum atteint par chaque joueur à chaque date T
    # Map rounds to numerical values for ordering
    round_mapping = {
        'R128': 1,
        'R64': 2,
        'R32': 3,
        'R16': 4,
        'QF': 5,
        'SF': 6,
        'F': 7,
        'W': 8
    }

    # Obtenir les rounds maximums atteints par les perdants
    df_loser_rounds = df_matches.copy()
    df_loser_rounds['round_numeric'] = df_loser_rounds['round'].map(round_mapping)
    df_loser_rounds = df_loser_rounds.dropna(subset=['round_numeric'])
    df_loser_rounds = df_loser_rounds.groupby(['loser_id', 'tourney_date'])['round_numeric'].max().reset_index()
    df_loser_rounds = df_loser_rounds.rename(columns={'loser_id': 'player_id', 'round_numeric': 'max_round_numeric'})

    # Obtenir les gagnants de la finale
    df_winner_finals = df_matches[df_matches['round'] == 'F'][['tourney_date', 'winner_id']]
    df_winner_finals['max_round_numeric'] = 8  # 8 pour 'W' (Winner)
    df_winner_finals = df_winner_finals.rename(columns={'winner_id': 'player_id'})

    # Combiner les rounds des perdants et des gagnants
    df_max_rounds = pd.concat([df_loser_rounds, df_winner_finals], ignore_index=True)

    # Map numeric round back to round names, en ajoutant 'W' pour 8
    inverse_round_mapping = {v: k for k, v in round_mapping.items()}
    inverse_round_mapping[8] = 'W'
    df_max_rounds['max_round'] = df_max_rounds['max_round_numeric'].map(inverse_round_mapping)

    # Fusionner 'max_round' avec 'df'
    df = df.merge(df_max_rounds[['player_id', 'tourney_date', 'max_round']], on=['player_id', 'tourney_date'],
                  how='left')

    # Trier la table par joueur, date, numéro de tournoi
    df = df.sort_values(by=['player_id', 'tourney_date', 'tournament_number'])

    return df


def generate_star_atp(df):
    """
    Generates a table with the top stars per date, ensuring that a star appears only once in the top positions.
    """
    # Filter players who are stars
    star_atp = df[df['is_star']].copy()

    # Remove duplicate players per date
    star_atp = star_atp.drop_duplicates(subset=['tourney_date', 'tourney_name', 'player_id'])

    # Rank stars by cumulative points for each date
    star_atp['rank'] = star_atp.groupby('tourney_date')['cumulative_points'].rank(
        ascending=False, method='first'
    )

    # Keep top 5 stars per date
    star_atp_filtered = star_atp[star_atp['rank'] <= 5]

    # Remove duplicate names within the same date
    star_atp_filtered = star_atp_filtered.sort_values(['tourney_date', 'rank']).drop_duplicates(
        subset=['tourney_date', 'player_name'], keep='first'
    )

    # Now, for each date, we need to get the top stars
    def get_top_stars(group):
        top_stars = group.sort_values('rank')['player_name'].unique().tolist()
        # Ensure that we have at most 5 unique stars
        top_stars = top_stars[:5]
        return pd.Series({
            'number_of_stars': len(top_stars),
            'top_1_star': top_stars[0] if len(top_stars) > 0 else None,
            'top_2_star': top_stars[1] if len(top_stars) > 1 else None,
            'top_3_star': top_stars[2] if len(top_stars) > 2 else None,
            'top_4_star': top_stars[3] if len(top_stars) > 3 else None,
            'top_5_star': top_stars[4] if len(top_stars) > 4 else None
        })

    star_summary = star_atp_filtered.groupby('tourney_date').apply(get_top_stars).reset_index()

    return star_summary


def generate_yearly_top_stars(df_stars):
    """
    Génère une table avec la liste des 3 premières stars par année avec leurs noms.
    """
    # Garder les données à partir de 1990
    df_stars = df_stars[df_stars['tourney_date'].dt.year >= 1990].copy()

    # Ajouter une colonne 'year'
    df_stars.loc[:, 'year'] = df_stars['tourney_date'].dt.year

    # Obtenir les points cumulés maximum par joueur et par année
    df_yearly = df_stars.groupby(['year', 'player_id', 'player_name']
                                 )['cumulative_points'].max().reset_index()

    # Obtenir les top 3 joueurs par année
    df_top3 = df_yearly.groupby('year').apply(
        lambda x: x.sort_values('cumulative_points', ascending=False).head(3)
    ).reset_index(drop=True)

    # Ajouter une colonne 'rank' pour le classement
    df_top3['rank'] = df_top3.groupby('year')['cumulative_points'].rank(ascending=False, method='first')

    # Transformer les données pour avoir les noms des stars par année
    df_top3_pivot = df_top3.pivot(index='year', columns='rank', values='player_name')
    df_top3_pivot.columns = ['star_1', 'star_2', 'star_3']
    df_top3_pivot = df_top3_pivot.reset_index()

    return df_top3_pivot
