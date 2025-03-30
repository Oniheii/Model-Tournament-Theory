#player level transformation
import os
import re
import pandas as pd
import numpy as np


# Output directory for the transformed tables
output_dir = 'data_ind/'

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


def transform_match_to_player(df_matches):
    """
    Transforms a match-level DataFrame into a player-level DataFrame.
    """
    # Columns common to both winner and loser
    common_cols = ['tourney_id', 'tourney_name', 'surface', 'draw_size', 'tourney_level',
                   'tourney_date', 'match_num', 'score', 'best_of', 'round', 'year',
                   'num_stars_in_tournament', 'Rafael_Nadal_star', 'Roger_Federer_star',
                   'Novak_Djokovic_star', 'Others_star', 'draw_size_cat', 'victory_dominant',
                   'sets_won_low']

    # Winner columns
    winner_cols = ['winner_id', 'winner_name', 'winner_hand', 'winner_ht', 'winner_ioc',
                   'winner_age', 'winner_rank', 'winner_rank_points', 'winner_lower_rank',
                   'height_diff', 'age_diff', 'winner_ioc_grouped', 'winner_age_cat',
                   'winner_ht_cat', 'winner_sets_won', 'loser_sets_won']

    winner_df = df_matches[common_cols + winner_cols].copy()
    # Rename columns
    winner_df = winner_df.rename(columns={
        'winner_id': 'player_id',
        'winner_name': 'player_name',
        'winner_hand': 'player_hand',
        'winner_ht': 'player_ht',
        'winner_ioc': 'player_ioc',
        'winner_age': 'player_age',
        'winner_rank': 'player_rank',
        'winner_rank_points': 'player_rank_points',
        'winner_lower_rank': 'player_lower_rank',
        'winner_ioc_grouped': 'player_ioc_grouped',
        'winner_age_cat': 'player_age_cat',
        'winner_ht_cat': 'player_ht_cat',
        'winner_sets_won': 'player_sets_won',
        'loser_sets_won': 'opponent_sets_won',
        'height_diff': 'height_diff',
        'age_diff': 'age_diff'
    })
    winner_df['player_victory'] = 1  # Indicate the player won the match

    # Loser columns
    loser_cols = ['loser_id', 'loser_name', 'loser_hand', 'loser_ht', 'loser_ioc',
                  'loser_age', 'loser_rank', 'loser_rank_points', 'height_diff', 'age_diff',
                  'loser_ioc_grouped', 'loser_age_cat', 'loser_ht_cat', 'winner_sets_won',
                  'loser_sets_won']

    loser_df = df_matches[common_cols + loser_cols].copy()
    # Rename columns
    loser_df = loser_df.rename(columns={
        'loser_id': 'player_id',
        'loser_name': 'player_name',
        'loser_hand': 'player_hand',
        'loser_ht': 'player_ht',
        'loser_ioc': 'player_ioc',
        'loser_age': 'player_age',
        'loser_rank': 'player_rank',
        'loser_rank_points': 'player_rank_points',
        'loser_ioc_grouped': 'player_ioc_grouped',
        'loser_age_cat': 'player_age_cat',
        'loser_ht_cat': 'player_ht_cat',
        'winner_sets_won': 'opponent_sets_won',
        'loser_sets_won': 'player_sets_won',
        'height_diff': 'height_diff',
        'age_diff': 'age_diff'
    })
    # Reverse the signs of 'height_diff' and 'age_diff' for losers
    loser_df['height_diff'] = -loser_df['height_diff']
    loser_df['age_diff'] = -loser_df['age_diff']
    # For losers, 'player_lower_rank' is the opposite
    loser_df['player_lower_rank'] = (~df_matches['winner_lower_rank']).astype(int)
    loser_df['player_victory'] = 0  # Indicate the player lost the match

    # For losers, 'victory_dominant' is always False
    loser_df['victory_dominant'] = False

    # Combine winner and loser DataFrames
    player_df = pd.concat([winner_df, loser_df], ignore_index=True)
    #player_df = player_df.set_index(['player_id', 'tourney_date'])

    return player_df


# Function to categorize quantitative variables into quantiles
def transform_quant(df, column_name, new_column_name, num_quantiles=4):
    # Remove NaN values from the column to avoid errors
    non_na_series = df[column_name].copy().dropna()

    # Create categories based on quantiles, handling duplicates
    categories, bin_edges = pd.qcut(
        non_na_series,
        q=num_quantiles,
        labels=False,
        retbins=True,
        duplicates='drop'
    )

    # Generate labels based on bin edges
    labels = [f'{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}' for i in range(len(bin_edges) - 1)]

    # Assign labels to the categories
    df[new_column_name] = pd.cut(
        df[column_name],
        bins=bin_edges,
        labels=labels,
        include_lowest=True
    )

    return df


# Function to impute missing values
def transform_miss(df, quant_vars, qual_vars):
    # Imputation for quantitative variables
    for var in quant_vars:
        # Compute the mean at the tournament level and assign back to the original DataFrame
        df[var] = df.groupby('tourney_id')[var].transform(
            lambda x: x.fillna(x.mean())
        )
        # If some values are still missing, use the global mean
        df[var] = df[var].fillna(df[var].mean())

    # Imputation for qualitative variables
    for var in qual_vars:
        # Compute the mode at the tournament level and assign back to the original DataFrame
        df[var] = df.groupby('tourney_id')[var].transform(
            lambda x: x.fillna(
                x.mode().iloc[0] if not x.mode().empty else np.nan)
        )
        # If some values are still missing, use the global mode
        global_mode = df[var].mode().iloc[0] if not df[var].mode().empty else np.nan
        df[var] = df[var].fillna(global_mode)

    return df


def calculate_effort_measures(df_matches):
    """
    Calculates 'winner_lower_rank', 'sets_won_low' and 'dominant_victory_low' variables based on the match data.
    """
    # Ensure rankings are numeric
    df_matches['winner_rank'] = pd.to_numeric(df_matches['winner_rank'], errors='coerce')
    df_matches['loser_rank'] = pd.to_numeric(df_matches['loser_rank'], errors='coerce')

    df_matches['winner_lower_rank'] = df_matches['winner_rank'] > df_matches['loser_rank']

    # Analyze scores to calculate 'winner_sets_won', 'loser_sets_won', 'victory_dominant'
    def parse_score(row):
        score = row['score']
        winner_sets_won = 0
        loser_sets_won = 0
        dominant_set_count = 0
        # Use regex to extract set scores
        sets = re.findall(r'(\d+)-(\d+)(?:\(\d+\))?', str(score))
        for winner_set_score, loser_set_score in sets:
            winner_set_score = int(winner_set_score)
            loser_set_score = int(loser_set_score)
            if winner_set_score > loser_set_score:
                winner_sets_won += 1
                if winner_set_score >= 6 and loser_set_score <= 2:
                    dominant_set_count += 1
            else:
                loser_sets_won += 1
        victory_dominant = dominant_set_count >= 2
        return pd.Series({
            'winner_sets_won': winner_sets_won,
            'loser_sets_won': loser_sets_won,
            'victory_dominant': victory_dominant
        })

    # Apply the parsing function to the DataFrame
    score_parsed = df_matches.apply(parse_score, axis=1)
    df_matches = pd.concat([df_matches, score_parsed], axis=1)

    # Calculate 'sets_won_low' - Number of sets won by the lower-ranked player
    def get_sets_won_by_lower_ranked(row):
        if pd.notnull(row['winner_rank']) and pd.notnull(row['loser_rank']):
            if row['winner_rank'] > row['loser_rank']:
                # Winner is lower-ranked
                return row['winner_sets_won']
            else:
                # Loser is lower-ranked
                return row['loser_sets_won']
        else:
            return None

    df_matches['sets_won_low'] = df_matches.apply(get_sets_won_by_lower_ranked, axis=1)

    return df_matches
