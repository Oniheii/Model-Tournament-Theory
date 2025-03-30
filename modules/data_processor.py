#data processor
import os
import pandas as pd
from modules.data_loader import download_file, load_data

data_dir = 'data/'
max_rows_per_sheet = 1048576  # Limite d'Excel pour les lignes


# Fonction pour reformater les dates (sans heure, minute, etc.)
def reformat_date_column(df, date_col):
    df[date_col] = pd.to_datetime(
        df[date_col], format='%Y%m%d', errors='coerce').dt.strftime('%d/%m/%Y')
    return df


# Fonction pour vérifier et ajuster les colonnes des fichiers mal alignés
def check_column_alignment(df):
    if 'tourney_id' not in df.columns and 'tourney_name' in df.columns:
        df.insert(0, 'tourney_id', '')  # Ajouter une colonne 'tourney_id' vide si elle est manquante
    return df


# Fonction principale pour traiter les fichiers par catégorie (annuels)
def process_annual_files(category, years_range):
    output_file = f"{data_dir}/{category}_merged.xlsx"

    # Vérification si le fichier final existe déjà
    if os.path.exists(output_file):
        print(f"{output_file} existe déjà. Processus ignoré.")
        return

    files_to_merge = []

    for year in years_range:
        file_name = f"{category}_{year}.csv"
        file_path = os.path.join(data_dir, file_name)
        download_file(file_name)  # Télécharger le fichier
        try:
            df = load_data(file_path, year)
            if df is not None:
                df = reformat_date_column(df, 'tourney_date')  # Reformater la date
                df = check_column_alignment(df)  # Vérifier l'alignement des colonnes
                files_to_merge.append(df)
        except FileNotFoundError:
            pass

    merge_and_save_data(files_to_merge, output_file)

    # Supprimer les fichiers individuels après fusion
    for year in years_range:
        file_name = f"{category}_{year}.csv"
        try:
            os.remove(os.path.join(data_dir, file_name))
        except FileNotFoundError:
            pass


# Fonction pour sauvegarder les fichiers fusionnés en gérant les grandes tables
def merge_and_save_data(dataframes, output_file):
    if len(dataframes) == 0:
        print("Aucune donnée à fusionner.")
        return

    merged_df = pd.concat(dataframes, ignore_index=True)

    if len(merged_df) > max_rows_per_sheet:
        pass
        # Diviser le fichier en plusieurs parties si trop grand
        # num_parts = (len(merged_df) // max_rows_per_sheet) + 1
        # for i in range(num_parts):
        # part_df = merged_df.iloc[i * max_rows_per_sheet: (i + 1) * max_rows_per_sheet]
        # part_output_file = output_file.replace('.xlsx', f'_part{i + 1}.xlsx')
        # part_df.to_excel(part_output_file, index=False)
        # print(f"Fichier {part_output_file} créé avec succès.")
    else:
        merged_df.to_excel(output_file, index=False)
        print(f"Fichier {output_file} créé avec succès.")


# Fonction pour traiter les fichiers spécifiques par décennie ou uniques
def process_special_files(file_name, output_file, year=None):
    # Vérifier si le fichier final existe déjà
    if os.path.exists(output_file):
        print(f"{output_file} existe déjà. Processus ignoré.")
        return

    download_file(file_name)  # Télécharger le fichier
    file_path = os.path.join(data_dir, file_name)
    df = load_data(file_path, year)
    if df is not None:
        df = reformat_date_column(df, 'tourney_date')  # Reformater la date
        df.to_excel(output_file, index=False)
        print(f"{output_file} créé avec succès.")
        os.remove(file_path)  # Supprimer le fichier téléchargé


# Fonction pour traiter les classements par décennie et les fusionner
def process_ranking_files():
    output_file = f"{data_dir}/atp_rankings_merged.xlsx"

    # Vérifier si le fichier final existe déjà
    if os.path.exists(output_file):
        print(f"{output_file} existe déjà. Processus ignoré.")
        return

    decades_files = {
        'atp_rankings_70s.csv': '1970s',
        'atp_rankings_80s.csv': '1980s',
        'atp_rankings_90s.csv': '1990s',
        'atp_rankings_00s.csv': '2000s',
        'atp_rankings_10s.csv': '2010s',
        'atp_rankings_20s.csv': '2020s',
        'atp_rankings_current.csv': '2024'
    }

    files_to_merge = []

    for file_name, period in decades_files.items():
        output_file_min = f"{data_dir}/atp_rankings_{period}.xlsx"

        # Vérifier si le fichier final existe déjà
        if os.path.exists(output_file_min):
            print(f"{output_file_min} existe déjà. Processus ignoré.")
            return

        download_file(file_name)  # Télécharger le fichier
        file_path = os.path.join(data_dir, file_name)

        df = load_data(file_path)
        if df is not None:
            df['période'] = period  # Ajouter une colonne indiquant la période (décennie)
            df = reformat_date_column(df, 'ranking_date')  # Reformater la date si nécessaire
            files_to_merge.append(df)

            df.to_excel(output_file_min, index=False)

        os.remove(file_path)  # Supprimer le fichier téléchargé

    print("Traitement des données de ranking...")
    # merge_and_save_data(files_to_merge, output_file)
