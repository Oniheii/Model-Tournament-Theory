#data loader
import pandas as pd
from io import StringIO
import os
import requests

# URL de base et répertoire de téléchargement
base_url = 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/'
data_dir = 'data/'

# Création du répertoire si nécessaire
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


# Fonction pour télécharger et sauvegarder un fichier
def download_file(file_name):
    url = f"{base_url}{file_name}"
    file_path = os.path.join(data_dir, file_name)

    # Vérifier si le fichier existe déjà
    if os.path.exists(file_path):
        print(f"{file_name} existe déjà, téléchargement ignoré.")
        return

    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
        # print(f"{file_name} téléchargé avec succès.")
    else:
        print(f"Échec du téléchargement de {file_name}.")


# Fonction pour charger un fichier CSV ou TXT en DataFrame
def load_data(file_path, year=None):
    ext = file_path.split('.')[-1]
    if ext == 'csv':
        df = pd.read_csv(file_path)
    elif ext == 'txt':
        with open(file_path, 'r') as f:
            data = f.read()
        df = pd.read_csv(StringIO(data), sep="\t")
    else:
        return None

    if year:
        # Ajouter une colonne année si une année est donnée
        df['year'] = year
    return df


def load_rankings_data():
    print("Chargement des données de classement...")
    rankings_period = []
    period = ["1970s", "1980s", "1990s", "2000s", "2010s", "2020s", "2024"]
    for p in period:
        part = pd.read_excel(f'data/atp_rankings_{p}.xlsx')
        part['ranking_date'] = pd.to_datetime(part['ranking_date'], dayfirst=True)
        rankings_period.append(part)
    return pd.concat(rankings_period, ignore_index=True)
