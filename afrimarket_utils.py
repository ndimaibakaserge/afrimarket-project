"""Chargement, nettoyage et feature engineering du dataset AfriMarket.

Module partagé par le notebook d'analyse et le dashboard Streamlit afin que
les deux utilisent exactement les mêmes règles de gestion.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "afrimarket_dataset_senior.csv"

# Hypothèse métier (le dataset ne contient pas les coûts d'achat) :
# coût des marchandises vendues en % du prix net, par catégorie.
TAUX_COGS = {"Électronique": 0.75, "Mode": 0.50, "Beauté": 0.40, "Maison": 0.60}

VILLES_OK = ["Abidjan", "Brazzaville", "Cotonou", "Dakar", "Douala",
             "Kinshasa", "Libreville", "Lomé"]


def audit(df_raw: pd.DataFrame) -> dict:
    """Retourne les indicateurs d'audit qualité sur le fichier brut."""
    return {
        "lignes": len(df_raw),
        "colonnes": df_raw.shape[1],
        "valeurs_manquantes": int(df_raw.isna().sum().sum()),
        "doublons_exacts": int(df_raw.duplicated().sum()),
        "prix_negatifs_ou_nuls": int((df_raw["prix_unitaire"] <= 0).sum()),
        "quantites_nulles": int((df_raw["quantite"] == 0).sum()),
        "remises_negatives": int((df_raw["remise"] < 0).sum()),
        "villes_mal_orthographiees": int((~df_raw["ville"].isin(VILLES_OK)).sum()),
        "categories_non_standard": int((~df_raw["categorie"].isin(
            ["Électronique", "Mode", "Beauté", "Maison"])).sum()),
        "statuts_non_standard": int((~df_raw["statut_commande"].isin(["Livrée", "Annulée", "Retournée"])).sum()),
    }


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df_raw: pd.DataFrame, return_log: bool = False):
    """Nettoie le dataset brut -> df_clean. Journalise chaque opération."""
    df = df_raw.copy()
    log = {}

    # 1. Dates -> datetime (format ISO attendu, coercition sinon)
    df["date_commande"] = pd.to_datetime(df["date_commande"], errors="coerce")
    log["dates_invalides"] = int(df["date_commande"].isna().sum())
    df = df.dropna(subset=["date_commande"])

    # 2. Catégories : casse + accents (« electronique » -> « Électronique »)
    df["categorie"] = (df["categorie"].str.strip().str.lower()
                       .replace({"electronique": "Électronique", "électronique": "Électronique",
                                 "mode": "Mode", "beauté": "Beauté", "beaute": "Beauté",
                                 "maison": "Maison"}))
    # 3. Villes : « Kinshassa » -> « Kinshasa »
    df["ville"] = df["ville"].str.strip().replace({"Kinshassa": "Kinshasa"})
    # 4. Statuts : « retournée » -> « Retournée »
    df["statut_commande"] = df["statut_commande"].str.strip().str.capitalize()

    # 5. Doublons (même id_commande, lignes identiques)
    n = len(df)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="id_commande", keep="first")
    log["doublons_supprimes"] = n - len(df)

    # 6. Remises négatives : une remise ne peut pas augmenter le prix -> 0
    log["remises_negatives_corrigees"] = int((df["remise"] < 0).sum())
    df["remise"] = df["remise"].clip(lower=0)

    # 7. Prix aberrants. Sentinelle -50 (impossible) -> NaN, imputé par la médiane
    #    du produit (à défaut de la catégorie). Les valeurs extrêmes hautes sont
    #    plafonnées par la règle IQR (Q3 + 3*IQR) au sein de chaque catégorie.
    df["prix_aberrant"] = df["prix_unitaire"] <= 0
    log["prix_invalides_imputes"] = int(df["prix_aberrant"].sum())
    prix = df["prix_unitaire"].where(~df["prix_aberrant"])
    med_prod = prix.groupby(df["nom_produit"]).transform("median")
    med_cat = prix.groupby(df["categorie"]).transform("median")
    df["prix_unitaire"] = prix.fillna(med_prod).fillna(med_cat)
    q1 = df.groupby("categorie")["prix_unitaire"].transform(lambda s: s.quantile(.25))
    q3 = df.groupby("categorie")["prix_unitaire"].transform(lambda s: s.quantile(.75))
    plafond = q3 + 3 * (q3 - q1)
    hauts = df["prix_unitaire"] > plafond
    log["prix_extremes_plafonnes"] = int(hauts.sum())
    df.loc[hauts, "prix_unitaire"] = plafond[hauts]

    # 8. Quantités nulles : commande sans article = ligne non exploitable -> supprimée
    log["quantites_nulles_supprimees"] = int((df["quantite"] <= 0).sum())
    df = df[df["quantite"] > 0]

    df = df.reset_index(drop=True)
    df = add_features(df)
    return (df, log) if return_log else df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering.

    Règles comptables (documentées dans le notebook) :
    - CA = montant net de remise des commandes *Livrées* uniquement ;
      annulées et retournées ne génèrent pas de revenu.
    - Coût de livraison supporté pour Livrée et Retournée (aller-retour perdu),
      pas pour Annulée (jamais expédiée).
    - Coût marketing supporté pour toute commande (le budget est déjà dépensé).
    """
    df = df.copy()
    df["mois"] = df["date_commande"].dt.to_period("M").dt.to_timestamp()
    df["mois_label"] = df["date_commande"].dt.strftime("%Y-%m")

    df["montant_brut"] = df["prix_unitaire"] * df["quantite"] * (1 - df["remise"])
    df["indicateur_retour"] = (df["statut_commande"] == "Retournée").astype(int)
    df["indicateur_annulation"] = (df["statut_commande"] == "Annulée").astype(int)
    livree = df["statut_commande"] == "Livrée"

    df["chiffre_affaires"] = np.where(livree, df["montant_brut"], 0.0)
    cogs = df["categorie"].map(TAUX_COGS)
    df["marge_brute"] = np.where(livree, df["montant_brut"] * (1 - cogs), 0.0)
    livraison = np.where(df["statut_commande"] == "Annulée", 0.0, df["cout_livraison"])
    df["cout_livraison_supporte"] = livraison
    df["profit_net"] = df["marge_brute"] - livraison - df["cout_marketing"]

    cli = df.groupby("id_client")
    df["nombre_commandes_par_client"] = cli["id_commande"].transform("count")
    # CLV simplifiée = profit net cumulé par client sur la période observée
    df["valeur_vie_client"] = cli["profit_net"].transform("sum")
    return df


def load_clean() -> pd.DataFrame:
    return clean(load_raw())
