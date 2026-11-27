# AfriMarket – Projet d'analyse de données

| Livrable | Fichier |
|---|---|
| Notebook (audit, cleaning, features, analyses, recommandations, conclusion) | `Analyse_AfriMarket.ipynb` |
| Résumé exécutif | `Resume_Executif_AfriMarket.docx` |
| Dashboard interactif | `app.py` |
| Règles de nettoyage / features partagées | `afrimarket_utils.py` |
| Données | `data/afrimarket_dataset_senior.csv` |
| Graphiques exportés | `figures/` |

## Lancer
```
pip install -r requirements.txt
streamlit run app.py          # dashboard
jupyter notebook Analyse_AfriMarket.ipynb   # exécuter depuis la racine du projet
```
Hypothèse clé : marge brute = CA × (1 − taux de coût par catégorie) – Électronique 75 %, Maison 60 %, Mode 50 %, Beauté 40 % (à remplacer par les coûts réels dans `afrimarket_utils.TAUX_COGS`).
