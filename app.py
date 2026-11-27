"""Dashboard interactif AfriMarket.  Lancer :  streamlit run app.py"""
import pandas as pd
import plotly.express as px
import streamlit as st

import afrimarket_utils as u

st.set_page_config(page_title="AfriMarket – Dashboard", page_icon="📊", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return u.load_clean()


def fmt(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


df = get_data()

# ------------------------------ Filtres ------------------------------
st.sidebar.header("Filtres")
d_min, d_max = df.date_commande.min().date(), df.date_commande.max().date()
periode = st.sidebar.date_input("Période", (d_min, d_max), min_value=d_min, max_value=d_max)
villes = st.sidebar.multiselect("Villes", sorted(df.ville.unique()), default=sorted(df.ville.unique()))
cats = st.sidebar.multiselect("Catégories", sorted(df.categorie.unique()), default=sorted(df.categorie.unique()))
canaux = st.sidebar.multiselect("Canaux marketing", sorted(df.canal_marketing.unique()),
                                default=sorted(df.canal_marketing.unique()))
st.sidebar.caption("Marge brute estimée avec un taux de coût par catégorie : "
                   + ", ".join(f"{k} {int(v * 100)} %" for k, v in u.TAUX_COGS.items()))

if len(periode) != 2:
    st.info("Sélectionnez une date de début et une date de fin.")
    st.stop()
mask = (df.date_commande.dt.date.between(*periode) & df.ville.isin(villes)
        & df.categorie.isin(cats) & df.canal_marketing.isin(canaux))
f = df[mask]
if f.empty:
    st.warning("Aucune commande ne correspond aux filtres.")
    st.stop()

st.title("📊 AfriMarket – Pilotage de l'activité")
st.caption("Montants en UM (devise du dataset). CA = commandes livrées uniquement.")

# ------------------------------ KPI ------------------------------
liv = f[f.statut_commande == "Livrée"]
c = st.columns(6)
c[0].metric("CA total", fmt(f.chiffre_affaires.sum()))
c[1].metric("Profit net estimé", fmt(f.profit_net.sum()))
c[2].metric("Marge nette", f"{f.profit_net.sum() / max(f.chiffre_affaires.sum(), 1) * 100:.1f} %")
c[3].metric("Panier moyen", fmt(liv.montant_brut.mean()) if len(liv) else "–")
c[4].metric("Taux d'annulation", f"{f.indicateur_annulation.mean() * 100:.1f} %")
c[5].metric("Taux de retour", f"{f.indicateur_retour.mean() * 100:.1f} %")

tab_cat, tab_geo, tab_mkt, tab_cli, tab_data = st.tabs(
    ["📦 Catégories", "🌍 Villes", "📣 Marketing", "👥 Clients", "🗂️ Données"])

# ------------------------------ Catégories ------------------------------
with tab_cat:
    cat = f.groupby("categorie").agg(CA=("chiffre_affaires", "sum"), profit=("profit_net", "sum"),
                                     marge=("marge_brute", "sum"), retour=("indicateur_retour", "mean"),
                                     commandes=("id_commande", "count")).reset_index()
    cat["taux_retour_%"] = cat.retour * 100
    a, b = st.columns(2)
    a.plotly_chart(px.bar(cat, x="categorie", y=["CA", "profit"], barmode="group",
                          title="CA et profit net par catégorie"), width="stretch")
    b.plotly_chart(px.bar(cat, x="categorie", y="taux_retour_%", color="taux_retour_%",
                          color_continuous_scale="Reds", title="Taux de retour (%)"), width="stretch")
    evo = f.pivot_table(index="mois_label", columns="categorie", values="chiffre_affaires", aggfunc="sum")
    st.plotly_chart(px.line(evo, markers=True, title="Évolution mensuelle du CA par catégorie"),
                    width="stretch")
    hm = f.pivot_table(index="categorie", columns="canal_marketing", values="indicateur_retour", aggfunc="mean") * 100
    st.plotly_chart(px.imshow(hm, text_auto=".1f", color_continuous_scale="Reds",
                              title="Taux de retour (%) : catégorie × canal"), width="stretch")

# ------------------------------ Villes ------------------------------
with tab_geo:
    v = f.groupby("ville").agg(CA=("chiffre_affaires", "sum"), profit=("profit_net", "sum"),
                               annulation=("indicateur_annulation", "mean")).reset_index()
    v["taux_annulation_%"] = v.annulation * 100
    a, b, c3 = st.columns(3)
    a.plotly_chart(px.bar(v.sort_values("CA"), x="CA", y="ville", orientation="h", title="CA par ville"),
                   width="stretch")
    b.plotly_chart(px.bar(v.sort_values("profit"), x="profit", y="ville", orientation="h",
                          title="Profit net par ville"), width="stretch")
    c3.plotly_chart(px.bar(v.sort_values("taux_annulation_%"), x="taux_annulation_%", y="ville",
                           orientation="h", color="taux_annulation_%", color_continuous_scale="Reds",
                           title="Taux d'annulation (%)"), width="stretch")
    pv = f.pivot_table(index="mois_label", columns="ville", values="chiffre_affaires", aggfunc="sum")
    st.plotly_chart(px.line(pv, markers=True, title="Évolution mensuelle du CA par ville"),
                    width="stretch")
    if len(pv) >= 2:
        g = ((pv.iloc[-1] / pv.iloc[0] - 1) * 100).round(1).sort_values()
        st.plotly_chart(px.bar(g, orientation="h", title=f"Croissance {pv.index[-1]} vs {pv.index[0]} (%)",
                               labels={"value": "%", "ville": ""}), width="stretch")

# ------------------------------ Marketing ------------------------------
with tab_mkt:
    m = f.groupby("canal_marketing").agg(CA=("chiffre_affaires", "sum"), cout=("cout_marketing", "sum"),
                                         commandes=("id_commande", "count")).reset_index()
    m["ROI"] = (m.CA - m.cout) / m.cout
    m["part_budget_%"] = m.cout / m.cout.sum() * 100
    m["part_CA_%"] = m.CA / m.CA.sum() * 100
    prem = f.sort_values("date_commande").drop_duplicates("id_client")[["id_client", "canal_marketing"]]
    cl = f.groupby("id_client").size().rename("n").reset_index().merge(prem)
    m["rétention_%"] = m.canal_marketing.map(cl.assign(r=cl.n > 1).groupby("canal_marketing").r.mean() * 100)
    a, b = st.columns(2)
    a.plotly_chart(px.bar(m, x="canal_marketing", y="ROI", color="ROI", title="ROI = (CA − coût) / coût"),
                   width="stretch")
    b.plotly_chart(px.bar(m.melt("canal_marketing", ["part_budget_%", "part_CA_%"]), x="canal_marketing",
                          y="value", color="variable", barmode="group", title="Part du budget vs part du CA (%)"),
                   width="stretch")
    st.plotly_chart(px.bar(m, x="canal_marketing", y="rétention_%", title="Taux de rétention par canal (%)"),
                    width="stretch")
    st.dataframe(m.round(2), hide_index=True, width="stretch")

# ------------------------------ Clients ------------------------------
with tab_cli:
    cl = f.groupby("id_client").agg(nb_commandes=("id_commande", "count"), CA=("chiffre_affaires", "sum"),
                                    profit=("profit_net", "sum")).reset_index()
    a, b, c3 = st.columns(3)
    a.metric("Clients", fmt(len(cl)))
    b.metric("% clients récurrents", f"{(cl.nb_commandes > 1).mean() * 100:.1f} %")
    c3.metric("Commandes / client (moy.)", f"{cl.nb_commandes.mean():.1f}")
    p = cl.sort_values("CA", ascending=False).reset_index(drop=True)
    p["% clients"] = (p.index + 1) / len(p) * 100
    p["% CA cumulé"] = p.CA.cumsum() / p.CA.sum() * 100
    st.plotly_chart(px.line(p, x="% clients", y="% CA cumulé", title="Courbe de Pareto (80/20)"),
                    width="stretch")
    cl["segment"] = pd.cut(cl.nb_commandes, [0, 1, 3, 7, 1000],
                           labels=["Occasionnel (1)", "Régulier (2-3)", "Fidèle (4-7)", "VIP (8+)"])
    s = cl.groupby("segment", observed=True).agg(clients=("id_client", "count"), CA=("CA", "sum")).reset_index()
    a, b = st.columns(2)
    a.plotly_chart(px.pie(s, names="segment", values="clients", title="Clients par segment"),
                   width="stretch")
    b.plotly_chart(px.pie(s, names="segment", values="CA", title="CA par segment"), width="stretch")
    st.subheader("Top 10 clients")
    st.dataframe(cl.sort_values("CA", ascending=False).head(10)[["id_client", "nb_commandes", "CA", "profit"]]
                 .round(0), hide_index=True, width="stretch")

# ------------------------------ Données ------------------------------
with tab_data:
    st.dataframe(f.drop(columns=["mois"]), width="stretch", hide_index=True)
    st.download_button("Télécharger la sélection (CSV)", f.to_csv(index=False).encode("utf-8"),
                       "afrimarket_selection.csv", "text/csv")
