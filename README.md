# 📚 StatRAG — RAG Évalué Mathématiquement

> Moteur de recherche et synthèse documentaire (`PostgreSQL + pgvector`) dont la fidélité, l'ancrage factuel et la pertinence sont quantifiés par métriques statistiques objectives et intervalles de confiance bootstrap (*From Data to Systems* — ElmasCore).

La majorité des prototypes RAG se contentent d'afficher une réponse générée sans mesurer le risque d'hallucination ni l'incertitude d'échantillonnage. **StatRAG** place la **validation statistique** et l'**isolation de sécurité en base (PostgreSQL RLS)** au cœur de l'architecture.

---

## 1. Choix technique et architecture

### Pourquoi Python + PostgreSQL (`pgvector` & `RLS`) ?
- **Python 3.11+** permet d'exprimer des fonctions pures et typées (`@dataclass(frozen=True)`) pour le découpage documentaire, l'algèbre vectorielle cosinus et l'inférence non paramétrique par rééchantillonnage bootstrap, sans imposer d'appel API payant pour tester la logique d'évaluation.
- **PostgreSQL + `pgvector` + Row-Level Security (`RLS`)** unifie dans une seule transaction relationnelle l'indexation vectorielle HNSW (`vector_cosine_ops`) et l'isolation multi-tenant stricte au niveau du moteur SQL (`FORCE ROW LEVEL SECURITY`), éliminant le risque de fuite inter-locataires souvent présent dans les bases vectorielles externes sans politique par ligne.

### Organisation par domaine métier (`ELMAS.md`)

```text
StatRAG/
├── src/statrag/
│   ├── ingestion/            # Découpage déterministe en segments avec chevauchement contrôlé
│   ├── retrieval/            # Similarité cosinus pure, filtrage par seuil & schéma SQL pgvector + RLS
│   └── evaluation/           # Fidélité (ancrage par assertion), pertinence & IC 95 % bootstrap
├── scripts/
│   └── run_benchmark_demo.py # Audit tabulaire reproductible sur jeu témoin (détection d'hallucination)
└── tests/
    └── test_statrag_pipeline.py
```

---

## 2. Métriques statistiques implémentées

1. **Fidélité / Ancrage factuel (*Faithfulness*)** : décompose la réponse en assertions élémentaires et mesure la proportion d'assertions étayées au-dessus d'un seuil lexical/sémantique par les segments récupérés (pénalise immédiatement les ajouts non sourçables).
2. **Pertinence du contexte (*Context Relevance*)** : ratio signal/bruit parmi les $k$ segments extraits vis-à-vis de la question posée.
3. **Qualité harmonique ($F_1$ d'ancrage)** : moyenne harmonique entre fidélité et pertinence — une réponse fluide mais non sourcée tombe immédiatement vers `0.000`.
4. **Intervalle de confiance à 95 % par Bootstrap** (`estimate_bootstrap_ci`) : rééchantillonnage déterministe ($B = 1000$, méthode des percentiles) quantifiant la dispersion réelle des performances sur un jeu de test sans hypothèse de normalité.

---

## 3. Lancement rapide

### Exécuter le benchmark témoin (zéro clé API requise)
```bash
python scripts/run_benchmark_demo.py
```

### Lancer la suite de tests unitaires
```bash
python -m unittest discover -s tests -v
```

### Déployer le schéma PostgreSQL + `pgvector` + `RLS`
```bash
psql "$STATRAG_DATABASE_URL" -f src/statrag/retrieval/pgvector_schema.sql
```

---

## 4. Feuille de route
- [x] Découpage documentaire immuable à recouvrement borné
- [x] Schéma PostgreSQL `pgvector` HNSW verrouillé par `FORCE ROW LEVEL SECURITY`
- [x] Évaluateur de fidélité + intervalle de confiance bootstrap à 95 %
- [ ] Adaptateur transactionnel `psycopg` avec injection automatique `SET LOCAL app.current_tenant_id`
- [ ] Couplage d'embeddings denses locaux (`ONNX` / `sentence-transformers`) pour le scoring d'entailment
