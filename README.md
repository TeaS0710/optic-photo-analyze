# Ollama Scene Pipeline v3

Pipeline unique d'analyse d'images pour produire, à partir de photos, des sorties structurées et éditoriales de meilleure qualité.

Le flux enchaîne :
- qualification du support ;
- observation du visible ;
- lecture du texte présent ;
- construction d'ancrages probatoires ;
- interprétation prudente ;
- critique de l'interprétation ;
- rédaction finale.

Configuration visée par défaut :
- `qwen3-vl:235b-cloud` pour les passes avec image : `support`, `observe`, `read`, `anchor` ;
- `qwen3.5:397b-cloud` pour les passes textuelles séquentielles : `interpret`, `critique`, `write`.

## Commandes

Installation :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Lancer le pipeline :

```bash
make run
```

Lancer sur une seule image :

```bash
make run-one
```

Exemple avec surcharge explicite des modèles et des paramètres :

```bash
make run VISION_MODEL="qwen3-vl:235b-cloud" REASONING_MODEL="qwen3.5:397b-cloud" LIMIT=1 WORKERS=1
```

Compatibilité :
- `MODEL` reste accepté et alimente le modèle vision si `VISION_MODEL` n'est pas défini.
- Les sorties JSON et Markdown incluent désormais des métriques de performance par étape et par image.

Exemple avec fichier de prompts :

```bash
make run PROMPT_OVERRIDES=configs/prompts.project_example.json
```

Désactiver le post-traitement LLM :

```bash
PYTHONPATH=src python src/ollama_scene_pipeline_v3.py --no-llm-postprocess
```

Exporter des données d'entraînement :

```bash
make export
make export-write
make export-critique
make export-interpret
```

Construire l'étude de corpus et le site :

```bash
make study
make site
```

Construire le bundle publiable :

```bash
make package-site
```

Tout exécuter :

```bash
make full
```

Nettoyer les sorties générées :

```bash
make clean
```
