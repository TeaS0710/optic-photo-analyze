# Photo Corpus Study Lab

Projet séparé pour étudier un corpus photographique au-delà du pipeline principal.

Objectifs :
- analyser la matière photographique des images ;
- extraire des métriques optiques, colorimétriques et compositionnelles ;
- aligner ces résultats avec les sorties sémantiques du pipeline principal ;
- produire des rapports par image, des familles, des figures et des tendances de corpus.

## Ce que le projet mesure

- format, ratio, orientation
- luminance, contraste, dynamique tonale
- saturation, chaleur chromatique, équilibre coloré
- densité de contours, netteté relative, texture, entropie
- centre de masse visuel et proximité avec la règle des tiers
- dominance horizontale / verticale des lignes
- tags sémantiques issus des `*.analysis.json`

## Sorties

- `artifacts/metrics/*.json`
- `artifacts/reports/*.md`
- `artifacts/figures/*.svg`
- `artifacts/figures/contact_sheet.png`
- `artifacts/families.json`
- `artifacts/semantic_trends.json`
- `artifacts/corpus_report.md`
- `artifacts/metrics_summary.csv`
- `artifacts/site/index.html`

## Environnement

Créer un environnement virtuel local dans cette sous-racine :

```bash
cd /home/saiga/Prog/ollama_scene_pipeline_v3/photo_corpus_study_lab
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Le fichier [`requirements.txt`](/home/saiga/Prog/ollama_scene_pipeline_v3/photo_corpus_study_lab/requirements.txt) contient les dépendances minimales :
- `numpy`
- `Pillow`

Par défaut, le `Makefile` utilise `python3`. Si tu veux forcer l’environnement virtuel local :

```bash
make all PYTHON=.venv/bin/python
```

## Workflow recommandé

1. lancer le pipeline principal dans la racine du projet pour générer ou mettre à jour les `*.analysis.json`
2. revenir dans cette sous-racine
3. lancer le labo photo

```bash
cd /home/saiga/Prog/ollama_scene_pipeline_v3
make run

cd /home/saiga/Prog/ollama_scene_pipeline_v3/photo_corpus_study_lab
make all
```

Si tu utilises le `.venv` local :

```bash
cd /home/saiga/Prog/ollama_scene_pipeline_v3/photo_corpus_study_lab
source .venv/bin/activate
make all PYTHON=.venv/bin/python
```

Le site statique généré automatiquement est ensuite disponible dans `artifacts/site/index.html`.

## Sources par défaut

- images: `../data/input`
- analyses du projet principal: `../output`

## Cibles Make

- `make extract` : extrait les métriques photo et les rapports par image
- `make report` : génère les figures, familles, tendances et le rapport corpus
- `make site` : génère le site statique à partir des artefacts
- `make all` : enchaîne `extract`, `report`, puis `site`
- `make clean` : supprime `artifacts`

## Public visé

Pensé pour des professionnels de l’image :
- photographes
- iconographes
- éditeurs photo
- curateurs
- chercheurs visuels
