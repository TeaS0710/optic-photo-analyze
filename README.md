# Ollama Scene Pipeline v3

Pipeline Python **séquentiel spécialisé** pour analyser des images avec **Ollama** et un modèle vision comme **gemma3:latest**.

La v3 ajoute la couche qui manquait à la v2 : une vraie **relecture critique automatique** entre l'interprétation et la rédaction.

Le but n'est plus seulement de produire une belle sortie, mais de forcer le pipeline à :
- détecter ses propres glissements ;
- corriger ses hypothèses trop fortes ;
- bloquer les conclusions séduisantes mais fragiles ;
- rédiger seulement après audit.

## Chaîne complète

```text
IMAGE
  │
  ├── [1] SUPPORT      → qualifier le support et poser les garde-fous
  ├── [2] OBSERVE      → décrire strictement le visible
  ├── [3] READ         → relever le texte visible avec prudence OCR
  ├── [4] ANCHOR       → construire des ancrages probatoires + anti-extrapolation
  ├── [5] INTERPRET    → proposer une lecture humaine / sociale / émotionnelle
  ├── [6] CRITIQUE     → auditer, scorer, corriger, réviser
  └── [7] WRITE        → rédiger à partir de la version révisée, pas de la brute
```

## Pourquoi cette v3 est plus convaincante

Le problème classique des pipelines multimodaux n'est pas l'absence d'idées, mais leur excès :
1. une observation légèrement incertaine ;
2. une interprétation plausible mais trop assurée ;
3. une rédaction brillante qui naturalise l'erreur.

La v3 casse cette chaîne.

### Ce qui change structurellement

- **INTERPRET** peut encore être ambitieux, mais il est traité comme une hypothèse de travail.
- **CRITIQUE** mesure la fidélité au visible, le risque de surinterprétation et la couverture par les preuves.
- **WRITE** rédige à partir des ancrages et de la **version révisée**, pas de l'interprétation brute.

Autrement dit :
- on autorise l'intelligence interprétative ;
- on refuse qu'elle passe directement en production.

## Structure du projet

```text
ollama_scene_pipeline_v3/
├── Makefile
├── data/
│   └── input/
├── output/
├── src/
│   ├── ollama_scene_pipeline_v3.py
│   ├── export_training_records.py
│   ├── prompts.py
│   └── schemas.py
├── README.md
└── requirements.txt
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commandes rapides

Le projet fournit maintenant un `Makefile` pour éviter de retaper les longues commandes CLI.

```bash
make help
make run
make run-one
make export
make export-write
make export-critique
make export-interpret
make clean
```

Variables surchargeables :

```bash
make run MODEL="gemma3:latest" LIMIT=1
make run INPUT_DIR=data/input OUTPUT_DIR=output
make run SYNC_TIMEOUT=600
make run PROMPT_OVERRIDES=configs/prompts.project_example.json
make export TASK=critique TRAINING_OUTPUT=training_critique.jsonl
make export ANALYSIS_DIR=output TRAINING_OUTPUT=training_records.jsonl
```

## Environnement

Variables d'environnement supportées par le pipeline principal :

```bash
export PYTHONPATH=src
export OLLAMA_INPUT_DIR=data/input
export OLLAMA_OUTPUT_DIR=output
export OLLAMA_MODEL="gemma3:latest"
export OLLAMA_LIMIT=0
export OLLAMA_SYNC_TIMEOUT=300
export OLLAMA_API_HOST=http://127.0.0.1:11434
export OLLAMA_PROMPT_OVERRIDES=
```

Variables d'environnement supportées par l'export :

```bash
export PYTHONPATH=src
export OLLAMA_ANALYSIS_DIR=output
export OLLAMA_TRAINING_OUTPUT=training_records.jsonl
export OLLAMA_PROMPT_OVERRIDES=
```

Le CLI reste prioritaire sur les variables d'environnement.

## Reproductibilité

Chaque fichier `*.analysis.json` embarque maintenant un bloc `run_metadata` avec :
- le modèle utilisé ;
- le timeout synchrone ;
- la configuration des passes ;
- le fichier de surcharge de prompts utilisé ;
- une empreinte SHA-256 de chaque prompt actif.

Cela permet de rejouer un lot avec les mêmes entrées de prompting, ou de comparer deux variantes proprement.

## Réutiliser le prompting sur d'autres projets

Le projet n'est plus figé sur un seul jeu de prompts. Tu peux fournir un fichier JSON de surcharge avec seulement les clés à remplacer.
Un exemple minimal est fourni dans `configs/prompts.project_example.json`.

Exemple :

```json
{
  "SUPPORT_SYSTEM": "Tu es un analyste de supports techniques.",
  "WRITE_USER": "À partir de l'image et de l'audit ci-dessous, rédige une synthèse technique exploitable."
}
```

Clés disponibles :
- `SUPPORT_SYSTEM`
- `OBSERVE_SYSTEM`
- `READ_SYSTEM`
- `ANCHOR_SYSTEM`
- `INTERPRET_SYSTEM`
- `CRITIQUE_SYSTEM`
- `WRITE_SYSTEM`
- `SUPPORT_USER`
- `OBSERVE_USER`
- `READ_USER`
- `ANCHOR_USER`
- `INTERPRET_USER`
- `CRITIQUE_USER`
- `WRITE_USER`

Tu peux donc garder le pipeline de raisonnement et changer seulement la couche métier.

## Exécution

Exécution recommandée via `make` :

```bash
make run
```

Exécution d'un seul fichier pour test rapide :

```bash
make run-one
```

Exécution avec variables surchargées :

```bash
make run MODEL="gemma3:latest" LIMIT=1
```

Équivalent CLI brut :

```bash
PYTHONPATH=src python src/ollama_scene_pipeline_v3.py \
  --input-dir data/input \
  --output-dir output \
  --model "gemma3:latest" \
  --api-host "http://127.0.0.1:11434"
```

Exécution pilotée par variables d'environnement :

```bash
export PYTHONPATH=src
export OLLAMA_INPUT_DIR=data/input
export OLLAMA_OUTPUT_DIR=output
export OLLAMA_MODEL="gemma3:latest"
export OLLAMA_API_HOST=http://127.0.0.1:11434
python src/ollama_scene_pipeline_v3.py
```

Exécution avec surcharge de prompts :

```bash
make run PROMPT_OVERRIDES=configs/prompts.project_example.json
```

## Sorties

Pour chaque image :
- `001_xxx.analysis.json` : toutes les passes structurées ;
- `001_xxx.report.md` : rapport lisible ;
- `manifest.json` : index global avec signaux de qualité et statut d'exécution.

### Nouveauté de la v3

Chaque JSON contient aussi :

```json
"quality": {
  "reviewed": false,
  "review_notes": "",
  "usable_for_training": false,
  "auto_quality": {
    "faithfulness_score": 82,
    "overreach_risk_score": 24,
    "evidence_coverage_score": 78,
    "writing_readiness": "prêt à rédiger",
    "severe_issue_count": 0,
    "hallucination_risk": "faible",
    "recommended_usable_for_training": true
  }
}
```

Ces champs servent de **pré-filtre**. Ils ne remplacent pas la revue humaine, mais ils évitent déjà de fine-tuner aveuglément sur des sorties fragiles.

## Réglages par passe

| Passe      | Température | Rôle |
|------------|-------------|------|
| support    | 0.10        | cadrer le support |
| observe    | 0.12        | décrire le visible |
| read       | 0.08        | OCR prudent |
| anchor     | 0.16        | structurer les preuves |
| interpret  | 0.24        | lecture humaine disciplinée |
| critique   | 0.10        | audit factuel et correction |
| write      | 0.42        | rédaction finale après audit |

## Stratégie de fine-tuning recommandée

### Ordre conseillé

Ne fine-tune pas tout d'un bloc.

Ordre conseillé :
1. **CRITIQUE**
2. **WRITE**
3. **INTERPRET**

### Pourquoi commencer par CRITIQUE

Parce que le vrai problème du projet initial n'était pas seulement le style final :
le modèle laissait passer des hypothèses trop fortes.

Fine-tuner **CRITIQUE** en premier permet d'apprendre au modèle à :
- repérer les glissements ;
- réduire les affirmations trop dures ;
- conserver les bonnes intuitions tout en coupant le hors-champ ;
- produire une version révisée plus stable.

### Pourquoi WRITE vient ensuite

Quand la révision critique est solide, tu peux ensuite fine-tuner **WRITE** pour améliorer :
- la densité ;
- la lisibilité ;
- la qualité photographique ;
- la tenue littéraire ;
- la réutilisabilité éditoriale.

### INTERPRET en dernier

INTERPRET est la passe la plus séduisante à entraîner, mais aussi la plus risquée.
Si tu la fine-tunes trop tôt, tu risques d'apprendre au modèle à surproduire des lectures humaines convaincantes mais instables.

## Export d'exemples d'entraînement

### Export pour la passe CRITIQUE

```bash
make export-critique
```

### Export pour la passe WRITE

```bash
make export-write
```

### Export pour la passe INTERPRET

```bash
make export-interpret
```

### Export générique piloté par variables `make`

```bash
make export TASK=write TRAINING_OUTPUT=training_write.jsonl
make export TASK=critique TRAINING_OUTPUT=training_critique.jsonl
make export TASK=interpret TRAINING_OUTPUT=training_interpret.jsonl
```

### Export strictement filtré

```bash
PYTHONPATH=src python src/export_training_records.py \
  --analysis-dir output \
  --output training_critique_strict.jsonl \
  --task critique \
  --require-usable
```

Export en mode variables d'environnement :

```bash
export PYTHONPATH=src
export OLLAMA_ANALYSIS_DIR=output
export OLLAMA_TRAINING_OUTPUT=training_write.jsonl
python src/export_training_records.py --task write
```

Export avec surcharge de prompts pour reconstruire les entrées d'entraînement :

```bash
make export-write PROMPT_OVERRIDES=configs/prompts.project_example.json
```

Le `Makefile` couvre les cas standards. Les options avancées comme `--require-usable` ou `--include-unreviewed` restent pour l'instant en CLI brut.

Par défaut, seuls les exemples `reviewed=true` sont exportés.
Les analyses en erreur ou invalides sont exclues automatiquement de l'export.

## Dépannage

### Erreur de sortie structurée invalide

Même avec Ollama et `gemma3`, certaines passes peuvent produire un JSON mal formé. Le pipeline retente maintenant automatiquement une fois avec des contraintes plus strictes.

### Erreur de connexion backend

Cette erreur indique généralement que :
- Ollama n'est pas lancé ;
- `OLLAMA_API_HOST` ou `API_HOST` ne pointe pas vers la bonne adresse ;
- le modèle indiqué dans `MODEL=...` n'est pas disponible localement.

En pratique :
- teste d'abord avec `make run-one` ;
- vérifie `curl -sS http://127.0.0.1:11434/api/tags` ;
- garde le même nom exact entre Ollama et `MODEL=...`.

### Interruption au clavier

`Ctrl+C` arrête maintenant le traitement proprement sans afficher un long traceback Python.

## Comment relire utilement les sorties

Quand tu fais ta revue humaine, regarde d'abord :
- les `issues` de la passe critique ;
- le score de fidélité ;
- le risque de surinterprétation ;
- les hypothèses émotionnelles révisées ;
- le texte final pour voir s'il réintroduit quelque chose qui avait été supprimé.

Ensuite seulement, marque :
- `reviewed=true`
- `usable_for_training=true`

## Ce qu'il faut éviter

N'entraîne pas sur des cas où :
- le texte visible est mal lu et pilote toute l'analyse ;
- les émotions ne reposent sur aucun ancrage clair ;
- la critique repère encore des problèmes élevés ;
- la rédaction finale redevient plus affirmée que la version révisée.

## Résumé stratégique

La v1 produisait des sorties ambitieuses.
La v2 les structurait mieux.
La v3 ajoute le maillon décisif :

**une intelligence critique intermédiaire qui sépare l'interprétation prometteuse de l'interprétation exploitable.**
