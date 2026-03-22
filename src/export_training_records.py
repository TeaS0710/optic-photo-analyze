#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from prompts import (
    CRITIQUE_SYSTEM,
    INTERPRET_SYSTEM,
    WRITE_SYSTEM,
    apply_prompt_overrides,
    render_critique_user,
    render_interpret_user,
    render_write_user,
)
from schemas import AnchorMap, CritiquePack, InterpretationPack, SceneScan, SupportProfile, TextScan, WritePack


TASKS = ("interpret", "critique", "write")


def env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return default


def env_path(name: str, default: str) -> Path:
    return Path(env_value(name, f"LMSP_{name.removeprefix('OLLAMA_')}", default=default) or default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export de jeux d'entraînement à partir des analyses générées.")
    parser.add_argument("--analysis-dir", type=Path, default=env_path("OLLAMA_ANALYSIS_DIR", "output"), help="Dossier contenant les .analysis.json")
    parser.add_argument("--output", type=Path, default=env_path("OLLAMA_TRAINING_OUTPUT", "training_records.jsonl"), help="Fichier JSONL de sortie")
    parser.add_argument(
        "--task",
        choices=TASKS,
        default="write",
        help="Passe ciblée pour l'export d'entraînement.",
    )
    parser.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="Inclure aussi les exemples non relus. Déconseillé pour le fine-tuning.",
    )
    parser.add_argument(
        "--require-usable",
        action="store_true",
        help="N'exporter que les exemples explicitement marqués comme utilisables pour l'entraînement.",
    )
    parser.add_argument(
        "--prompt-overrides",
        type=Path,
        default=env_path("OLLAMA_PROMPT_OVERRIDES", "") if env_value("OLLAMA_PROMPT_OVERRIDES", "LMSP_PROMPT_OVERRIDES") else None,
        help="Fichier JSON optionnel de surcharge des prompts utilisé pour reconstruire les entrées d'entraînement.",
    )
    return parser.parse_args()



def iter_payloads(analysis_dir: Path):
    for path in sorted(analysis_dir.glob("*.analysis.json")):
        yield path, json.loads(path.read_text(encoding="utf-8"))


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    support = SupportProfile.model_validate(payload["support"]).model_dump()
    observe = SceneScan.model_validate(payload["observe"]).model_dump()
    text = TextScan.model_validate(payload["text"]).model_dump()
    anchors = AnchorMap.model_validate(payload["anchors"]).model_dump()
    interpretation = InterpretationPack.model_validate(payload["interpretation"]).model_dump()
    critique = CritiquePack.model_validate(payload["critique"]).model_dump()
    writing = WritePack.model_validate(payload["writing"]).model_dump()

    validated = dict(payload)
    validated["support"] = support
    validated["observe"] = observe
    validated["text"] = text
    validated["anchors"] = anchors
    validated["interpretation"] = interpretation
    validated["critique"] = critique
    validated["writing"] = writing
    return validated



def build_record(task: str, payload: dict[str, Any]) -> dict[str, Any]:
    support = payload["support"]
    observe = payload["observe"]
    text = payload["text"]
    anchors = payload["anchors"]
    interpretation = payload["interpretation"]
    critique = payload["critique"]
    writing = payload["writing"]

    if task == "interpret":
        system_prompt = INTERPRET_SYSTEM
        user_prompt = render_interpret_user(support, observe, text, anchors)
        assistant = json.dumps(interpretation, ensure_ascii=False, indent=2)
    elif task == "critique":
        system_prompt = CRITIQUE_SYSTEM
        user_prompt = render_critique_user(support, observe, text, anchors, interpretation)
        assistant = json.dumps(critique, ensure_ascii=False, indent=2)
    else:
        system_prompt = WRITE_SYSTEM
        user_prompt = render_write_user(support, observe, text, anchors, critique)
        assistant = json.dumps(writing, ensure_ascii=False, indent=2)

    quality = payload.get("quality", {})
    return {
        "image": payload["file_name"],
        "task": task,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "assistant": assistant,
        "source_run_metadata": payload.get("run_metadata", {}),
        "reviewed": bool(quality.get("reviewed", False)),
        "usable_for_training": bool(quality.get("usable_for_training", False)),
        "review_notes": quality.get("review_notes", ""),
        "auto_quality": quality.get("auto_quality", {}),
    }



def main() -> int:
    args = parse_args()
    apply_prompt_overrides(args.prompt_overrides)
    records = []
    skipped = 0
    for path, raw_payload in iter_payloads(args.analysis_dir):
        if "error" in raw_payload:
            print(f"Exclu {path.name}: analyse en erreur.")
            skipped += 1
            continue

        try:
            payload = validate_payload(raw_payload)
        except (KeyError, ValidationError, TypeError, ValueError) as exc:
            print(f"Exclu {path.name}: payload invalide ({exc}).")
            skipped += 1
            continue

        quality = payload.get("quality", {})
        reviewed = bool(quality.get("reviewed", False))
        usable = bool(quality.get("usable_for_training", False))

        if not args.include_unreviewed and not reviewed:
            continue
        if args.require_usable and not usable:
            continue

        records.append(build_record(args.task, payload))

    args.output.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else ""),
        encoding="utf-8",
    )
    print(f"{len(records)} enregistrements exportés vers {args.output} ({skipped} exclus)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
