#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import socket
import time
from typing import Any
import urllib.error
import urllib.request

from pydantic import ValidationError
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from prompts import (
    ANCHOR_SYSTEM,
    CRITIQUE_SYSTEM,
    INTERPRET_SYSTEM,
    OBSERVE_SYSTEM,
    READ_SYSTEM,
    SUPPORT_SYSTEM,
    WRITE_SYSTEM,
    apply_prompt_overrides,
    prompt_fingerprints,
    render_anchor_user,
    render_critique_user,
    render_interpret_user,
    render_observe_user,
    render_read_user,
    render_support_user,
    render_write_user,
)
from schemas import AnchorMap, CritiquePack, InterpretationPack, SceneScan, SupportProfile, TextScan, WritePack

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

PASS_CONFIGS: dict[str, dict[str, Any]] = {
    "support": {"temperature": 0.10, "maxTokens": 700, "topPSampling": 0.90, "repeatPenalty": 1.05},
    "observe": {"temperature": 0.12, "maxTokens": 1100, "topPSampling": 0.90, "repeatPenalty": 1.05},
    "read": {"temperature": 0.08, "maxTokens": 1200, "topPSampling": 0.90, "repeatPenalty": 1.03},
    "anchor": {"temperature": 0.16, "maxTokens": 1300, "topPSampling": 0.90, "repeatPenalty": 1.05},
    "interpret": {"temperature": 0.24, "maxTokens": 1300, "topPSampling": 0.92, "repeatPenalty": 1.05},
    "critique": {"temperature": 0.08, "maxTokens": 900, "topPSampling": 0.80, "repeatPenalty": 1.08},
    "write": {"temperature": 0.36, "maxTokens": 1200, "topPSampling": 0.88, "repeatPenalty": 1.05},
}

STAGES_WITH_IMAGE = {"support", "observe", "read", "anchor"}


def env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return default


def env_path(name: str, default: str) -> Path:
    return Path(env_value(name, f"LMSP_{name.removeprefix('OLLAMA_')}", default=default) or default)


def env_int(name: str, default: int) -> int:
    value = env_value(name, f"LMSP_{name.removeprefix('OLLAMA_')}")
    return int(value) if value is not None else default


def env_str(name: str, default: str) -> str:
    return env_value(name, f"LMSP_{name.removeprefix('OLLAMA_')}", default=default) or default


def env_bool(name: str, default: bool) -> bool:
    value = env_value(name, f"LMSP_{name.removeprefix('OLLAMA_')}")
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_optional_str(name: str) -> str | None:
    value = env_value(name, f"LMSP_{name.removeprefix('OLLAMA_')}")
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline séquentiel spécialisé v3 pour Ollama.")
    parser.add_argument("--input-dir", type=Path, default=env_path("OLLAMA_INPUT_DIR", "data/input"), help="Dossier des images à analyser.")
    parser.add_argument("--output-dir", type=Path, default=env_path("OLLAMA_OUTPUT_DIR", "output"), help="Dossier des sorties.")
    parser.add_argument(
        "--model",
        type=str,
        default=env_str("OLLAMA_MODEL", "gemma3:latest"),
        help="Nom du modèle Ollama. Il doit accepter les images.",
    )
    parser.add_argument("--limit", type=int, default=env_int("OLLAMA_LIMIT", 0), help="Nombre max d'images à traiter. 0 = sans limite.")
    parser.add_argument("--sync-timeout", type=int, default=env_int("OLLAMA_SYNC_TIMEOUT", 300), help="Timeout HTTP Ollama en secondes.")
    parser.add_argument("--api-host", type=str, default=env_str("OLLAMA_API_HOST", "http://127.0.0.1:11434"), help="Adresse de l'API Ollama.")
    parser.add_argument(
        "--prompt-overrides",
        type=Path,
        default=env_path("OLLAMA_PROMPT_OVERRIDES", "") if env_value("OLLAMA_PROMPT_OVERRIDES", "LMSP_PROMPT_OVERRIDES") else None,
        help="Fichier JSON optionnel de surcharge des prompts pour réutiliser le pipeline sur d'autres projets.",
    )
    parser.add_argument(
        "--llm-postprocess",
        action="store_true",
        default=env_bool("OLLAMA_LLM_POSTPROCESS", False),
        help="Utiliser Ollama aussi pour les passes critique et write. Désactivé par défaut pour privilégier la robustesse.",
    )
    parser.add_argument(
        "--temporary-context",
        type=str,
        default=env_optional_str("OLLAMA_TEMPORARY_CONTEXT"),
        help="Contexte temporaire de lecture, à utiliser comme hypothèse sensible mais jamais comme preuve factuelle.",
    )
    return parser.parse_args()



def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )



def parse_json_string(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            decoded = json.loads(text[start : end + 1])
        else:
            raise
    if not isinstance(decoded, dict):
        raise TypeError(f"JSON structuré inattendu: {type(decoded)!r}")
    return decoded


def read_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def ollama_options(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "temperature": config["temperature"],
        "top_p": config["topPSampling"],
        "repeat_penalty": config["repeatPenalty"],
        "num_predict": config["maxTokens"],
    }


def call_ollama(
    api_host: str,
    timeout_seconds: int,
    payload: dict[str, Any],
    request_label: str,
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{api_host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Échec Ollama HTTP {exc.code}: {body}") from exc
    except (TimeoutError, socket.timeout) as exc:
        elapsed = time.perf_counter() - started_at
        model = payload.get("model", "<inconnu>")
        options = payload.get("options") or {}
        max_tokens = options.get("num_predict", "<inconnu>")
        raise RuntimeError(
            f"Timeout Ollama sur {request_label} après {elapsed:.1f}s "
            f"(timeout={timeout_seconds}s, model={model}, max_tokens={max_tokens})"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Échec de connexion à Ollama: {exc.reason}") from exc



def call_structured(
    stage_name: str,
    primary_model: str,
    api_host: str,
    timeout_seconds: int,
    image_path: Path,
    image_base64: str,
    system_prompt: str,
    user_prompt: str,
    response_format: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "model": primary_model,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "format": response_format.model_json_schema(),
        "options": ollama_options(config),
    }
    if stage_name in STAGES_WITH_IMAGE:
        payload["images"] = [image_base64]

    try:
        response = call_ollama(
            api_host=api_host,
            timeout_seconds=timeout_seconds,
            request_label=f"stage={stage_name}, model={primary_model}, file={image_path.name}",
            payload=payload,
        )
        decoded = parse_json_string(response.get("response", ""))
        return response_format.model_validate(decoded).model_dump()
    except (TypeError, json.JSONDecodeError, ValidationError, RuntimeError) as exc:
        raise RuntimeError(
            f"Échec de la passe '{stage_name}' pour {image_path.name}: {exc}"
        ) from exc



def derive_auto_quality(critique: dict[str, Any]) -> dict[str, Any]:
    faithfulness = int(critique.get("faithfulness_score", 0))
    overreach = int(critique.get("overreach_risk_score", 100))
    evidence = int(critique.get("evidence_coverage_score", 0))
    readiness = critique.get("writing_readiness", "à réviser")
    severe_issues = sum(1 for issue in critique.get("issues", []) if issue.get("severity") == "élevée")

    recommended_usable = (
        readiness == "prêt à rédiger"
        and faithfulness >= 75
        and evidence >= 65
        and overreach <= 35
        and severe_issues == 0
    )

    if overreach >= 70 or faithfulness <= 45:
        hallucination_risk = "élevé"
    elif overreach >= 40 or faithfulness <= 65:
        hallucination_risk = "moyen"
    else:
        hallucination_risk = "faible"

    return {
        "faithfulness_score": faithfulness,
        "overreach_risk_score": overreach,
        "evidence_coverage_score": evidence,
        "writing_readiness": readiness,
        "severe_issue_count": severe_issues,
        "hallucination_risk": hallucination_risk,
        "recommended_usable_for_training": recommended_usable,
    }


def _truncate_items(items: list[str], limit: int = 3) -> list[str]:
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    return cleaned[:limit]


def normalize_sentence(value: str) -> str:
    return " ".join(str(value).split()).strip()


def context_lens_phrase(temporary_context: str | None) -> str:
    context = normalize_sentence(temporary_context or "")
    if not context:
        return ""
    return (
        f" Si ce contexte est bien celui de la série, l'image peut aussi se lire à travers cet horizon provisoire: {context}. "
        "Cette lecture n'apporte aucune preuve supplémentaire et ne doit jamais servir à inventer des faits absents du visible."
    )


def sentence_case(value: str) -> str:
    text = normalize_sentence(value)
    if not text:
        return ""
    return text[0].upper() + text[1:]


def join_fragments(parts: list[str]) -> str:
    return " ".join(part.strip() for part in parts if normalize_sentence(part)).strip()


def filter_text_regions(text_payload: dict[str, Any]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for region in text_payload.get("text_regions", []):
        transcription = normalize_sentence(region.get("transcription", ""))
        notes = normalize_sentence(region.get("notes", "")).lower()
        if not transcription:
            continue
        if transcription.lower() in {"photographie", "photo", "image"} and ("support" in notes or "descript" in notes):
            continue
        filtered.append(
            {
                "region_label": normalize_sentence(region.get("region_label", "")) or "zone",
                "transcription": transcription,
                "confidence": region.get("confidence", "faible"),
                "notes": normalize_sentence(region.get("notes", "")),
            }
        )
    return filtered


def sanitize_text_payload(text_payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(text_payload)
    filtered_regions = filter_text_regions(text_payload)
    sanitized["text_regions"] = filtered_regions
    transcriptions = [region["transcription"] for region in filtered_regions if region["transcription"]]
    combined_text = normalize_sentence(", ".join(transcriptions))
    if combined_text:
        sanitized["has_text"] = "oui"
        sanitized["combined_text"] = combined_text
        if not sanitized.get("text_role") or normalize_sentence(str(sanitized.get("text_role", ""))).lower() in {"description du support", "indéterminé"}:
            sanitized["text_role"] = "logo" if len(transcriptions) == 1 and len(transcriptions[0]) <= 24 else "texte visible"
    else:
        sanitized["has_text"] = "non"
        sanitized["combined_text"] = ""
        sanitized["text_role"] = "aucun texte lisible"
    return sanitized


def sanitize_support_payload(support_payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(support_payload)
    support_kind = normalize_sentence(sanitized.get("support_kind", "photographie")) or "photographie"
    sanitized["support_kind"] = support_kind
    primary_focus = normalize_sentence(sanitized.get("primary_focus", ""))
    if len(primary_focus) > 120:
        primary_focus = "scène humaine"
    sanitized["primary_focus"] = primary_focus or "scène"
    depicts = normalize_sentence(sanitized.get("depicts_direct_scene", ""))
    if len(depicts) > 120:
        depicts = "oui"
    sanitized["depicts_direct_scene"] = depicts or "incertain"
    boundary = normalize_sentence(sanitized.get("support_boundary", ""))
    if "élément ajouté" in boundary.lower():
        boundary = "La scène est montrée directement dans le cadre, sans autre support représenté identifiable."
    sanitized["support_boundary"] = boundary or "frontière du support non déterminée"
    sanitized["reasoning_guardrails"] = _truncate_items(sanitized.get("reasoning_guardrails", []), limit=4)
    if not sanitized["reasoning_guardrails"]:
        sanitized["reasoning_guardrails"] = [
            "Ne pas attribuer d'identité ou de biographie sans preuve visible.",
            "Ne pas convertir une impression en fait établi.",
            "Laisser subsister les zones d'incertitude.",
        ]
    return sanitized


def sanitize_observe_payload(observe_payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(observe_payload)
    sanitized["scene_summary"] = normalize_sentence(sanitized.get("scene_summary", "")) or "Scène non résumée."
    sanitized["setting"] = normalize_sentence(sanitized.get("setting", "")) or "cadre non déterminé"
    sanitized["salient_objects"] = _truncate_items(sanitized.get("salient_objects", []), limit=5)
    sanitized["visible_actions"] = _truncate_items(sanitized.get("visible_actions", []), limit=5)
    sanitized["spatial_relations"] = _truncate_items(sanitized.get("spatial_relations", []), limit=5)
    sanitized["uncertainties"] = _truncate_items(sanitized.get("uncertainties", []), limit=4)
    return sanitized


def sanitize_interpretation_payload(interpretation_payload: dict[str, Any], temporary_context: str | None) -> dict[str, Any]:
    sanitized = dict(interpretation_payload)
    prohibited = _truncate_items(sanitized.get("prohibited_conclusions", []), limit=4)
    if normalize_sentence(temporary_context or ""):
        prohibited.append("Le contexte temporaire fourni ne suffit jamais à prouver le lieu, la date, le camp ou l'événement exact.")
    sanitized["prohibited_conclusions"] = _truncate_items(prohibited, limit=5)
    sanitized["alternative_readings"] = _truncate_items(sanitized.get("alternative_readings", []), limit=3)
    residual = normalize_sentence(sanitized.get("residual_uncertainty", ""))
    if not residual:
        residual = "Le sens général de la scène reste partiellement ouvert."
    sanitized["residual_uncertainty"] = residual
    return sanitized


def detail_phrase(observe: dict[str, Any], text: dict[str, Any]) -> str:
    details = []
    details.extend(_truncate_items(observe.get("salient_objects", []), limit=3))
    combined_text = normalize_sentence(text.get("combined_text", ""))
    if combined_text:
        details.append(f"le texte visible '{combined_text}'")
    if not details:
        return "quelques détails concrets du cadre"
    if len(details) == 1:
        return details[0]
    return ", ".join(details[:-1]) + f" et {details[-1]}"


def emotion_phrase(critique: dict[str, Any]) -> str:
    emotions = critique.get("revised_emotional_hypotheses") or []
    if not emotions:
        return ""
    emotion = emotions[0]
    refs = ", ".join(emotion.get("anchor_refs", []))
    return (
        f" Une tonalité de {normalize_sentence(emotion.get('emotion', 'fragilité')).lower()} peut être ressentie, "
        f"mais seulement comme une hypothèse appuyée sur {refs or 'les ancrages visibles'}."
    )


def build_fallback_support(image_path: Path) -> dict[str, Any]:
    payload = {
        "support_kind": "photographie",
        "primary_focus": "scène humaine",
        "depicts_direct_scene": "incertain",
        "support_boundary": "La scène est montrée directement dans le cadre, sans autre support représenté clairement identifiable.",
        "reasoning_guardrails": [
            "Ne pas attribuer d'identité ou de biographie sans preuve visible.",
            "Ne pas convertir une impression en fait établi.",
            f"L'image {image_path.name} doit être lue prudemment si des détails restent ambigus.",
        ],
    }
    return SupportProfile.model_validate(payload).model_dump()


def build_fallback_observe(support: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "scene_summary": "Scène humaine visible, décrite de manière prudente faute de sortie d'observation totalement stable.",
        "setting": "cadre non déterminé avec précision",
        "subjects": [],
        "salient_objects": [],
        "visible_actions": [],
        "spatial_relations": [],
        "composition": "composition non stabilisée",
        "lighting_and_color": "lumière et couleurs non stabilisées",
        "uncertainties": [
            "La description détaillée du visible a été remplacée par un fallback local.",
            f"Le support reste classé comme {support.get('support_kind', 'image')}.",
        ],
    }
    return SceneScan.model_validate(payload).model_dump()


def build_fallback_text(observe: dict[str, Any]) -> dict[str, Any]:
    object_hints = " ".join(observe.get("salient_objects", [])).lower()
    has_text = "incertain" if any(token in object_hints for token in ("logo", "texte", "inscription", "écran", "affiche")) else "non"
    payload = {
        "has_text": has_text,
        "text_regions": [],
        "combined_text": "",
        "language_guesses": [],
        "text_role": "aucun texte lisible" if has_text == "non" else "texte possiblement présent mais non stabilisé",
        "reading_limits": [
            "La passe de lecture du texte a échoué et a été remplacée par un fallback local."
        ],
    }
    return TextScan.model_validate(payload).model_dump()


def build_fallback_anchor(observe: dict[str, Any], text: dict[str, Any]) -> dict[str, Any]:
    anchors: list[dict[str, Any]] = []
    for index, item in enumerate(_truncate_items(observe.get("salient_objects", []), limit=2), start=1):
        anchors.append(
            {
                "anchor_id": f"A{index}",
                "observation": sentence_case(item),
                "supports": ["présence visible dans la scène"],
                "certainty": "moyenne",
                "anti_overreach": "Cet élément ne suffit pas à établir un contexte, une intention ou un récit complet.",
            }
        )
    if not anchors:
        anchors.append(
            {
                "anchor_id": "A1",
                "observation": "Présence humaine visible dans le cadre",
                "supports": ["scène humaine"],
                "certainty": "faible",
                "anti_overreach": "Cette observation ne suffit pas à déduire une relation, un lieu précis ou un événement.",
            }
        )
    payload = {
        "dominant_axes": _truncate_items(observe.get("spatial_relations", []), limit=3),
        "anchors": anchors,
        "safe_inferences": _truncate_items([
            observe.get("scene_summary", ""),
            text.get("text_role", ""),
        ], limit=3),
        "open_questions": _truncate_items(observe.get("uncertainties", []), limit=3),
        "do_not_claim": [
            "Ne pas transformer une impression générale en fait établi.",
            "Ne pas déduire un contexte historique précis sans preuve visible.",
        ],
    }
    return AnchorMap.model_validate(payload).model_dump()


def build_fallback_interpretation(
    observe: dict[str, Any],
    text: dict[str, Any],
    anchors: dict[str, Any],
    temporary_context: str | None,
) -> dict[str, Any]:
    scene_summary = normalize_sentence(observe.get("scene_summary", ""))
    text_role = normalize_sentence(text.get("text_role", ""))
    payload = {
        "core_reading": scene_summary or "Scène humaine lue de manière prudente, sans interprétation forte.",
        "social_dynamics": "Les rapports humains restent ouverts et ne peuvent pas être stabilisés avec certitude à partir de cette seule image.",
        "emotional_hypotheses": [],
        "text_scene_interaction": (
            f"Le texte visible joue un rôle de {text_role}." if text_role and text_role != "aucun texte lisible" else "Le texte visible n'apporte pas de cadrage interprétatif assuré."
        ),
        "alternative_readings": _truncate_items([
            "La scène peut relever d'une simple présence silencieuse.",
            "Elle peut aussi se lire comme un moment suspendu sans récit explicite.",
            context_lens_phrase(temporary_context).strip(),
        ], limit=3),
        "prohibited_conclusions": _truncate_items([
            "Ne pas déduire un lieu, une date ou un événement précis sans preuve visible.",
            *anchors.get("do_not_claim", []),
        ], limit=4),
        "residual_uncertainty": "; ".join(_truncate_items(anchors.get("open_questions", []), limit=3)) or "Le sens général de la scène reste partiellement ouvert.",
    }
    return InterpretationPack.model_validate(payload).model_dump()


def build_fallback_critique(
    interpretation: dict[str, Any],
    anchors: dict[str, Any],
    text: dict[str, Any],
    temporary_context: str | None,
) -> dict[str, Any]:
    open_questions = _truncate_items(anchors.get("open_questions", []), limit=3)
    safe_inferences = _truncate_items(anchors.get("safe_inferences", []), limit=3)
    prohibited = _truncate_items(interpretation.get("prohibited_conclusions", []), limit=3)
    revised_emotions = [
        item for item in interpretation.get("emotional_hypotheses", [])
        if item.get("confidence") in {"moyenne", "basse"}
    ][:2]

    issues = [
        {
            "issue_type": "fallback_technique",
            "severity": "moyenne",
            "location": "critique",
            "explanation": "La passe critique structurée Ollama a échoué. Cette version de secours conserve une révision prudente mais moins fine qu'un audit complet.",
            "suggested_fix": "Relancer ultérieurement la passe critique avec un modèle plus stable si une version éditoriale de haute qualité est nécessaire.",
        }
    ]
    overreach_score = 38
    if normalize_sentence(temporary_context or ""):
        issues.append(
            {
                "issue_type": "contexte_temporaire",
                "severity": "moyenne",
                "location": "interpretation",
                "explanation": "Un contexte externe temporaire peut enrichir la lecture humaine, mais il ne doit jamais devenir une preuve sur la scène, le lieu, le camp ou l'événement exact.",
                "suggested_fix": "Garder des formulations hypothétiques et revenir explicitement aux ancrages visibles pour toute affirmation concrète.",
            }
        )
        overreach_score = 46

    critique = {
        "faithfulness_score": 72,
        "overreach_risk_score": overreach_score,
        "evidence_coverage_score": 68,
        "writing_readiness": "à réviser",
        "global_assessment": "Audit de secours généré localement après échec de la passe critique Ollama. Le contenu reste exploitable mais demande une relecture humaine.",
        "issues": issues,
        "keep_as_is": _truncate_items(
            [
                "core_reading" if interpretation.get("core_reading") else "",
                "text_scene_interaction" if interpretation.get("text_scene_interaction") else "",
                "prohibited_conclusions" if prohibited else "",
            ]
        ),
        "revision_priorities": _truncate_items(
            [
                "resserrer les formulations spéculatives",
                "conserver uniquement les ancrages explicitement visibles",
                "signaler les zones d'incertitude restantes",
            ]
        ),
        "revised_core_reading": interpretation.get("core_reading", ""),
        "revised_social_dynamics": interpretation.get("social_dynamics", ""),
        "revised_emotional_hypotheses": revised_emotions,
        "revised_text_scene_interaction": interpretation.get("text_scene_interaction", ""),
        "revised_alternative_readings": _truncate_items(interpretation.get("alternative_readings", []), limit=2),
        "revised_prohibited_conclusions": prohibited,
        "revised_residual_uncertainty": "; ".join(open_questions) if open_questions else (
            text.get("text_role") or interpretation.get("residual_uncertainty", "Incertitude résiduelle non nulle.")
        ),
    }
    return CritiquePack.model_validate(critique).model_dump()


def build_fallback_writing(
    payload_file_name: str,
    support: dict[str, Any],
    observe: dict[str, Any],
    text: dict[str, Any],
    critique: dict[str, Any],
    temporary_context: str | None,
) -> dict[str, Any]:
    support_kind = normalize_sentence(support.get("support_kind", "image"))
    setting = normalize_sentence(observe.get("setting", "cadre non déterminé"))
    scene_summary = normalize_sentence(observe.get("scene_summary", "Scène non résumée."))
    combined_text = normalize_sentence(text.get("combined_text", ""))
    text_role = normalize_sentence(text.get("text_role", "indéterminé"))
    uncertainty = normalize_sentence(critique.get("revised_residual_uncertainty") or "Certaines zones restent incertaines.")
    revised_core = normalize_sentence(critique.get("revised_core_reading", scene_summary))
    revised_social = normalize_sentence(critique.get("revised_social_dynamics", ""))
    lens_phrase = context_lens_phrase(temporary_context)
    title = f"{support_kind.capitalize()} : {payload_file_name.rsplit('.', 1)[0]}"
    emotional_line = emotion_phrase(critique)
    detail_line = detail_phrase(observe, text)
    text_clause = f" Le texte visible relevé est: {combined_text}." if combined_text else ""

    writing = {
        "short_title": title[:80],
        "analytic_brief": (
            f"{sentence_case(scene_summary)} Le cadre visible correspond à {setting}.{text_clause} "
            f"La lecture retenue reste la suivante: {revised_core}.{lens_phrase} "
            f"Les limites à conserver explicitement demeurent: {uncertainty}"
        ),
        "human_commentary": (
            f"{sentence_case(scene_summary)} Ce qui retient ici n'est pas seulement le fait montré, mais la manière dont il demeure en suspens: "
            f"{detail_line} composent une scène qui résiste à l'explication rapide. "
            f"{revised_social or 'La relation entre les présences visibles ne se laisse pas fixer avec certitude.'}"
            f"{emotional_line}{lens_phrase} Le texte assume une lecture plus humaine et plus sensible, mais il se refuse à faire passer une impression pour une preuve."
        ),
        "photographic_commentary": (
            f"La lecture photographique s'appuie sur la composition visible, le cadrage et les rapports entre sujets et objets. "
            f"Le regard est conduit par {detail_line}, tandis que {setting.lower()} installe un espace de retrait plus que d'action. "
            f"La scène reste restituée sans extrapolation hors champ, avec une attention particulière au support {support_kind} et à sa matérialité."
        ),
        "literary_commentary": (
            f"Le texte retient une voix plus proche du témoignage que du commentaire neutre. Il décrit {scene_summary.lower()} sans fabriquer d'arrière-scène, "
            f"et laisse ce qui affleure au bord du dicible plutôt que de le transformer en certitude. "
            f"Le rôle du texte visible reste {text_role}. L'ambiguïté qui demeure ne doit pas être comblée: {uncertainty}"
        ),
        "keywords": _truncate_items(
            [support_kind, setting, *(observe.get("salient_objects", []) or []), *(text.get("language_guesses", []) or [])],
            limit=8,
        ),
    }
    return WritePack.model_validate(writing).model_dump()



def analyze_one(
    model: str,
    llm_postprocess: bool,
    temporary_context: str | None,
    api_host: str,
    timeout_seconds: int,
    image_path: Path,
    run_metadata: dict[str, Any],
    progress_hook: Any = None,
) -> dict[str, Any]:
    image_base64 = read_image_base64(image_path)
    stage_fallbacks: dict[str, str] = {}

    try:
        if progress_hook:
            progress_hook("support", "start")
        support = call_structured(
            stage_name="support",
            primary_model=model,
            api_host=api_host,
            timeout_seconds=timeout_seconds,
            image_path=image_path,
            image_base64=image_base64,
            system_prompt=SUPPORT_SYSTEM,
            user_prompt=render_support_user(image_path),
            response_format=SupportProfile,
            config=PASS_CONFIGS["support"],
        )
        if progress_hook:
            progress_hook("support", "done")
    except Exception as exc:
        support = build_fallback_support(image_path)
        stage_fallbacks["support"] = str(exc)
        if progress_hook:
            progress_hook("support", "fallback")

    try:
        if progress_hook:
            progress_hook("observe", "start")
        observe = call_structured(
            stage_name="observe",
            primary_model=model,
            api_host=api_host,
            timeout_seconds=timeout_seconds,
            image_path=image_path,
            image_base64=image_base64,
            system_prompt=OBSERVE_SYSTEM,
            user_prompt=render_observe_user(support),
            response_format=SceneScan,
            config=PASS_CONFIGS["observe"],
        )
        if progress_hook:
            progress_hook("observe", "done")
    except Exception as exc:
        observe = build_fallback_observe(support)
        stage_fallbacks["observe"] = str(exc)
        if progress_hook:
            progress_hook("observe", "fallback")

    try:
        if progress_hook:
            progress_hook("read", "start")
        text = call_structured(
            stage_name="read",
            primary_model=model,
            api_host=api_host,
            timeout_seconds=timeout_seconds,
            image_path=image_path,
            image_base64=image_base64,
            system_prompt=READ_SYSTEM,
            user_prompt=render_read_user(support, observe),
            response_format=TextScan,
            config=PASS_CONFIGS["read"],
        )
        if progress_hook:
            progress_hook("read", "done")
    except Exception as exc:
        text = build_fallback_text(observe)
        stage_fallbacks["read"] = str(exc)
        if progress_hook:
            progress_hook("read", "fallback")

    try:
        if progress_hook:
            progress_hook("anchor", "start")
        anchors = call_structured(
            stage_name="anchor",
            primary_model=model,
            api_host=api_host,
            timeout_seconds=timeout_seconds,
            image_path=image_path,
            image_base64=image_base64,
            system_prompt=ANCHOR_SYSTEM,
            user_prompt=render_anchor_user(support, observe, text),
            response_format=AnchorMap,
            config=PASS_CONFIGS["anchor"],
        )
        if progress_hook:
            progress_hook("anchor", "done")
    except Exception as exc:
        anchors = build_fallback_anchor(observe, text)
        stage_fallbacks["anchor"] = str(exc)
        if progress_hook:
            progress_hook("anchor", "fallback")

    try:
        if progress_hook:
            progress_hook("interpret", "start")
        interpretation = call_structured(
            stage_name="interpret",
            primary_model=model,
            api_host=api_host,
            timeout_seconds=timeout_seconds,
            image_path=image_path,
            image_base64=image_base64,
            system_prompt=INTERPRET_SYSTEM,
            user_prompt=render_interpret_user(support, observe, text, anchors, temporary_context),
            response_format=InterpretationPack,
            config=PASS_CONFIGS["interpret"],
        )
        if progress_hook:
            progress_hook("interpret", "done")
    except Exception as exc:
        interpretation = build_fallback_interpretation(observe, text, anchors, temporary_context)
        stage_fallbacks["interpret"] = str(exc)
        if progress_hook:
            progress_hook("interpret", "fallback")
    support = sanitize_support_payload(support)
    observe = sanitize_observe_payload(observe)
    text = sanitize_text_payload(text)
    interpretation = sanitize_interpretation_payload(interpretation, temporary_context)

    critique_fallback_used = not llm_postprocess
    critique_fallback_reason = "disabled_by_default_for_resilience" if not llm_postprocess else None
    if llm_postprocess:
        try:
            if progress_hook:
                progress_hook("critique", "start")
            critique = call_structured(
                stage_name="critique",
                primary_model=model,
                api_host=api_host,
                timeout_seconds=timeout_seconds,
                image_path=image_path,
                image_base64=image_base64,
                system_prompt=CRITIQUE_SYSTEM,
                user_prompt=render_critique_user(support, observe, text, anchors, interpretation),
                response_format=CritiquePack,
                config=PASS_CONFIGS["critique"],
            )
            if progress_hook:
                progress_hook("critique", "done")
        except Exception as exc:
            critique = build_fallback_critique(interpretation, anchors, text, temporary_context)
            critique_fallback_used = True
            critique_fallback_reason = str(exc)
            if progress_hook:
                progress_hook("critique", "fallback")
    else:
        if progress_hook:
            progress_hook("critique", "fallback")
        critique = build_fallback_critique(interpretation, anchors, text, temporary_context)

    write_fallback_used = not llm_postprocess
    write_fallback_reason = "disabled_by_default_for_resilience" if not llm_postprocess else None
    if llm_postprocess:
        try:
            if progress_hook:
                progress_hook("write", "start")
            writing = call_structured(
                stage_name="write",
                primary_model=model,
                api_host=api_host,
                timeout_seconds=timeout_seconds,
                image_path=image_path,
                image_base64=image_base64,
                system_prompt=WRITE_SYSTEM,
                user_prompt=render_write_user(support, observe, text, anchors, critique, temporary_context),
                response_format=WritePack,
                config=PASS_CONFIGS["write"],
            )
            if progress_hook:
                progress_hook("write", "done")
        except Exception as exc:
            writing = build_fallback_writing(image_path.name, support, observe, text, critique, temporary_context)
            write_fallback_used = True
            write_fallback_reason = str(exc)
            if progress_hook:
                progress_hook("write", "fallback")
    else:
        if progress_hook:
            progress_hook("write", "fallback")
        writing = build_fallback_writing(image_path.name, support, observe, text, critique, temporary_context)

    auto_quality = derive_auto_quality(critique)

    payload = {
        "file_name": image_path.name,
        "run_metadata": run_metadata,
        "support": support,
        "observe": observe,
        "text": text,
        "anchors": anchors,
        "interpretation": interpretation,
        "critique": critique,
        "writing": writing,
        "quality": {
            "reviewed": False,
            "review_notes": "",
            "usable_for_training": False,
            "auto_quality": auto_quality,
        },
    }
    if critique_fallback_used or write_fallback_used:
        payload["pipeline_warnings"] = {
            "critique_fallback_used": critique_fallback_used,
            "critique_fallback_reason": critique_fallback_reason,
            "write_fallback_used": write_fallback_used,
            "write_fallback_reason": write_fallback_reason,
        }
    if stage_fallbacks:
        payload.setdefault("pipeline_warnings", {})
        payload["pipeline_warnings"]["stage_fallbacks"] = stage_fallbacks
    return payload


def build_error_payload(image_path: Path, error: Exception, run_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "file_name": image_path.name,
        "run_metadata": run_metadata,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "quality": {
            "reviewed": False,
            "review_notes": "",
            "usable_for_training": False,
            "auto_quality": {
                "faithfulness_score": 0,
                "overreach_risk_score": 100,
                "evidence_coverage_score": 0,
                "writing_readiness": "à bloquer",
                "severe_issue_count": 0,
                "hallucination_risk": "élevé",
                "recommended_usable_for_training": False,
            },
        },
    }



def report_markdown(payload: dict[str, Any], model_name: str) -> str:
    support = payload["support"]
    observe = payload["observe"]
    text = payload["text"]
    anchors = payload["anchors"]
    interpretation = payload["interpretation"]
    critique = payload["critique"]
    writing = payload["writing"]
    quality = payload["quality"]
    auto_quality = quality.get("auto_quality", {})

    lines: list[str] = []
    lines += [f"# {writing['short_title']}", ""]
    lines += [f"**Fichier** : `{payload['file_name']}`  ", f"**Modèle** : `{model_name}`", ""]

    lines += ["## 1. Profil de support", ""]
    lines += [f"**Nature du support** : {support['support_kind']}", ""]
    lines += [f"**Focalisation primaire** : {support['primary_focus']}", ""]
    lines += [f"**Scène directe ou support représenté** : {support['depicts_direct_scene']}", ""]
    lines += [f"**Frontière du support** : {support['support_boundary']}", ""]
    lines += ["### Garde-fous", ""]
    for item in support["reasoning_guardrails"]:
        lines.append(f"- {item}")

    lines += ["", "## 2. Observation", ""]
    lines += [observe["scene_summary"], ""]
    lines += [f"**Cadre / environnement** : {observe['setting']}", ""]
    lines += ["### Sujets", ""]
    for subject in observe["subjects"]:
        lines.append(
            f"- **{subject['role']}** — {subject['description']} | posture/état : {subject['posture_or_state']} | rôle visuel : {subject['salience']}"
        )
    lines += ["", "### Objets saillants", ""]
    for item in observe["salient_objects"]:
        lines.append(f"- {item}")
    lines += ["", "### Actions visibles", ""]
    for item in observe["visible_actions"]:
        lines.append(f"- {item}")
    lines += ["", "### Relations spatiales", ""]
    for item in observe["spatial_relations"]:
        lines.append(f"- {item}")
    lines += ["", f"**Composition** : {observe['composition']}", ""]
    lines += [f"**Lumière / couleur** : {observe['lighting_and_color']}", ""]
    lines += ["### Incertitudes", ""]
    for item in observe["uncertainties"]:
        lines.append(f"- {item}")

    lines += ["", "## 3. Texte visible", ""]
    lines += [f"**Texte présent** : {text['has_text']}", ""]
    if text["text_regions"]:
        for region in text["text_regions"]:
            lines.append(f"- **{region['region_label']}** ({region['confidence']}) — {region['transcription'] or '[illisible]'}")
            if region["notes"]:
                lines.append(f"  - Notes : {region['notes']}")
    if text["combined_text"].strip():
        lines += ["", "> " + text["combined_text"].replace("\n", "\n> "), ""]
    lines += [f"**Fonction textuelle** : {text['text_role']}", ""]
    if text["language_guesses"]:
        lines += [f"**Langues probables** : {', '.join(text['language_guesses'])}", ""]
    if text["reading_limits"]:
        lines += ["### Limites de lecture", ""]
        for item in text["reading_limits"]:
            lines.append(f"- {item}")

    lines += ["", "## 4. Carte d'ancrages", ""]
    if anchors["dominant_axes"]:
        lines += [f"**Axes dominants** : {', '.join(anchors['dominant_axes'])}", ""]
    for anchor in anchors["anchors"]:
        lines += [f"### {anchor['anchor_id']}", ""]
        lines += [f"**Observation** : {anchor['observation']}", ""]
        lines += ["**Permet de soutenir** :", ""]
        for item in anchor["supports"]:
            lines.append(f"- {item}")
        lines += ["", f"**Certitude** : {anchor['certainty']}", ""]
        lines += [f"**À ne pas extrapoler** : {anchor['anti_overreach']}", ""]
    if anchors["safe_inferences"]:
        lines += ["### Inférences raisonnables", ""]
        for item in anchors["safe_inferences"]:
            lines.append(f"- {item}")
    if anchors["open_questions"]:
        lines += ["", "### Questions ouvertes", ""]
        for item in anchors["open_questions"]:
            lines.append(f"- {item}")
    if anchors["do_not_claim"]:
        lines += ["", "### À ne pas affirmer", ""]
        for item in anchors["do_not_claim"]:
            lines.append(f"- {item}")

    lines += ["", "## 5. Interprétation brute", ""]
    lines += [f"**Lecture centrale** : {interpretation['core_reading']}", ""]
    lines += [f"**Dynamiques sociales / humaines** : {interpretation['social_dynamics']}", ""]
    lines += [f"**Texte et scène** : {interpretation['text_scene_interaction']}", ""]
    lines += ["### Hypothèses émotionnelles", ""]
    for item in interpretation["emotional_hypotheses"]:
        refs = ", ".join(item["anchor_refs"])
        lines.append(f"- **{item['emotion']}** ({item['confidence']}) — ancrages : {refs}. {item['reason']}")
    if interpretation["alternative_readings"]:
        lines += ["", "### Lectures alternatives", ""]
        for item in interpretation["alternative_readings"]:
            lines.append(f"- {item}")
    if interpretation["prohibited_conclusions"]:
        lines += ["", "### Conclusions interdites", ""]
        for item in interpretation["prohibited_conclusions"]:
            lines.append(f"- {item}")
    lines += ["", f"**Incertitude résiduelle** : {interpretation['residual_uncertainty']}", ""]

    lines += ["## 6. Audit critique", ""]
    lines += [f"**Fidélité au visible** : {critique['faithfulness_score']}/100  ", f"**Risque de surinterprétation** : {critique['overreach_risk_score']}/100  ", f"**Couverture par les preuves** : {critique['evidence_coverage_score']}/100", ""]
    lines += [f"**État pour la rédaction** : {critique['writing_readiness']}", ""]
    lines += [f"**Diagnostic global** : {critique['global_assessment']}", ""]
    if critique["keep_as_is"]:
        lines += ["### À conserver", ""]
        for item in critique["keep_as_is"]:
            lines.append(f"- {item}")
    if critique["issues"]:
        lines += ["", "### Problèmes détectés", ""]
        for issue in critique["issues"]:
            lines.append(f"- **{issue['issue_type']}** ({issue['severity']}) — `{issue['location']}` : {issue['explanation']}")
            lines.append(f"  - Correction : {issue['suggested_fix']}")
    if critique["revision_priorities"]:
        lines += ["", "### Priorités de révision", ""]
        for item in critique["revision_priorities"]:
            lines.append(f"- {item}")

    lines += ["", "## 7. Interprétation révisée", ""]
    lines += [f"**Lecture centrale révisée** : {critique['revised_core_reading']}", ""]
    lines += [f"**Dynamiques sociales révisées** : {critique['revised_social_dynamics']}", ""]
    lines += [f"**Texte et scène (révisé)** : {critique['revised_text_scene_interaction']}", ""]
    lines += ["### Hypothèses émotionnelles révisées", ""]
    for item in critique["revised_emotional_hypotheses"]:
        refs = ", ".join(item["anchor_refs"])
        lines.append(f"- **{item['emotion']}** ({item['confidence']}) — ancrages : {refs}. {item['reason']}")
    if critique["revised_alternative_readings"]:
        lines += ["", "### Lectures alternatives révisées", ""]
        for item in critique["revised_alternative_readings"]:
            lines.append(f"- {item}")
    if critique["revised_prohibited_conclusions"]:
        lines += ["", "### Conclusions à bloquer", ""]
        for item in critique["revised_prohibited_conclusions"]:
            lines.append(f"- {item}")
    lines += ["", f"**Incertitude finale** : {critique['revised_residual_uncertainty']}", ""]

    lines += ["## 8. Sorties éditoriales", ""]
    lines += ["### Résumé analytique", "", writing["analytic_brief"], ""]
    lines += ["### Interprétation humaine", "", writing["human_commentary"], ""]
    lines += ["### Commentaire photographique", "", writing["photographic_commentary"], ""]
    lines += ["### Commentaire littéraire", "", writing["literary_commentary"], ""]
    lines += ["### Mots-clés", ""]
    lines.append(", ".join(writing["keywords"]))
    lines.append("")

    lines += ["## 9. Qualité et entraînement", ""]
    lines += [f"**Risque d'hallucination** : {auto_quality.get('hallucination_risk', 'inconnu')}  ", f"**Exemple recommandé pour l'entraînement** : {auto_quality.get('recommended_usable_for_training', False)}  ", f"**Revu humainement** : {quality.get('reviewed', False)}", ""]
    if quality.get("review_notes"):
        lines += [f"**Notes de revue** : {quality['review_notes']}", ""]

    return "\n".join(lines)



def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_prompt_overrides(args.prompt_overrides)

    images = list_images(args.input_dir)
    if args.limit > 0:
        images = images[: args.limit]

    if not images:
        raise SystemExit(f"Aucune image trouvée dans {args.input_dir}")

    manifest: list[dict[str, Any]] = []
    stage_names = ("support", "observe", "read", "anchor", "interpret", "critique", "write")
    run_metadata = {
        "backend": "ollama",
        "api_host": args.api_host,
        "model": args.model,
        "llm_postprocess": args.llm_postprocess,
        "temporary_context": args.temporary_context,
        "sync_timeout": args.sync_timeout,
        "pass_configs": PASS_CONFIGS,
        "prompt_overrides": str(args.prompt_overrides) if args.prompt_overrides else None,
        "prompt_fingerprints": prompt_fingerprints(),
    }

    for index, image_path in enumerate(images, start=1):
        stem = f"{index:03d}_{image_path.stem}"
        print(f"[{index}/{len(images)}] {image_path.name}")
        json_path = args.output_dir / f"{stem}.analysis.json"
        md_path = args.output_dir / f"{stem}.report.md"
        stage_bar = None
        completed_stages: set[str] = set()

        if tqdm is not None:
            stage_bar = tqdm(
                total=len(stage_names),
                desc=f"{index}/{len(images)} {image_path.name[:32]}",
                unit="stage",
                leave=False,
            )

        def progress_hook(stage_name: str, status: str) -> None:
            if stage_bar is None:
                return
            label = stage_name if status == "done" else f"{stage_name} ({status})"
            stage_bar.set_postfix_str(label)
            if status in {"done", "fallback"} and stage_name not in completed_stages:
                completed_stages.add(stage_name)
                stage_bar.update(1)

        try:
            payload = analyze_one(
                args.model,
                args.llm_postprocess,
                args.temporary_context,
                args.api_host,
                args.sync_timeout,
                image_path,
                run_metadata,
                progress_hook=progress_hook,
            )
        except KeyboardInterrupt:
            if stage_bar is not None:
                stage_bar.close()
            print("\nInterruption utilisateur. Arrêt du traitement.")
            break
        except Exception as exc:
            if stage_bar is not None:
                stage_bar.close()
            payload = build_error_payload(image_path, exc, run_metadata)
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest.append(
                {
                    "file_name": image_path.name,
                    "json": json_path.name,
                    "markdown": None,
                    "title": None,
                    "reviewed": False,
                    "hallucination_risk": payload["quality"]["auto_quality"]["hallucination_risk"],
                    "writing_readiness": payload["quality"]["auto_quality"]["writing_readiness"],
                    "model": args.model,
                    "prompt_overrides": str(args.prompt_overrides) if args.prompt_overrides else None,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            print(f"  ERREUR: {exc}")
            continue
        finally:
            if stage_bar is not None:
                stage_bar.set_postfix_str("done")
                stage_bar.close()

        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(report_markdown(payload, args.model), encoding="utf-8")

        manifest.append(
            {
                "file_name": image_path.name,
                "json": json_path.name,
                "markdown": md_path.name,
                "title": payload["writing"]["short_title"],
                "reviewed": payload["quality"]["reviewed"],
                "hallucination_risk": payload["quality"]["auto_quality"]["hallucination_risk"],
                "writing_readiness": payload["quality"]["auto_quality"]["writing_readiness"],
                "model": args.model,
                "prompt_overrides": str(args.prompt_overrides) if args.prompt_overrides else None,
                "status": "ok",
                "error_type": None,
                "error_message": None,
            }
        )

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Terminé. {len(images)} image(s) traitée(s). Sorties dans {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
