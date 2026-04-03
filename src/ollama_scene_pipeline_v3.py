#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Callable
import urllib.error
import urllib.request

from pydantic import ValidationError
from PIL import Image, ImageOps
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
    "support": {"temperature": 0.10, "maxTokens": 520, "topPSampling": 0.90, "repeatPenalty": 1.05},
    "observe": {"temperature": 0.12, "maxTokens": 820, "topPSampling": 0.90, "repeatPenalty": 1.05},
    "read": {"temperature": 0.08, "maxTokens": 900, "topPSampling": 0.90, "repeatPenalty": 1.03},
    "anchor": {"temperature": 0.16, "maxTokens": 980, "topPSampling": 0.90, "repeatPenalty": 1.05},
    "interpret": {"temperature": 0.22, "maxTokens": 960, "topPSampling": 0.90, "repeatPenalty": 1.05},
    "critique": {"temperature": 0.08, "maxTokens": 720, "topPSampling": 0.80, "repeatPenalty": 1.08},
    "write": {"temperature": 0.32, "maxTokens": 860, "topPSampling": 0.88, "repeatPenalty": 1.05},
}

STAGES_WITH_IMAGE = {"support", "observe", "read", "anchor"}


@dataclass(frozen=True)
class StageSpec:
    name: str
    system_prompt: str
    response_format: Any
    config: dict[str, Any]
    prompt_builder: Callable[[dict[str, Any], Path], str]
    fallback_builder: Callable[[dict[str, Any], Path], dict[str, Any]]
    sanitizer: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None
    llm_enabled: Callable[[dict[str, Any]], bool] = lambda _context: True
    disabled_reason: str | None = None


@dataclass(frozen=True)
class StageRunMetrics:
    elapsed_seconds: float
    used_model: str | None
    used_fallback: bool
    response_metrics: dict[str, Any]


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
        default=env_str("OLLAMA_MODEL", "qwen3-vl:235b-cloud"),
        help="Alias de compatibilité pour le modèle vision. Il doit accepter les images.",
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        default=env_str("OLLAMA_VISION_MODEL", ""),
        help="Modèle utilisé pour les passes avec image: support, observe, read, anchor.",
    )
    parser.add_argument(
        "--reasoning-model",
        type=str,
        default=env_str("OLLAMA_REASONING_MODEL", "qwen3.5:397b-cloud"),
        help="Modèle utilisé séquentiellement pour les passes textuelles: interpret, critique, write.",
    )
    parser.add_argument("--limit", type=int, default=env_int("OLLAMA_LIMIT", 0), help="Nombre max d'images à traiter. 0 = sans limite.")
    parser.add_argument("--sync-timeout", type=int, default=env_int("OLLAMA_SYNC_TIMEOUT", 300), help="Timeout HTTP Ollama en secondes.")
    parser.add_argument("--api-host", type=str, default=env_str("OLLAMA_API_HOST", "http://127.0.0.1:11434"), help="Adresse de l'API Ollama.")
    parser.add_argument("--api-token", type=str, default=env_optional_str("OLLAMA_API_TOKEN"), help="Token bearer optionnel pour un backend Ollama cloud ou distant.")
    parser.add_argument("--workers", type=int, default=env_int("OLLAMA_WORKERS", 2), help="Nombre d'images analysées en parallèle.")
    parser.add_argument(
        "--image-max-dimension",
        type=int,
        default=env_int("OLLAMA_IMAGE_MAX_DIMENSION", 1600),
        help="Dimension max des images envoyées au modèle pour réduire le temps et le volume transféré.",
    )
    parser.add_argument(
        "--image-jpeg-quality",
        type=int,
        default=env_int("OLLAMA_IMAGE_JPEG_QUALITY", 88),
        help="Qualité JPEG des images redimensionnées avant envoi au modèle.",
    )
    parser.add_argument(
        "--prompt-overrides",
        type=Path,
        default=env_path("OLLAMA_PROMPT_OVERRIDES", "") if env_value("OLLAMA_PROMPT_OVERRIDES", "LMSP_PROMPT_OVERRIDES") else None,
        help="Fichier JSON optionnel de surcharge des prompts pour réutiliser le pipeline sur d'autres projets.",
    )
    parser.add_argument(
        "--llm-postprocess",
        action="store_true",
        dest="llm_postprocess",
        default=env_bool("OLLAMA_LLM_POSTPROCESS", True),
        help="Utiliser Ollama aussi pour les passes critique et write. Activé par défaut pour maximiser la qualité.",
    )
    parser.add_argument(
        "--no-llm-postprocess",
        action="store_false",
        dest="llm_postprocess",
        help="Désactiver les passes critique et write via LLM et revenir aux fallbacks locaux.",
    )
    parser.add_argument(
        "--temporary-context",
        type=str,
        default=env_optional_str("OLLAMA_TEMPORARY_CONTEXT"),
        help="Contexte temporaire de lecture, à utiliser comme hypothèse sensible mais jamais comme preuve factuelle.",
    )
    args = parser.parse_args()
    if not args.vision_model:
        args.vision_model = args.model
    return args



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


def encode_analysis_image(image_path: Path, max_dimension: int, jpeg_quality: int) -> str:
    if max_dimension <= 0:
        return read_image_base64(image_path)

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        needs_resize = max(image.size) > max_dimension
        if needs_resize:
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=max(40, min(jpeg_quality, 95)), optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def ollama_options(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "temperature": config["temperature"],
        "top_p": config["topPSampling"],
        "repeat_penalty": config["repeatPenalty"],
        "num_predict": config["maxTokens"],
    }


def call_ollama(
    api_host: str,
    api_token: str | None,
    timeout_seconds: int,
    payload: dict[str, Any],
    request_label: str,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    request = urllib.request.Request(
        f"{api_host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
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
    api_token: str | None,
    timeout_seconds: int,
    image_path: Path,
    image_base64: str,
    system_prompt: str,
    user_prompt: str,
    response_format: Any,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
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
            api_token=api_token,
            timeout_seconds=timeout_seconds,
            request_label=f"stage={stage_name}, model={primary_model}, file={image_path.name}",
            payload=payload,
        )
        decoded = parse_json_string(response.get("response", ""))
        response_metrics = {
            key: response.get(key)
            for key in (
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            )
            if response.get(key) is not None
        }
        return response_format.model_validate(decoded).model_dump(), response_metrics
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


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        normalized = normalize_sentence(item)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def truncate_unique_items(items: list[str], limit: int) -> list[str]:
    return dedupe_preserve_order([str(item) for item in items])[:limit]


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


def sanitize_subjects(subjects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized_subjects: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for subject in subjects or []:
        role = normalize_sentence(subject.get("role", "")) or "sujet"
        description = normalize_sentence(subject.get("description", "")) or "présence non détaillée"
        key = (role.casefold(), description.casefold())
        if key in seen:
            continue
        seen.add(key)
        sanitized_subjects.append(
            {
                "role": role,
                "description": description,
                "posture_or_state": normalize_sentence(subject.get("posture_or_state", "")) or "état non stabilisé",
                "salience": normalize_sentence(subject.get("salience", "")) or "participe à la lecture visuelle de l'image",
            }
        )
    return sanitized_subjects[:6]


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
    sanitized["language_guesses"] = truncate_unique_items(sanitized.get("language_guesses", []), limit=4)
    sanitized["reading_limits"] = truncate_unique_items(sanitized.get("reading_limits", []), limit=5)
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
    sanitized["reasoning_guardrails"] = truncate_unique_items(sanitized.get("reasoning_guardrails", []), limit=5)
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
    sanitized["subjects"] = sanitize_subjects(sanitized.get("subjects", []))
    sanitized["salient_objects"] = truncate_unique_items(sanitized.get("salient_objects", []), limit=6)
    sanitized["visible_actions"] = truncate_unique_items(sanitized.get("visible_actions", []), limit=6)
    sanitized["spatial_relations"] = truncate_unique_items(sanitized.get("spatial_relations", []), limit=6)
    sanitized["uncertainties"] = truncate_unique_items(sanitized.get("uncertainties", []), limit=5)
    sanitized["composition"] = normalize_sentence(sanitized.get("composition", "")) or "composition non stabilisée"
    sanitized["lighting_and_color"] = normalize_sentence(sanitized.get("lighting_and_color", "")) or "lumière et couleurs non stabilisées"
    return sanitized


def sanitize_interpretation_payload(interpretation_payload: dict[str, Any], temporary_context: str | None) -> dict[str, Any]:
    sanitized = dict(interpretation_payload)
    prohibited = truncate_unique_items(sanitized.get("prohibited_conclusions", []), limit=5)
    if normalize_sentence(temporary_context or ""):
        prohibited.append("Le contexte temporaire fourni ne suffit jamais à prouver le lieu, la date, le camp ou l'événement exact.")
    sanitized["prohibited_conclusions"] = truncate_unique_items(prohibited, limit=6)
    sanitized["alternative_readings"] = truncate_unique_items(sanitized.get("alternative_readings", []), limit=4)
    residual = normalize_sentence(sanitized.get("residual_uncertainty", ""))
    if not residual:
        residual = "Le sens général de la scène reste partiellement ouvert."
    sanitized["residual_uncertainty"] = residual
    sanitized["core_reading"] = normalize_sentence(sanitized.get("core_reading", "")) or "Lecture prudente non stabilisée."
    sanitized["social_dynamics"] = normalize_sentence(sanitized.get("social_dynamics", "")) or "Les dynamiques humaines restent indéterminées."
    sanitized["text_scene_interaction"] = normalize_sentence(sanitized.get("text_scene_interaction", "")) or "Le rôle du texte reste secondaire ou indéterminé."
    emotions: list[dict[str, Any]] = []
    seen_emotions: set[tuple[str, tuple[str, ...]]] = set()
    for emotion in sanitized.get("emotional_hypotheses", []) or []:
        label = normalize_sentence(emotion.get("emotion", ""))
        refs = [normalize_sentence(ref) for ref in emotion.get("anchor_refs", []) if normalize_sentence(ref)]
        reason = normalize_sentence(emotion.get("reason", ""))
        if not label or not refs or not reason:
            continue
        key = (label.casefold(), tuple(refs))
        if key in seen_emotions:
            continue
        seen_emotions.add(key)
        emotions.append(
            {
                "emotion": label,
                "anchor_refs": refs[:3],
                "confidence": emotion.get("confidence", "basse"),
                "reason": reason,
            }
        )
    sanitized["emotional_hypotheses"] = emotions[:4]
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


def sanitize_anchor_payload(anchor_payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(anchor_payload)
    sanitized["dominant_axes"] = truncate_unique_items(sanitized.get("dominant_axes", []), limit=5)
    cleaned_anchors: list[dict[str, Any]] = []
    for index, anchor in enumerate(sanitized.get("anchors", []) or [], start=1):
        cleaned_anchors.append(
            {
                "anchor_id": normalize_sentence(anchor.get("anchor_id", "")) or f"A{index}",
                "observation": normalize_sentence(anchor.get("observation", "")) or "Observation visible non stabilisée.",
                "supports": truncate_unique_items(anchor.get("supports", []), limit=4) or ["présence visible dans la scène"],
                "certainty": anchor.get("certainty", "faible"),
                "anti_overreach": normalize_sentence(anchor.get("anti_overreach", "")) or "Cet ancrage ne suffit pas à établir un récit complet.",
            }
        )
    sanitized["anchors"] = cleaned_anchors[:6]
    sanitized["safe_inferences"] = truncate_unique_items(sanitized.get("safe_inferences", []), limit=5)
    sanitized["open_questions"] = truncate_unique_items(sanitized.get("open_questions", []), limit=5)
    sanitized["do_not_claim"] = truncate_unique_items(sanitized.get("do_not_claim", []), limit=6)
    return sanitized


def sanitize_critique_payload(critique_payload: dict[str, Any], interpretation: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(critique_payload)
    sanitized["global_assessment"] = normalize_sentence(sanitized.get("global_assessment", "")) or "Audit non stabilisé."
    sanitized["keep_as_is"] = truncate_unique_items(sanitized.get("keep_as_is", []), limit=5)
    sanitized["revision_priorities"] = truncate_unique_items(sanitized.get("revision_priorities", []), limit=5)
    sanitized["revised_core_reading"] = normalize_sentence(
        sanitized.get("revised_core_reading", "") or interpretation.get("core_reading", "")
    ) or "Lecture centrale révisée non stabilisée."
    sanitized["revised_social_dynamics"] = normalize_sentence(
        sanitized.get("revised_social_dynamics", "") or interpretation.get("social_dynamics", "")
    ) or "Les dynamiques humaines restent prudentes et ouvertes."
    sanitized["revised_text_scene_interaction"] = normalize_sentence(
        sanitized.get("revised_text_scene_interaction", "") or interpretation.get("text_scene_interaction", "")
    ) or "Le rôle du texte reste prudent et secondaire."
    sanitized["revised_alternative_readings"] = truncate_unique_items(sanitized.get("revised_alternative_readings", []), limit=4)
    sanitized["revised_prohibited_conclusions"] = truncate_unique_items(sanitized.get("revised_prohibited_conclusions", []), limit=6)
    sanitized["revised_residual_uncertainty"] = normalize_sentence(sanitized.get("revised_residual_uncertainty", "")) or "Une part d'incertitude demeure."
    sanitized["revised_emotional_hypotheses"] = sanitize_interpretation_payload(
        {"emotional_hypotheses": sanitized.get("revised_emotional_hypotheses", [])},
        None,
    )["emotional_hypotheses"]
    cleaned_issues: list[dict[str, Any]] = []
    for issue in sanitized.get("issues", []) or []:
        issue_type = normalize_sentence(issue.get("issue_type", "problème"))
        location = normalize_sentence(issue.get("location", "interprétation"))
        explanation = normalize_sentence(issue.get("explanation", ""))
        suggested_fix = normalize_sentence(issue.get("suggested_fix", ""))
        if not explanation or not suggested_fix:
            continue
        cleaned_issues.append(
            {
                "issue_type": issue_type,
                "severity": issue.get("severity", "moyenne"),
                "location": location,
                "explanation": explanation,
                "suggested_fix": suggested_fix,
            }
        )
    sanitized["issues"] = cleaned_issues[:8]
    return sanitized


def sanitize_writing_payload(writing_payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(writing_payload)
    sanitized["short_title"] = normalize_sentence(sanitized.get("short_title", "")) or "Analyse visuelle"
    for field in ("analytic_brief", "human_commentary", "photographic_commentary", "literary_commentary"):
        sanitized[field] = normalize_sentence(sanitized.get(field, ""))
    sanitized["keywords"] = truncate_unique_items(sanitized.get("keywords", []), limit=10)
    return sanitized


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


def build_stage_specs() -> tuple[StageSpec, ...]:
    return (
        StageSpec(
            name="support",
            system_prompt=SUPPORT_SYSTEM,
            response_format=SupportProfile,
            config=PASS_CONFIGS["support"],
            prompt_builder=lambda _ctx, image_path: render_support_user(image_path),
            fallback_builder=lambda _ctx, image_path: build_fallback_support(image_path),
            sanitizer=lambda value, _ctx: sanitize_support_payload(value),
        ),
        StageSpec(
            name="observe",
            system_prompt=OBSERVE_SYSTEM,
            response_format=SceneScan,
            config=PASS_CONFIGS["observe"],
            prompt_builder=lambda ctx, _image_path: render_observe_user(ctx["support"]),
            fallback_builder=lambda ctx, _image_path: build_fallback_observe(ctx["support"]),
            sanitizer=lambda value, _ctx: sanitize_observe_payload(value),
        ),
        StageSpec(
            name="read",
            system_prompt=READ_SYSTEM,
            response_format=TextScan,
            config=PASS_CONFIGS["read"],
            prompt_builder=lambda ctx, _image_path: render_read_user(ctx["support"], ctx["observe"]),
            fallback_builder=lambda ctx, _image_path: build_fallback_text(ctx["observe"]),
            sanitizer=lambda value, _ctx: sanitize_text_payload(value),
        ),
        StageSpec(
            name="anchor",
            system_prompt=ANCHOR_SYSTEM,
            response_format=AnchorMap,
            config=PASS_CONFIGS["anchor"],
            prompt_builder=lambda ctx, _image_path: render_anchor_user(ctx["support"], ctx["observe"], ctx["text"]),
            fallback_builder=lambda ctx, _image_path: build_fallback_anchor(ctx["observe"], ctx["text"]),
            sanitizer=lambda value, _ctx: sanitize_anchor_payload(value),
        ),
        StageSpec(
            name="interpret",
            system_prompt=INTERPRET_SYSTEM,
            response_format=InterpretationPack,
            config=PASS_CONFIGS["interpret"],
            prompt_builder=lambda ctx, _image_path: render_interpret_user(
                ctx["support"], ctx["observe"], ctx["text"], ctx["anchors"], ctx.get("temporary_context")
            ),
            fallback_builder=lambda ctx, _image_path: build_fallback_interpretation(
                ctx["observe"], ctx["text"], ctx["anchors"], ctx.get("temporary_context")
            ),
            sanitizer=lambda value, ctx: sanitize_interpretation_payload(value, ctx.get("temporary_context")),
        ),
        StageSpec(
            name="critique",
            system_prompt=CRITIQUE_SYSTEM,
            response_format=CritiquePack,
            config=PASS_CONFIGS["critique"],
            prompt_builder=lambda ctx, _image_path: render_critique_user(
                ctx["support"], ctx["observe"], ctx["text"], ctx["anchors"], ctx["interpretation"], ctx.get("temporary_context")
            ),
            fallback_builder=lambda ctx, _image_path: build_fallback_critique(
                ctx["interpretation"], ctx["anchors"], ctx["text"], ctx.get("temporary_context")
            ),
            sanitizer=lambda value, ctx: sanitize_critique_payload(value, ctx["interpretation"]),
            llm_enabled=lambda ctx: bool(ctx.get("llm_postprocess")),
            disabled_reason="llm_postprocess_disabled",
        ),
        StageSpec(
            name="write",
            system_prompt=WRITE_SYSTEM,
            response_format=WritePack,
            config=PASS_CONFIGS["write"],
            prompt_builder=lambda ctx, _image_path: render_write_user(
                ctx["support"], ctx["observe"], ctx["text"], ctx["anchors"], ctx["critique"], ctx.get("temporary_context")
            ),
            fallback_builder=lambda ctx, image_path: build_fallback_writing(
                image_path.name, ctx["support"], ctx["observe"], ctx["text"], ctx["critique"], ctx.get("temporary_context")
            ),
            sanitizer=lambda value, _ctx: sanitize_writing_payload(value),
            llm_enabled=lambda ctx: bool(ctx.get("llm_postprocess")),
            disabled_reason="llm_postprocess_disabled",
        ),
    )


def stage_model_for(stage_name: str, vision_model: str, reasoning_model: str) -> str:
    return vision_model if stage_name in STAGES_WITH_IMAGE else reasoning_model


def run_stage(
    spec: StageSpec,
    context: dict[str, Any],
    vision_model: str,
    reasoning_model: str,
    api_host: str,
    api_token: str | None,
    timeout_seconds: int,
    image_path: Path,
    image_base64: str,
    progress_hook: Any = None,
) -> tuple[dict[str, Any], str | None, StageRunMetrics]:
    if progress_hook:
        progress_hook(spec.name, "start")

    started_at = time.perf_counter()
    fallback_reason: str | None = None
    response_metrics: dict[str, Any] = {}
    selected_model = stage_model_for(spec.name, vision_model, reasoning_model)
    if spec.llm_enabled(context):
        try:
            value, response_metrics = call_structured(
                stage_name=spec.name,
                primary_model=selected_model,
                api_host=api_host,
                api_token=api_token,
                timeout_seconds=timeout_seconds,
                image_path=image_path,
                image_base64=image_base64,
                system_prompt=spec.system_prompt,
                user_prompt=spec.prompt_builder(context, image_path),
                response_format=spec.response_format,
                config=spec.config,
            )
            if progress_hook:
                progress_hook(spec.name, "done")
        except Exception as exc:
            value = spec.fallback_builder(context, image_path)
            fallback_reason = str(exc)
            if progress_hook:
                progress_hook(spec.name, "fallback")
    else:
        value = spec.fallback_builder(context, image_path)
        fallback_reason = spec.disabled_reason or "llm_stage_disabled"
        if progress_hook:
            progress_hook(spec.name, "fallback")

    if spec.sanitizer is not None:
        value = spec.sanitizer(value, context)
    elapsed_seconds = time.perf_counter() - started_at
    metrics = StageRunMetrics(
        elapsed_seconds=elapsed_seconds,
        used_model=selected_model if spec.llm_enabled(context) else None,
        used_fallback=fallback_reason is not None,
        response_metrics=response_metrics,
    )
    return value, fallback_reason, metrics


def stage_output_key(stage_name: str) -> str:
    return {
        "anchor": "anchors",
        "interpret": "interpretation",
        "read": "text",
        "write": "write",
    }.get(stage_name, stage_name)



def analyze_one(
    vision_model: str,
    reasoning_model: str,
    llm_postprocess: bool,
    temporary_context: str | None,
    api_host: str,
    api_token: str | None,
    timeout_seconds: int,
    image_path: Path,
    run_metadata: dict[str, Any],
    image_max_dimension: int,
    image_jpeg_quality: int,
    progress_hook: Any = None,
) -> dict[str, Any]:
    analysis_started_at = time.perf_counter()
    image_base64 = encode_analysis_image(image_path, image_max_dimension, image_jpeg_quality)
    stage_fallbacks: dict[str, str] = {}
    stage_metrics: dict[str, Any] = {}
    context: dict[str, Any] = {
        "temporary_context": temporary_context,
        "llm_postprocess": llm_postprocess,
    }

    for spec in build_stage_specs():
        value, fallback_reason, metrics = run_stage(
            spec=spec,
            context=context,
            vision_model=vision_model,
            reasoning_model=reasoning_model,
            api_host=api_host,
            api_token=api_token,
            timeout_seconds=timeout_seconds,
            image_path=image_path,
            image_base64=image_base64,
            progress_hook=progress_hook,
        )
        context[stage_output_key(spec.name)] = value
        stage_metrics[spec.name] = {
            "elapsed_seconds": round(metrics.elapsed_seconds, 3),
            "model": metrics.used_model,
            "used_fallback": metrics.used_fallback,
            "response_metrics": metrics.response_metrics,
        }
        if fallback_reason:
            stage_fallbacks[spec.name] = fallback_reason

    support = context["support"]
    observe = context["observe"]
    text = context["text"]
    anchors = context["anchors"]
    interpretation = context["interpretation"]
    critique = context["critique"]
    writing = context["write"]
    critique_fallback_reason = stage_fallbacks.get("critique")
    write_fallback_reason = stage_fallbacks.get("write")
    critique_fallback_used = critique_fallback_reason is not None
    write_fallback_used = write_fallback_reason is not None

    auto_quality = derive_auto_quality(critique)
    performance = {
        "total_elapsed_seconds": round(time.perf_counter() - analysis_started_at, 3),
        "image_bytes_base64_length": len(image_base64),
        "stage_metrics": stage_metrics,
    }

    payload = {
        "file_name": image_path.name,
        "run_metadata": run_metadata,
        "performance": performance,
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



def report_markdown(payload: dict[str, Any], vision_model: str, reasoning_model: str) -> str:
    support = payload["support"]
    observe = payload["observe"]
    text = payload["text"]
    anchors = payload["anchors"]
    interpretation = payload["interpretation"]
    critique = payload["critique"]
    writing = payload["writing"]
    quality = payload["quality"]
    auto_quality = quality.get("auto_quality", {})
    performance = payload.get("performance", {})

    lines: list[str] = []
    lines += [f"# {writing['short_title']}", ""]
    lines += [
        f"**Fichier** : `{payload['file_name']}`  ",
        f"**Modèle vision** : `{vision_model}`  ",
        f"**Modèle raisonnement** : `{reasoning_model}`",
        "",
    ]

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

    if performance:
        lines += ["## 10. Performance", ""]
        lines += [f"**Temps total** : {performance.get('total_elapsed_seconds', 'inconnu')} s  ", f"**Taille image encodée** : {performance.get('image_bytes_base64_length', 'inconnue')} caractères base64", ""]
        for stage_name, metrics in performance.get("stage_metrics", {}).items():
            lines.append(
                f"- **{stage_name}** : {metrics.get('elapsed_seconds', 'inconnu')} s | modèle : {metrics.get('model') or 'fallback local'} | fallback : {metrics.get('used_fallback', False)}"
            )

    return "\n".join(lines)


def process_single_image(
    index: int,
    total: int,
    image_path: Path,
    args: argparse.Namespace,
    run_metadata: dict[str, Any],
    stage_names: tuple[str, ...],
) -> tuple[int, dict[str, Any]]:
    stem = f"{index:03d}_{image_path.stem}"
    json_path = args.output_dir / f"{stem}.analysis.json"
    md_path = args.output_dir / f"{stem}.report.md"
    stage_bar = None
    completed_stages: set[str] = set()

    if tqdm is not None and args.workers == 1:
        stage_bar = tqdm(
            total=len(stage_names),
            desc=f"{index}/{total} {image_path.name[:32]}",
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
            args.vision_model,
            args.reasoning_model,
            args.llm_postprocess,
            args.temporary_context,
            args.api_host,
            args.api_token,
            args.sync_timeout,
            image_path,
            run_metadata,
            args.image_max_dimension,
            args.image_jpeg_quality,
            progress_hook=progress_hook,
        )
    except Exception as exc:
        payload = build_error_payload(image_path, exc, run_metadata)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        entry = {
            "file_name": image_path.name,
            "json": json_path.name,
            "markdown": None,
            "title": None,
            "reviewed": False,
            "hallucination_risk": payload["quality"]["auto_quality"]["hallucination_risk"],
            "writing_readiness": payload["quality"]["auto_quality"]["writing_readiness"],
            "vision_model": args.vision_model,
            "reasoning_model": args.reasoning_model,
            "total_elapsed_seconds": payload.get("performance", {}).get("total_elapsed_seconds"),
            "prompt_overrides": str(args.prompt_overrides) if args.prompt_overrides else None,
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        return index, entry
    finally:
        if stage_bar is not None:
            stage_bar.set_postfix_str("done")
            stage_bar.close()

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(report_markdown(payload, args.vision_model, args.reasoning_model), encoding="utf-8")
    entry = {
        "file_name": image_path.name,
        "json": json_path.name,
        "markdown": md_path.name,
        "title": payload["writing"]["short_title"],
        "reviewed": payload["quality"]["reviewed"],
        "hallucination_risk": payload["quality"]["auto_quality"]["hallucination_risk"],
        "writing_readiness": payload["quality"]["auto_quality"]["writing_readiness"],
        "vision_model": args.vision_model,
        "reasoning_model": args.reasoning_model,
        "total_elapsed_seconds": payload.get("performance", {}).get("total_elapsed_seconds"),
        "prompt_overrides": str(args.prompt_overrides) if args.prompt_overrides else None,
        "status": "ok",
        "error_type": None,
        "error_message": None,
    }
    return index, entry



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
    stage_names = tuple(spec.name for spec in build_stage_specs())
    run_metadata = {
        "backend": "ollama",
        "api_host": args.api_host,
        "api_token_configured": bool(args.api_token),
        "model": args.model,
        "vision_model": args.vision_model,
        "reasoning_model": args.reasoning_model,
        "llm_postprocess": args.llm_postprocess,
        "temporary_context": args.temporary_context,
        "sync_timeout": args.sync_timeout,
        "workers": args.workers,
        "image_max_dimension": args.image_max_dimension,
        "image_jpeg_quality": args.image_jpeg_quality,
        "pass_configs": PASS_CONFIGS,
        "prompt_overrides": str(args.prompt_overrides) if args.prompt_overrides else None,
        "prompt_fingerprints": prompt_fingerprints(),
    }
    indexed_images = list(enumerate(images, start=1))
    results: dict[int, dict[str, Any]] = {}

    try:
        if args.workers <= 1:
            for index, image_path in indexed_images:
                print(f"[{index}/{len(images)}] {image_path.name}")
                result_index, entry = process_single_image(index, len(images), image_path, args, run_metadata, stage_names)
                results[result_index] = entry
                if entry["status"] == "error":
                    print(f"  ERREUR: {entry['error_message']}")
        else:
            print(f"Traitement parallèle activé: {args.workers} workers")
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_map = {
                    executor.submit(process_single_image, index, len(images), image_path, args, run_metadata, stage_names): (index, image_path)
                    for index, image_path in indexed_images
                }
                for future in as_completed(future_map):
                    index, image_path = future_map[future]
                    print(f"[{index}/{len(images)}] {image_path.name}")
                    result_index, entry = future.result()
                    results[result_index] = entry
                    if entry["status"] == "error":
                        print(f"  ERREUR: {entry['error_message']}")
    except KeyboardInterrupt:
        print("\nInterruption utilisateur. Arrêt du traitement.")

    for index, _image_path in indexed_images:
        entry = results.get(index)
        if entry is not None:
            manifest.append(entry)

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Terminé. {len(manifest)} image(s) traitée(s). Sorties dans {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
