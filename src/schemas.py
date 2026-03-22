from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Subject(BaseModel):
    role: str = Field(description="Catégorie prudente du sujet : personne, groupe, objet, animal, document, architecture, écran, etc.")
    description: str = Field(description="Description brève et factuelle du sujet.")
    posture_or_state: str = Field(description="Posture, état ou mode de présence visible.")
    salience: str = Field(description="Pourquoi ce sujet structure la lecture de l'image.")


class TextRegion(BaseModel):
    region_label: str = Field(description="Repère simple de zone textuelle : panneau, carnet, écran, affiche, légende, etc.")
    transcription: str = Field(description="Texte recopié le plus fidèlement possible. Chaîne vide si illisible.")
    confidence: Literal["élevé", "moyen", "faible"] = Field(description="Niveau de confiance qualitatif : élevé, moyen, faible.")
    notes: str = Field(description="Notes de lecture, coupures, langue supposée, parties manquantes.")


class EvidenceAnchor(BaseModel):
    anchor_id: str = Field(description="Identifiant court et stable du type A1, A2, A3.")
    observation: str = Field(description="Observation concrète ancrée dans le visible ou le lisible.")
    supports: list[str] = Field(description="Ce que cette observation permet raisonnablement de soutenir.")
    certainty: Literal["forte", "moyenne", "faible"] = Field(description="Force de l'ancrage : forte, moyenne, faible.")
    anti_overreach: str = Field(description="Ce qu'il ne faut pas déduire à partir de cet ancrage seul.")


class EmotionHypothesis(BaseModel):
    emotion: str = Field(description="Émotion, climat ou dynamique affective probable.")
    anchor_refs: list[str] = Field(description="Liste d'ancrages mobilisés, par exemple ['A1', 'A3'].")
    confidence: Literal["haute", "moyenne", "basse"] = Field(description="Confiance qualitative : haute, moyenne, basse.")
    reason: str = Field(description="Justification explicite reliant l'hypothèse aux ancrages.")


class SupportProfile(BaseModel):
    support_kind: str = Field(description="Nature du support : photo, document, capture d'écran, illustration, page, objet, mixte, ou incertain.")
    primary_focus: str = Field(description="Ce que l'analyse doit regarder en premier : scène, texte, objet, document, composition, ou mixte.")
    depicts_direct_scene: str = Field(description="Indique si l'image montre directement une scène du monde ou surtout un support qui représente autre chose.")
    support_boundary: str = Field(description="Distinction entre ce qui est physiquement dans le cadre et ce qui est simplement représenté sur un support visible.")
    reasoning_guardrails: list[str] = Field(description="Garde-fous pour éviter les mauvaises inférences sur ce type de support.")


class SceneScan(BaseModel):
    scene_summary: str = Field(description="Résumé factuel de la scène ou de la page visible en 2 à 4 phrases.")
    setting: str = Field(description="Environnement, lieu ou support visible, ou mention explicite qu'il reste indéterminé.")
    subjects: list[Subject] = Field(description="Sujets principaux présents dans l'image.")
    salient_objects: list[str] = Field(description="Objets ou détails saillants utiles à la lecture.")
    visible_actions: list[str] = Field(description="Actions visibles ou états observables, formulés prudemment.")
    spatial_relations: list[str] = Field(description="Relations de distance, hiérarchie, orientation, isolement ou regroupement.")
    composition: str = Field(description="Cadrage, hiérarchie visuelle, profondeur, découpage des masses.")
    lighting_and_color: str = Field(description="Lumière, contraste, couleurs, texture et densité visuelle.")
    uncertainties: list[str] = Field(description="Zones ambiguës ou invisibles qu'il ne faut pas sur-affirmer.")


class TextScan(BaseModel):
    has_text: Literal["oui", "non", "incertain"] = Field(description="Oui, non, ou incertain.")
    text_regions: list[TextRegion] = Field(description="Zones textuelles identifiées dans l'image.")
    combined_text: str = Field(description="Texte visible global, fusionné proprement. Chaîne vide si aucun texte lisible.")
    language_guesses: list[str] = Field(description="Langues probables du texte visible.")
    text_role: str = Field(description="Fonction probable du texte : légende, note intime, signalétique, écran d'interface, titre, consigne, etc.")
    reading_limits: list[str] = Field(description="Limites d'OCR, parties coupées, flou, angle, résolution ou ambiguïtés.")


class AnchorMap(BaseModel):
    dominant_axes: list[str] = Field(description="Axes forts de lecture : regard, posture, isolement, densité textuelle, frontalité, vide, lumière, etc.")
    anchors: list[EvidenceAnchor] = Field(description="Ancrages probatoires organisant l'interprétation.")
    safe_inferences: list[str] = Field(description="Inférences raisonnables autorisées par les ancrages.")
    open_questions: list[str] = Field(description="Questions ou ambiguïtés qui restent ouvertes.")
    do_not_claim: list[str] = Field(description="Conclusions interdites ou trop spéculatives à ce stade.")


class InterpretationPack(BaseModel):
    core_reading: str = Field(description="Lecture centrale prudente et argumentée de l'image.")
    social_dynamics: str = Field(description="Rapports humains, distances, présences, absences ou neutralité si rien ne s'impose.")
    emotional_hypotheses: list[EmotionHypothesis] = Field(description="Hypothèses émotionnelles reliées explicitement aux ancrages.")
    text_scene_interaction: str = Field(description="Rôle du texte visible dans la lecture de la scène, ou mention qu'il est secondaire / absent.")
    alternative_readings: list[str] = Field(description="Lectures alternatives plausibles à conserver.")
    prohibited_conclusions: list[str] = Field(description="Conclusions séduisantes mais non justifiables par l'image seule.")
    residual_uncertainty: str = Field(description="Ce qui reste incertain même après interprétation.")


class CritiqueIssue(BaseModel):
    issue_type: str = Field(description="Type de problème : surinterprétation, contradiction, émotion non ancrée, OCR sur-utilisé, contexte inventé, style trop affirmatif, etc.")
    severity: Literal["faible", "moyenne", "élevée"] = Field(description="Gravité qualitative : faible, moyenne, élevée.")
    location: str = Field(description="Zone touchée : core_reading, social_dynamics, emotional_hypotheses[1], etc.")
    explanation: str = Field(description="Pourquoi cet élément pose problème au regard du visible, du texte ou des ancrages.")
    suggested_fix: str = Field(description="Correction concrète et minimale à appliquer.")


class CritiquePack(BaseModel):
    faithfulness_score: int = Field(ge=0, le=100, description="Score de fidélité au visible de 0 à 100.")
    overreach_risk_score: int = Field(ge=0, le=100, description="Risque de surinterprétation de 0 à 100.")
    evidence_coverage_score: int = Field(ge=0, le=100, description="Couverture par les ancrages de 0 à 100.")
    writing_readiness: Literal["prêt à rédiger", "à réviser", "à bloquer"] = Field(description="Prêt à rédiger, à réviser, ou à bloquer.")
    global_assessment: str = Field(description="Diagnostic bref sur la qualité de l'interprétation brute.")
    issues: list[CritiqueIssue] = Field(description="Liste détaillée des problèmes repérés.")
    keep_as_is: list[str] = Field(description="Éléments suffisamment bons à conserver tels quels.")
    revision_priorities: list[str] = Field(description="Ordre conseillé des corrections à faire avant rédaction finale.")
    revised_core_reading: str = Field(description="Version révisée et plus fidèle de la lecture centrale.")
    revised_social_dynamics: str = Field(description="Version révisée et plus fidèle des dynamiques humaines.")
    revised_emotional_hypotheses: list[EmotionHypothesis] = Field(description="Hypothèses émotionnelles révisées après audit.")
    revised_text_scene_interaction: str = Field(description="Version révisée du rôle du texte dans la lecture de la scène.")
    revised_alternative_readings: list[str] = Field(description="Lectures alternatives nettoyées après critique.")
    revised_prohibited_conclusions: list[str] = Field(description="Conclusions interdites reformulées ou complétées après audit.")
    revised_residual_uncertainty: str = Field(description="Incertitude finale après révision.")


class WritePack(BaseModel):
    short_title: str = Field(description="Titre court, sobre et réutilisable.")
    analytic_brief: str = Field(description="Résumé analytique de 80 à 120 mots, net et réutilisable.")
    human_commentary: str = Field(description="Interprétation humaine de 120 à 180 mots, sensible mais disciplinée.")
    photographic_commentary: str = Field(description="Commentaire photographique de 100 à 160 mots, centré sur forme, lisibilité, rythme, cadre et matière.")
    literary_commentary: str = Field(description="Commentaire littéraire de 100 à 160 mots, fidèle au visible et sans fiction hors champ.")
    keywords: list[str] = Field(description="5 à 10 mots-clés précis et réutilisables.")
