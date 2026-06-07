"""
Meld Coach Engine, Research-Informed AI Architecture

Implements 7 architectural principles from the AI research corpus:
1. Evidence-bound coaching (EviBound: 0% hallucination)
2. Deliberation-first routing (DOVA: 40-60% cost reduction)
3. Hierarchical memory (MAPLE/Memoria/HiMem)
4. Knowledge graph for cross-domain connections (GAAMA)
5. Multi-agent sub-tasks (memory, analysis, coaching)
6. Safety routing for concerning data (Health-ORSC-Bench, ILION)
7. Explainable model routing (Topaz)
"""

import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import anthropic
from app.config import settings
from app.core.time import utcnow_naive

# Signal Engine Phase 5 context. Imported via TYPE_CHECKING so the boundary
# rule in ml.api stays honored and ``process_query`` can keep its sync
# signature while the router pre-loads SignalContext asynchronously.
if TYPE_CHECKING:
    from ml.api import SignalContext

logger = logging.getLogger("meld.coach")


# ============================================================
# 7. EXPLAINABLE MODEL ROUTING (Topaz)
# Every routing decision is logged with reasoning.
# ============================================================

class ModelTier(Enum):
    RULES = "rules"          # No AI needed, deterministic answer
    HAIKU = "haiku"          # Simple formatting/summarization
    SONNET = "sonnet"        # Routine coaching, daily insights
    OPUS = "opus"            # Deep analysis, safety-critical, cross-domain


@dataclass
class RoutingDecision:
    tier: ModelTier
    reason: str
    confidence: float  # 0-1
    safety_flag: bool = False
    timestamp: str = field(default_factory=lambda: utcnow_naive().isoformat())

    def to_dict(self) -> dict:
        return {
            "tier": self.tier.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "safety_flag": self.safety_flag,
            "timestamp": self.timestamp,
        }


# ============================================================
# 6. SAFETY ROUTING (Health-ORSC-Bench, ILION, MHDash)
# Pre-execution safety gates for concerning health data.
# ============================================================

@dataclass
class SafetyCheck:
    is_concerning: bool
    reasons: list[str]
    requires_disclaimer: bool
    requires_opus: bool

    @staticmethod
    def check_health_data(data: dict) -> "SafetyCheck":
        """Deterministic safety gates applied BEFORE any AI processing."""
        reasons = []
        concerning = False

        # HRV critically low (< 20ms suggests autonomic dysfunction)
        hrv = data.get("hrv_average")
        if hrv is not None and hrv < 20:
            reasons.append(f"HRV critically low ({hrv}ms)")
            concerning = True

        # Resting HR elevated (> 100 bpm at rest is tachycardia)
        rhr = data.get("resting_hr")
        if rhr is not None and rhr > 100:
            reasons.append(f"Resting HR elevated ({rhr} bpm)")
            concerning = True

        # Sleep efficiency critically low (< 50%)
        efficiency = data.get("sleep_efficiency")
        if efficiency is not None and efficiency < 50:
            reasons.append(f"Sleep efficiency very low ({efficiency}%)")
            concerning = True

        # Readiness critically low
        readiness = data.get("readiness_score")
        if readiness is not None and readiness < 20:
            reasons.append(f"Readiness critically low ({readiness})")
            concerning = True

        # Sudden large changes (> 30% deviation from baseline)
        # Skip when baseline_days < 3, too little history for meaningful comparison
        baseline_hrv = data.get("baseline_hrv")
        baseline_days = data.get("baseline_days", 0)
        if (hrv is not None and baseline_hrv is not None and baseline_hrv > 0
                and baseline_days >= 3
                and abs(hrv - baseline_hrv) / baseline_hrv > 0.3):
            reasons.append(f"HRV deviated {abs(hrv - baseline_hrv):.0f}ms from baseline")
            concerning = True

        return SafetyCheck(
            is_concerning=concerning,
            reasons=reasons,
            requires_disclaimer=concerning,
            requires_opus=concerning,
        )

    @staticmethod
    def check_message_content(text: str) -> "SafetyCheck":
        """Detect potential crisis language in user messages.

        Runs BEFORE AI processing. If crisis language is found, forces
        Opus routing for deeper, more careful reasoning.
        """
        crisis_phrases = [
            "want to die", "want to end it", "kill myself", "end my life",
            "no reason to live", "better off without me",
            "can't go on", "cant go on",
            "hurt myself", "self-harm", "self harm", "suicide", "suicidal",
            "don't want to be here", "dont want to be here",
            "not worth living",
            "no point in living", "want it to be over",
            "feeling like a burden", "everyone would be better off",
            "don't want to live", "dont want to live",
            "can't take it anymore", "cant take it anymore",
            "i don't want to be alive", "i dont want to be alive",
        ]
        text_lower = text.lower()
        reasons = []
        for phrase in crisis_phrases:
            if phrase in text_lower:
                reasons.append(f"Crisis language detected: '{phrase}'")

        is_crisis = len(reasons) > 0
        return SafetyCheck(
            is_concerning=is_crisis,
            reasons=reasons,
            # Don't add health-metric disclaimer for emotional crisis ,
            # the prompt's Rule 8 handles the response tone.
            requires_disclaimer=False,
            requires_opus=is_crisis,
        )


# ============================================================
# 3. HIERARCHICAL MEMORY (deferred, see UserCorrelation for now)
# ============================================================
#
# UserMemory was a TODO scaffold for per-user pattern memory. Removed in
# P2-7 cleanup because (1) nothing populated it, (2) it had no persistence,
# and (3) the correlation engine already discovers patterns and stores them
# in the UserCorrelation table. When per-user memory ships for real, it
# should pull from UserCorrelation + ChatMessageRecord, not be reinvented
# from scratch.


# ============================================================
# 4. KNOWLEDGE GRAPH, DELETED in Phase 5 (Signal Engine).
#
# The hardcoded KnowledgeGraph seeded with three literature connections was a
# Phase 0 placeholder for "personalized cross-domain relationships". It has
# been replaced by:
#
#   - active_patterns  : load_active_patterns() reads UserCorrelation at
#                        developing+ tier (populated by Phase 3 L2 engine)
#   - recent_anomalies : ml_anomalies rows from the last 7 days (Phase 2)
#   - personal_forecast: ml_forecasts rows for today/tomorrow (Phase 2)
#
# All three are assembled into a ``SignalContext`` by ``ml.api`` and passed
# into ``CoachEngine.process_query`` as an optional parameter. The prompt
# template renders them dynamically.
#
# See ~/.claude/plans/golden-floating-creek.md Phase 5 for the full spec.
# ============================================================


# ============================================================
# 2. DELIBERATION-FIRST ROUTING (DOVA, The Plausibility Trap)
# Check DB/rules before calling AI. 40-60% cost reduction.
# ============================================================

class Deliberator:
    """Decides whether a query needs AI or can be answered from rules/data."""

    # Deterministic rules that don't need AI
    RULES = {
        "readiness_high": "Your readiness is high. Today is good for a hard workout.",
        "readiness_moderate": "Your readiness is moderate. Keep it easy today.",
        "readiness_low": "Your body needs rest. Take it easy today.",
        "hrv_above_baseline": "Your HRV is above your average. Your body is handling stress well.",
        "hrv_below_baseline": "Your HRV is below your average. Your body may need more recovery.",
        "rhr_stable": "Your resting heart rate is holding steady. That means your fitness level is consistent right now.",
        "rhr_dropping": "Your resting heart rate is trending down. That's a good sign: your cardiovascular fitness is improving.",
        "rhr_rising": "Your resting heart rate is trending up. This can mean stress, poor sleep, or overtraining. Give your body some extra recovery time.",
    }

    @staticmethod
    def can_answer_from_rules(query: str, health_data: dict) -> tuple[bool, str | None]:
        """Check if the query can be answered deterministically."""

        query_lower = query.lower()

        # Resting heart rate, HRV, sleep, these need AI with real data, not canned rules.
        # The user wants their actual numbers with context, not generic advice.
        if any(w in query_lower for w in ["resting heart rate", "resting hr", "rhr", "heart rate"]):
            return False, None  # Route to AI with full health data context

        # Readiness queries (use full words to avoid matching "resting")
        from app.core.constants import ReadinessThreshold
        readiness = health_data.get("readiness_score", 0)
        if any(w in query_lower for w in ["readiness", "recovery", "ready", "push hard", "should i rest"]):
            if readiness >= ReadinessThreshold.HIGH:
                return True, Deliberator.RULES["readiness_high"]
            elif readiness >= ReadinessThreshold.MODERATE:
                return True, Deliberator.RULES["readiness_moderate"]
            else:
                return True, Deliberator.RULES["readiness_low"]

        # HRV queries with clear baseline comparison
        # Only compare against baseline when we have >= 3 days of history
        hrv = health_data.get("hrv_average")
        baseline_hrv = health_data.get("baseline_hrv")
        baseline_days = health_data.get("baseline_days", 0)
        if (hrv is not None and baseline_hrv is not None and baseline_days >= 3
                and any(w in query_lower for w in ["hrv", "heart rate variability"])):
            if hrv > baseline_hrv * 1.05:
                return True, Deliberator.RULES["hrv_above_baseline"]
            elif hrv < baseline_hrv * 0.95:
                return True, Deliberator.RULES["hrv_below_baseline"]

        # Can't answer from rules, needs AI
        return False, None

    @staticmethod
    def route(query: str, health_data: dict, safety: SafetyCheck) -> RoutingDecision:
        """Determine the optimal model tier for this query.

        Priority order:
        1. Safety-critical → Opus (non-negotiable)
        2. Cross-domain / deep reasoning → Opus (needs synthesis)
        3. Simple factual → Rules (no AI cost)
        4. Everything else → Sonnet (routine coaching)
        """

        # 1. Safety-critical → always Opus
        if safety.requires_opus:
            return RoutingDecision(
                tier=ModelTier.OPUS,
                reason=f"Safety flag: {', '.join(safety.reasons)}",
                confidence=1.0,
                safety_flag=True,
            )

        # 2. Complex cross-domain queries → Opus (check BEFORE rules)
        query_lower = query.lower()
        # P3-8: Use word-boundary regex instead of substring matching.
        # Substring matching had two bugs:
        #   - "cause" doesn't match "causing" (missing stem coverage)
        #   - "caus" stem matches "because" (false positive → Opus cost leak)
        # Word boundaries with explicit inflections fix both at once.
        import re as _re
        cross_domain_pattern = _re.compile(
            r"\b(why|connect|connects|connected|pattern|patterns|"
            r"correlation|correlate|correlates|"
            r"cause|causes|causing|caused|"
            r"affect|affects|affected|effect|effects|"
            r"relationship|related|"
            r"explain|explains)\b",
            _re.IGNORECASE,
        )
        if cross_domain_pattern.search(query_lower):
            return RoutingDecision(
                tier=ModelTier.OPUS,
                reason="Cross-domain analysis requires deep reasoning",
                confidence=0.8,
            )

        # 3. Can answer from rules → no AI needed
        can_rule, _ = Deliberator.can_answer_from_rules(query, health_data)
        if can_rule:
            return RoutingDecision(
                tier=ModelTier.RULES,
                reason="Deterministic rule match",
                confidence=1.0,
            )

        # 4. Everything else → Sonnet
        return RoutingDecision(
            tier=ModelTier.SONNET,
            reason="Routine coaching query",
            confidence=0.9,
        )


# ============================================================
# 1. EVIDENCE-BOUND COACHING (EviBound: 0% hallucination)
# Every claim must cite specific data points.
# ============================================================

EVIDENCE_BOUND_SYSTEM_PROMPT = """You are a personal health coach for {user_name}. You provide evidence-based coaching.

CRITICAL RULES:
1. EVERY claim you make MUST reference a specific data point from the user's data below.
   - GOOD: "Your sleep efficiency was 91% last night, which is above your 7-day average of 85%."
   - BAD: "You had a great night of sleep." (no specific data cited)

2. If you don't have data to support a claim, say "I don't have enough data for that yet." But if the user's question can be answered with general evidence-based knowledge (like supplement questions, nutrition basics, or workout principles), answer it directly. Only decline when the question specifically requires their personal data to answer.

3. NEVER make up numbers. Only use the exact values provided below, INCLUDING decimal precision (cite "62.3" as "62.3", not "62"). If a prior message in this conversation cited a different number than the current data, TRUST THE CURRENT DATA. It is the ground truth, not the chat history. Numbers change day to day as new data syncs. Do not anchor on previous turns.

4. Write at a 4th grade reading level. MANDATORY RULES, follow every one:
   - Count words in EVERY sentence before writing it. 10 words max. If over 10, split it.
   - One idea per sentence. Never join two ideas with "and", "but", "which", "since", or "because".
   - Replace jargon with plain words. "How well you slept" not "sleep efficiency". "Heart calm score" not "HRV". "Body energy score" not "readiness score".
   - Default to 2 to 4 short paragraphs. When you have 3 or more parallel items (workout options, meal ideas, individual metrics), put them in a bulleted list instead of prose.

   EXAMPLE for workout questions:
   BAD (too long, jargon, compound): "Your readiness score is 65, which is moderate, and your sleep efficiency was 72%, so you can work out today but not too hard."
   GOOD (short, plain, one idea each): "Your body energy score is 65. That is in the middle. You slept okay last night. You can work out today. Keep it at medium effort."

5. Be warm but direct. Use the user's name at least once when it adds warmth, typically in the opening verdict or closing action (e.g., "Nice work, Marcus. Your sleep efficiency was..."). Do not sprinkle it into every sentence. Tell them what the data means and what to do. When recommending workouts, give specific examples (e.g., "30-minute walk" or "upper body strength training"). When explaining causes, use the data to suggest likely factors even if lifestyle details aren't explicitly provided.

6. You are a wellness coach, NOT a doctor. Never diagnose conditions or suggest the user is sick. Instead, describe what the DATA shows (elevated, below baseline) and recommend they talk to a doctor if concerned.

7. USE MARKDOWN for scannability, but lightly:
   - **bold** for the key numbers and the bottom-line verdict (e.g. "**91%** sleep efficiency")
   - Bulleted lists with a leading hyphen when listing 3+ items (the client renders these with a proper bullet glyph)
   - Blank lines between paragraphs
   Do NOT use headers (# or ##), tables, code blocks, or nested lists. Keep formatting simple and inline.

8. STRUCTURE longer responses using BLUF (bottom line up front):
   - One short verdict line first (the answer, not the analysis). Bold the verdict.
   - 2 to 4 bullets with the supporting data, each leading with the bolded number.
   - One short closing line with the recommended action.
   Short answers (1 to 3 sentences) don't need this structure. Use it when you're citing multiple data points.

9. RICH DATA CALLOUTS: when a specific metric is the headline fact (sleep efficiency, HRV, deep sleep, steps, readiness, RHR), wrap it in a data-card tag INSTEAD of writing it out inline. The client renders these tags as visual cards.
   Syntax: [[data:METRIC_KEY:VALUE:UNIT:SUBTITLE]]
   - METRIC_KEY: snake_case from health data (sleep_efficiency, hrv, deep_sleep_minutes, resting_hr, readiness_score, steps, active_calories)
   - VALUE: the numeric value exactly as given in the data
   - UNIT: short unit string (%, bpm, ms, min, steps, cal, or empty)
   - SUBTITLE: one short phrase with context (e.g. "above 7-day avg of 85%", "below baseline of 45ms")
   Example: "[[data:sleep_efficiency:91:%:above 7-day avg of 85%]]"
   Use 1 to 2 data tags per response at most. The remaining metrics can stay as bolded inline text.

10. NEVER use em dashes (—, U+2014). Use commas, colons, parentheses, or a new sentence instead. Hyphens in compound words ("7-day average", "4th-grade") are fine. This applies everywhere in your response.

11. EMOTIONAL DISTRESS HANDLING (two scenarios, treat them differently):
    (a) NON-CRISIS DISTRESS (anxiety, hopelessness, "not wanting to get out of bed", "tired of everything"): validate feelings FIRST, then offer 1-2 immediate calming techniques (deep breathing, grounding exercise), then briefly discuss data.
    (b) MENTAL HEALTH CRISIS (self-harm, suicidal thoughts, wanting to die, "feeling like a burden", "no reason to live", "nothing matters", "everyone would be better off without me"): your response MUST be EXCLUSIVELY about safety and human connection. Order: validation first (acknowledge the pain, affirm their worth, "I hear you", "you matter"); then crisis resources verbatim (988 Suicide & Crisis Lifeline call or text 988, Crisis Text Line text HOME to 741741); then encouragement to reach out to a mental health professional or trusted person. Do NOT include ANY health metrics, sleep data, HRV, steps, patterns, exercises, calming techniques, body-state observations ("your body is under stress"), or coaching content in a crisis response. The response is short, focused, contains zero data references.

12. For requests outside your scope (detailed meal plans, specific exercise programs, financial advice), acknowledge the limit and recommend the appropriate professional (dietitian, trainer, financial advisor).

13. For any extreme dietary restriction (under 1200 cal/day), always flag it as potentially dangerous and recommend they consult a doctor or registered dietitian before making changes.

14. NEVER recommend specific supplement dosages, brands, or protocols. For supplement questions (melatonin, magnesium, creatine, etc.): FIRST list 2 to 3 specific behavioral alternatives (for sleep supplements: consistent bedtime, cool/dark bedroom, no screens before bed). THEN briefly note the supplement and defer to a doctor. Do not lead with the supplement, do not explain its mechanism in detail. Behavioral interventions are the headline; the supplement note is a one-liner.

15. AVOID CAUSAL LANGUAGE for patterns and correlations. The coach reports correlation, not causation. Forbidden verbs when describing relationships in the data: "helping", "helps", "causes", "caused", "making", "improves", "boosts", "fixes". Use instead: "is associated with", "tends to correlate with", "on days when X is higher, Y also tends to be higher", "in your trend". This rule applies even for established-tier or literature-supported patterns.

16. MULTI-FACTOR CONNECTIONS. When a user asks about progress on weight, sleep, or fitness goals, connect ALL relevant signals in their data, not just one. If the data has weight + activity + sleep + nutrition, address each in the response. Make the goal connection prominent, not an afterthought tacked on at the end.

17. CITE EVIDENCE-BASED CLAIMS. When discussing well-established physiological relationships (exercise and HRV, sleep and recovery, protein and muscle synthesis, walking and glucose), cite the source: "research shows", "studies support", "the WHO recommends", "the Oura readiness whitepaper notes", or named guidelines (CDC, ACSM). For active_patterns marked literature_supported, cite the literature_ref string when present.

18. STEP GOALS USE ESTABLISHED BENCHMARKS. When recommending step targets, use only well-known goals: 7,000 (Lancet study minimum for mortality benefit), 8,000 (CDC moderate-activity guideline), or 10,000 (popular convention). Do not improvise intermediate targets like 6,000 or 9,000.

{safety_disclaimer}

USER'S HEALTH DATA (cite these values):
{health_data}

USER'S GOALS: {goals}

{custom_goal_context}
{memory_context}

{active_patterns}

{recent_anomalies}

{personal_forecast}
"""

def _prettify(metric: str) -> str:
    """Turn a snake_case feature key into human-readable prose."""
    return metric.replace("_", " ")


def _render_active_patterns(ctx: "SignalContext | None") -> str:
    """Render the Phase 5 ACTIVE PATTERNS prompt section.

    Renders nothing when the context is missing or empty, so the prompt
    template does not emit a header with no content underneath.
    """
    if ctx is None or not ctx.active_patterns:
        return ""
    lines: list[str] = ["ACTIVE PATTERNS (cite when relevant):"]
    for p in ctx.active_patterns:
        if p.effect_description:
            sentence = p.effect_description
        else:
            direction_word = "higher too" if p.direction == "positive" else "lower"
            sentence = (
                f"When {_prettify(p.source_metric)} is higher, "
                f"{_prettify(p.target_metric)} tends to be {direction_word}."
            )
        tier_label = p.confidence_tier.replace("_", " ")
        lines.append(f"- {sentence} (tier: {tier_label}, n={p.sample_size})")
        if p.literature_ref:
            lines.append(f"  Source: {p.literature_ref}")
    return "\n".join(lines)


def _render_recent_anomalies(ctx: "SignalContext | None") -> str:
    """Render the Phase 5 RECENT ANOMALIES prompt section."""
    if ctx is None or not ctx.recent_anomalies:
        return ""
    lines: list[str] = ["RECENT ANOMALIES (last 7 days, BOCPD-confirmed):"]
    for a in ctx.recent_anomalies:
        observed = f"{a.observed_value:.1f}" if a.observed_value is not None else "n/a"
        forecasted = f"{a.forecasted_value:.1f}" if a.forecasted_value is not None else "n/a"
        lines.append(
            f"- {_prettify(a.metric_key)} on {a.observation_date}: "
            f"observed {observed} vs forecast {forecasted} "
            f"(z={a.z_score:.1f}, {a.direction})"
        )
    return "\n".join(lines)


def _render_personal_forecast(ctx: "SignalContext | None") -> str:
    """Render the Phase 5 PERSONAL FORECAST prompt section."""
    if ctx is None or not ctx.personal_forecasts:
        return ""
    lines: list[str] = ["PERSONAL FORECAST (today and tomorrow):"]
    for f in ctx.personal_forecasts:
        if f.y_hat is None:
            continue
        if f.y_hat_low is not None and f.y_hat_high is not None:
            interval = f" (95% interval {f.y_hat_low:.1f} to {f.y_hat_high:.1f})"
        else:
            interval = ""
        lines.append(
            f"- {_prettify(f.metric_key)} on {f.target_date}: {f.y_hat:.1f}{interval}"
        )
    # If every forecast had y_hat=None, the loop adds nothing beyond the header.
    # Drop the header in that case so we do not render a stub section.
    return "\n".join(lines) if len(lines) > 1 else ""


SAFETY_DISCLAIMER = """
IMPORTANT: Some of this user's health metrics are outside normal ranges.
Concerning values: {concerns}
Include a note suggesting they consult their healthcare provider if these patterns persist.
Do NOT diagnose or alarm. Gently suggest professional follow-up.
"""


# ============================================================
# 5. MULTI-AGENT PIPELINE (MAPLE, DOVA)
# Memory → Analysis → Coaching, each as a distinct step.
# ============================================================

class CoachEngine:
    """Research-informed AI coaching engine.

    CRITICAL: instantiated via FastAPI dependency injection in routers
    (coach.py, insights.py). Static call graphs do not model `Depends(...)`
    as a CALLS edge to `__init__`, so impact tools may report this class
    as low-impact. In reality, any change here is Tier 2 product surface
    and gates user-visible coach output.
    """

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def process_query(
        self,
        query: str,
        health_data: dict,
        user_name: str = "Brock",
        user_goals: list[str] | None = None,
        custom_goal_text: str | None = None,
        history: list[dict] | None = None,
        signal_context: "SignalContext | None" = None,
    ) -> dict:
        """Full multi-agent pipeline for processing a coaching query.

        Returns dict with: response, routing_decision, safety_check, evidence_citations
        """

        # Step 1: Safety check (ILION: deterministic pre-execution gates)
        health_safety = SafetyCheck.check_health_data(health_data)
        message_safety = SafetyCheck.check_message_content(query)

        # Merge: either concerning health data OR crisis language → flag
        safety = SafetyCheck(
            is_concerning=health_safety.is_concerning or message_safety.is_concerning,
            reasons=health_safety.reasons + message_safety.reasons,
            requires_disclaimer=health_safety.requires_disclaimer,
            requires_opus=health_safety.requires_opus or message_safety.requires_opus,
        )
        logger.info(f"Safety check: concerning={safety.is_concerning}, reasons={safety.reasons}")

        # Step 2: Deliberation-first routing (DOVA: decide before computing)
        routing = Deliberator.route(query, health_data, safety)
        logger.info(f"Routing: tier={routing.tier.value}, reason={routing.reason}")

        # Step 3: If rules can answer, return immediately (no AI cost)
        if routing.tier == ModelTier.RULES:
            _, rule_answer = Deliberator.can_answer_from_rules(query, health_data)
            return {
                "response": rule_answer,
                "routing": routing.to_dict(),
                "safety": {"is_concerning": False},
                "evidence_citations": [],
                "model_used": "rules",
            }

        # Step 4: Build evidence-bound prompt with Signal Engine context.
        # (Memory context removed in P2-7 cleanup, was always empty.)
        memory_context = ""

        # Phase 5: signal_context (active patterns, recent anomalies, personal
        # forecast) is loaded async by the caller and passed in. When None or
        # empty, the prompt simply omits those sections rather than render a
        # placeholder, so non-router callers (notifications, tests) stay
        # uncluttered.
        active_patterns_section = _render_active_patterns(signal_context)
        recent_anomalies_section = _render_recent_anomalies(signal_context)
        personal_forecast_section = _render_personal_forecast(signal_context)

        safety_disclaimer = ""
        if safety.requires_disclaimer:
            safety_disclaimer = SAFETY_DISCLAIMER.format(
                concerns=", ".join(safety.reasons)
            )

        # Free-form onboarding goal text from the Goals step. Feeds the coach
        # the user's actual situation in their own words so responses can speak
        # to "I want to lose weight for my wedding" rather than only the chip
        # set. Omitted cleanly when the user left the field blank.
        custom_goal_trimmed = (custom_goal_text or "").strip()
        custom_goal_context = (
            f"WHAT THE USER TOLD US IN THEIR OWN WORDS:\n{custom_goal_trimmed}\n"
            if custom_goal_trimmed else ""
        )

        # Privacy (audit B3): do NOT send the user's real first name to the
        # model. The privacy policy states the name is not shared with the LLM
        # provider unless necessary, and it is not necessary for coaching. Use a
        # generic placeholder; the prompt constant (and eval parity) are
        # unchanged, and the `user_name` argument is retained for API
        # compatibility but intentionally not forwarded to the model.
        system_prompt = EVIDENCE_BOUND_SYSTEM_PROMPT.format(
            user_name="the user",
            health_data=json.dumps(health_data, indent=2),
            goals=", ".join(user_goals or ["general wellness"]),
            custom_goal_context=custom_goal_context,
            memory_context=f"USER PREFERENCES (learned over time):\n{memory_context}" if memory_context else "",
            active_patterns=active_patterns_section,
            recent_anomalies=recent_anomalies_section,
            personal_forecast=personal_forecast_section,
            safety_disclaimer=safety_disclaimer,
        )

        # Step 5: Call the appropriate model tier
        model = self._get_model(routing.tier)

        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": query})

        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=500,
                system=system_prompt,
                messages=messages,
            )
            response_text = response.content[0].text if response.content else "I'm having trouble right now. Please try again."
        except anthropic.APIError as e:
            # Covers anthropic.APIConnectionError, APITimeoutError, APIStatusError,
            # RateLimitError, AuthenticationError, and BadRequestError.
            logger.error("Claude API call failed: %s", e)

            # P1 safety: if the user sent crisis language but the API is down,
            # we MUST still surface crisis resources, not a generic retry message.
            if message_safety.is_concerning:
                fallback_text = (
                    "I'm having trouble connecting right now, but I want to make sure you're safe.\n\n"
                    "If you're in crisis, please reach out:\n"
                    "988 Suicide & Crisis Lifeline: call or text 988\n"
                    "Crisis Text Line: text HOME to 741741\n"
                    "If you're in immediate danger, call 911.\n\n"
                    "You're not alone. Please talk to someone."
                )
            else:
                fallback_text = "I'm having trouble connecting right now. Please try again in a moment."

            return {
                "response": fallback_text,
                "routing": routing.to_dict(),
                # Preserve the real safety check, data can still be concerning
                # even when the Claude call fails. Monitoring depends on this.
                "safety": {
                    "is_concerning": safety.is_concerning,
                    "reasons": safety.reasons,
                    "disclaimer_included": safety.requires_disclaimer,
                },
                "model_used": model,
                "tokens": {"input": 0, "output": 0},
            }

        # Step 6: Log for explainability (Topaz)
        logger.info(f"Model used: {model}, tokens: {response.usage.input_tokens}+{response.usage.output_tokens}")

        return {
            "response": response_text,
            "routing": routing.to_dict(),
            "safety": {
                "is_concerning": safety.is_concerning,
                "reasons": safety.reasons,
                "disclaimer_included": safety.requires_disclaimer,
            },
            "model_used": model,
            "tokens": {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            },
        }

    def generate_daily_insight(self, health_data: dict, user_goals: list[str] | None = None) -> dict:
        """Generate the daily coaching insight for the dashboard."""
        return self.process_query(
            query="Generate a brief daily coaching insight based on my latest health data. Be specific about the numbers.",
            health_data=health_data,
            user_goals=user_goals,
        )

    def _get_model(self, tier: ModelTier) -> str:
        """Map tier to Anthropic model ID.

        IDs centralized in `Settings` (PR #95) so a model deprecation is
        one Railway env-var override, not a grep across services.
        """
        return {
            ModelTier.HAIKU: settings.anthropic_model_haiku,
            ModelTier.SONNET: settings.anthropic_model_sonnet,
            ModelTier.OPUS: settings.anthropic_model_opus,
        }.get(tier, settings.anthropic_model_sonnet)
