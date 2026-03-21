"""
Gemini AI chat endpoint - conversational EMI shielding design assistant.

Accepts a user message and conversation history, enriches the request with
optional simulation context, and returns a Gemini-generated response along
with suggested follow-up actions.
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

from core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert electromagnetic interference shielding engineer and materials scientist. "
    "You help users design EMI shields by recommending materials, layer configurations, and "
    "processing parameters. You have access to a physics simulation engine that can calculate "
    "shielding effectiveness using Schelkunoff theory, Transfer Matrix Method for multilayer "
    "shields, percolation theory for composites, and Snoek's limit for magnetic materials. "
    "When users describe their shielding needs, extract the key parameters (frequency range, "
    "target SE, weight/thickness constraints, environment) and provide specific actionable "
    "recommendations with numerical values. Always explain the physics behind your recommendations."
)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class HistoryMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant' / 'model'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    history: List[HistoryMessage] = Field(
        default_factory=list,
        description="Prior conversation turns for multi-turn context",
    )
    # Optional simulation context forwarded by the frontend
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Current simulation state (composition, frequency, results, …)",
    )


class ChatResponse(BaseModel):
    response: str
    suggestions: List[str]


# ---------------------------------------------------------------------------
# Suggestion extractor
# ---------------------------------------------------------------------------

def _extract_suggestions(text: str, fallback_suggestions: List[str]) -> List[str]:
    """
    Return up to 3 follow-up suggestions.

    If the model included a JSON/markdown suggestions block we skip parsing
    for simplicity and use contextual heuristics instead, falling back to
    static defaults when nothing domain-specific can be inferred.
    """
    text_lower = text.lower()

    dynamic: List[str] = []
    if "copper" in text_lower or "cu" in text_lower:
        dynamic.append("Show me a copper-based alloy frequency sweep")
    if "permeability" in text_lower or "magnetic" in text_lower:
        dynamic.append("How does permeability affect low-frequency shielding?")
    if "skin depth" in text_lower:
        dynamic.append("Calculate minimum thickness for 60 dB at 1 GHz")
    if "multilayer" in text_lower or "layer" in text_lower:
        dynamic.append("Design a 3-layer Cu/Ni/Fe composite shield")
    if "absorption" in text_lower:
        dynamic.append("Run a thickness sweep to visualise absorption loss")
    if "reflection" in text_lower:
        dynamic.append("Explain impedance mismatch at the air-material interface")
    if "composite" in text_lower or "polymer" in text_lower:
        dynamic.append("What filler fraction maximises SE in a polymer composite?")
    if "frequency" in text_lower:
        dynamic.append("Run a frequency sweep from 100 MHz to 10 GHz")

    suggestions = (dynamic + fallback_suggestions)[:3]
    if not suggestions:
        suggestions = fallback_suggestions[:3]
    return suggestions


_DEFAULT_SUGGESTIONS = [
    "What materials give the best SE at 1 GHz?",
    "Run a frequency sweep for my current composition",
    "Explain the difference between absorption and reflection loss",
]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/message", response_model=ChatResponse)
async def chat_message(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the Gemini AI EMI shielding assistant.

    Accepts an optional `context` object with the current simulation state
    so the assistant can give composition- and result-aware advice.
    """
    if not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Gemini API key is not configured. "
                "Set the GEMINI_API_KEY environment variable to enable the AI assistant."
            ),
        )

    try:
        import google.generativeai as genai  # type: ignore[import]
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "google-generativeai package is not installed. "
                "Run: pip install google-generativeai"
            ),
        )

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=_SYSTEM_PROMPT,
        )

        # Build the conversation history in the format Gemini expects.
        # Gemini uses 'model' for assistant turns; map accordingly.
        gemini_history = []
        for turn in request.history:
            role = "model" if turn.role in ("assistant", "model") else "user"
            gemini_history.append({"role": role, "parts": [turn.content]})

        # Prepend simulation context to the current user message when present.
        user_message = request.message
        if request.context:
            ctx = request.context
            context_lines: List[str] = ["[Current simulation context]"]

            composition = ctx.get("composition", {})
            if composition:
                comp_str = ", ".join(
                    f"{sym}: {pct:.1f}%" for sym, pct in composition.items()
                )
                context_lines.append(f"Composition: {comp_str}")

            if ctx.get("frequency_mhz") is not None:
                context_lines.append(f"Frequency: {ctx['frequency_mhz']} MHz")

            if ctx.get("thickness_mm") is not None:
                context_lines.append(f"Thickness: {ctx['thickness_mm']} mm")

            if ctx.get("grain_size_um") is not None:
                context_lines.append(f"Grain size: {ctx['grain_size_um']} µm")

            if ctx.get("analysisMode"):
                context_lines.append(f"Analysis mode: {ctx['analysisMode']}")

            latest = ctx.get("latestResult")
            if isinstance(latest, dict):
                se = latest.get("shielding_effectiveness_db") or latest.get(
                    "total_se_db"
                )
                if se is not None:
                    if isinstance(se, list):
                        context_lines.append(
                            f"Latest SE result: sweep with {len(se)} points, "
                            f"range {min(se):.1f}–{max(se):.1f} dB"
                        )
                    else:
                        context_lines.append(f"Latest SE result: {se:.1f} dB")

            context_prefix = "\n".join(context_lines)
            user_message = f"{context_prefix}\n\n{request.message}"

        # Start a multi-turn chat with existing history, then send the message.
        chat = model.start_chat(history=gemini_history)
        gemini_response = chat.send_message(user_message)
        response_text = gemini_response.text

        suggestions = _extract_suggestions(response_text, _DEFAULT_SUGGESTIONS)

        return ChatResponse(response=response_text, suggestions=suggestions)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Gemini chat request failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI assistant error: {str(exc)}",
        )
