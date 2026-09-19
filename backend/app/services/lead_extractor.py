"""
Pull visitor contact details out of an ongoing conversation.

Visitors volunteer their details in the middle of normal chat -- "sure, it's
rahul@gmail.com" or "call me on 98470 12345" -- but until now nothing looked
for them, so the Leads table stayed empty unless someone filled in the widget's
pre-chat form.

Two passes, cheapest first:

1. Regex over the visitor's own messages. Catches emails and Indian phone
   numbers reliably and costs nothing.
2. An LLM pass for the name, which regex cannot do ("I'm Rahul" vs "I'm
   looking for a car"). Only runs when a contact detail was actually found,
   so it costs one call per real lead rather than one per message.

Everything is best-effort: a failure here must never break the chat reply.
"""
import re
from typing import Dict, List, Optional

from loguru import logger

# Deliberately strict: a missed lead costs less than a wrong phone number in
# the sales team's list.
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# Indian mobile numbers. Rather than assume a grouping (98470 12345, 98470-12345,
# 9847 012 345 are all common), match any run of digits and separators that
# could hold 10-12 digits, then validate the digits in _clean_phone.
_PHONE_RE = re.compile(
    r"(?<![\d])(?:\+?91[\s.-]?|0)?[6-9][\d\s.-]{8,14}(?![\d])"
)

# Strings that look like phone numbers but are not.
_PHONE_BLOCKLIST = {
    "1234567890", "9999999999", "0000000000", "1111111111",
}


def _clean_phone(raw: str) -> Optional[str]:
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    if digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]
    if len(digits) != 10 or digits[0] not in "6789":
        return None
    if digits in _PHONE_BLOCKLIST or len(set(digits)) <= 2:
        return None
    return digits


def extract_contacts(messages: List[Dict]) -> Dict[str, Optional[str]]:
    """
    Scan the visitor's messages for an email and a phone number.

    Only 'user' messages are searched -- the assistant may quote an example
    address, and indexed page content often contains the business's own
    contact details, neither of which is the visitor's.
    """
    email: Optional[str] = None
    phone: Optional[str] = None

    for msg in messages:
        if msg.get("role") != "user":
            continue
        text = str(msg.get("content") or "")

        if not email:
            m = _EMAIL_RE.search(text)
            if m:
                email = m.group(0).lower()

        if not phone:
            for m in _PHONE_RE.finditer(text):
                cleaned = _clean_phone(m.group(0))
                if cleaned:
                    phone = cleaned
                    break

        if email and phone:
            break

    return {"email": email, "phone": phone}


async def extract_name(messages: List[Dict], llm_service) -> Optional[str]:
    """
    Ask the model for the visitor's name, or None when they never gave one.

    Regex cannot separate "I'm Rahul" from "I'm looking for a car", and a
    wrong name in the CRM is worse than a blank one, so this is left to the
    model with an explicit instruction to return NONE when unsure.
    """
    transcript = "\n".join(
        f"{m.get('role')}: {m.get('content')}"
        for m in messages
        if m.get("role") in ("user", "assistant")
    )[:4000]

    prompt = (
        "Read this chat transcript and find the VISITOR's own name, if they "
        "stated it.\n\n"
        "Rules:\n"
        "- Reply with the name only, nothing else.\n"
        "- Reply exactly NONE if they never gave their name, or if you are "
        "unsure.\n"
        "- A company, product or place name is not a person's name.\n"
        "- Do not guess from the email address.\n\n"
        f"Transcript:\n{transcript}\n\nName:"
    )

    try:
        raw = await llm_service.generate(prompt, temperature=0, max_tokens=20)
    except Exception as e:
        logger.warning(f"Name extraction failed: {e}")
        return None

    name = (raw or "").strip().strip('."\'')
    if not name or name.upper().startswith("NONE"):
        return None
    # A real first/last name is short; anything longer is the model rambling.
    if len(name) > 40 or len(name.split()) > 4:
        return None
    return name


async def capture_lead_from_conversation(
    db,
    llm_service,
    site_id: str,
    session_id: str,
    messages: List[Dict],
) -> Optional[Dict]:
    """
    Save or update a lead when the visitor has shared contact details.

    Returns the lead when one was written, else None. Safe to call after every
    message: it exits immediately when there is nothing new to record.
    """
    try:
        contacts = extract_contacts(messages)
        if not contacts["email"] and not contacts["phone"]:
            return None

        existing = await db.get_lead_by_session(site_id, session_id)

        # Nothing new since last time -- don't spend an LLM call on the name.
        if existing:
            known_email = existing.get("email")
            known_phone = (existing.get("metadata") or {}).get("phone")
            if (contacts["email"] or known_email) == known_email and \
               (contacts["phone"] or known_phone) == known_phone:
                return None

        name = await extract_name(messages, llm_service)
        profile = await extract_profile(messages, llm_service)

        metadata = dict((existing or {}).get("metadata") or {})
        if contacts["phone"]:
            metadata["phone"] = contacts["phone"]
        metadata.update(profile)
        metadata["score"] = score_lead(messages, contacts, profile)
        metadata["band"] = score_band(metadata["score"])
        metadata["captured_by"] = "conversation"
        metadata["message_count"] = len(messages)

        lead_data = {
            "site_id": site_id,
            "session_id": session_id,
            "email": contacts["email"] or (existing or {}).get("email"),
            "name": name or (existing or {}).get("name"),
            "source": "chat",
            "metadata": metadata,
        }

        if existing:
            await db.delete_lead(existing["lead_id"])

        lead = await db.save_lead(lead_data)
        logger.info(
            f"Lead captured [{metadata['band']} {metadata['score']}]: "
            f"site={site_id} name={lead_data['name']} "
            f"email={lead_data['email']} phone={metadata.get('phone')} "
            f"want={metadata.get('product') or metadata.get('intent')}"
        )
        return lead

    except Exception as e:
        # Lead capture is a bonus; never let it break the chat reply.
        logger.warning(f"Lead capture failed for session {session_id}: {e}")
        return None

# ---------------------------------------------------------------------------
# Lead profile
#
# Contact details alone tell the sales team who to call, not what to say. The
# conversation already contains that -- what they asked for, what they baulked
# at, how urgent it is -- so one extra model call turns a bare email into a
# briefing.
# ---------------------------------------------------------------------------

_PROFILE_PROMPT = """Read this chat transcript between a website visitor and a bot.
Return ONLY a JSON object, no other text, with these keys:

{
  "intent": "what the visitor wants, 3-6 words, or null",
  "product": "specific product/service/model they asked about, or null",
  "budget": "any budget or price range they mentioned, or null",
  "timeline": "when they want it (e.g. 'this month'), or null",
  "objection": "their main hesitation or concern, or null",
  "location": "their city/area if mentioned, or null",
  "summary": "one sentence a salesperson can read before calling"
}

Use null when the transcript does not say -- never guess.
Quote the visitor's own words for objection where possible.

Transcript:
{transcript}

JSON:"""


def score_lead(messages: List[Dict], contacts: Dict, profile: Dict) -> int:
    """
    Rank a lead 0-100 so the sales team calls the hottest first.

    Weighted towards things that actually predict a sale: leaving a phone
    number is a stronger signal than leaving an email, and asking about price
    or timing is stronger than general browsing.
    """
    score = 0
    if contacts.get("phone"):
        score += 40          # gave a number they expect to be called on
    if contacts.get("email"):
        score += 20

    user_msgs = [m for m in messages if m.get("role") == "user"]
    score += min(len(user_msgs) * 3, 15)

    if profile.get("product"):
        score += 10          # asked about something specific
    if profile.get("budget"):
        score += 10
    if profile.get("timeline"):
        score += 10

    return min(score, 100)


def score_band(score: int) -> str:
    if score >= 70:
        return "hot"
    if score >= 40:
        return "warm"
    return "cold"


async def extract_profile(messages: List[Dict], llm_service) -> Dict:
    """
    Summarise the conversation into sales-usable fields.

    Returns an empty dict on any failure -- a lead with contact details and no
    profile is still worth saving.
    """
    import json

    transcript = "\n".join(
        f"{m.get('role')}: {m.get('content')}"
        for m in messages
        if m.get("role") in ("user", "assistant")
    )[:6000]

    try:
        raw = await llm_service.generate(
            _PROFILE_PROMPT.replace("{transcript}", transcript),
            temperature=0,
            max_tokens=400,
        )
    except Exception as e:
        logger.warning(f"Profile extraction failed: {e}")
        return {}

    # Models often wrap JSON in ```json fences despite instructions.
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}

    try:
        data = json.loads(match.group(0))
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    allowed = {"intent", "product", "budget", "timeline",
               "objection", "location", "summary"}
    out = {}
    for k, v in data.items():
        if k in allowed and v not in (None, "", "null", "None"):
            out[k] = str(v)[:300]
    return out
