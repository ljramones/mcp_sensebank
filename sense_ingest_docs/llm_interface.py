#!/usr/bin/env python3
"""
LLM interface and prompt management for sensory extraction.
Handles OpenAI API calls, prompt generation, and JSON parsing.
"""

import os
import re
import json
import time
import logging
import requests
from typing import Dict, List, Tuple

# ------------------ LLM Configuration ------------------
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
MODEL_ID = os.getenv("SENSEBANK_MODEL", "llama3.1:8b")

# ------------------ Base Prompts ------------------
SYSTEM_PROMPT = """You are a specialist in extracting PHYSICAL SENSORY EXPERIENCES from historical texts.

EXTRACT ONLY these 5 types:
1. SMELL: Scents/odors - "acrid smoke", "sweet incense", "putrid decay", "fragrant blossoms"
2. SOUND: Audio qualities - "thunderous roar", "soft whisper", "piercing shriek", "distant chiming"  
3. TASTE: Flavors in mouth - "bitter medicine", "sweet honey", "metallic blood", "salty tears"
4. TOUCH: Physical sensations - "rough bark", "silky cloth", "burning heat", "icy wind"
5. SIGHT: Visual qualities - "blazing fire", "dim shadows", "brilliant gold", "shimmering water"

NEVER extract:
❌ Emotions: "angry look", "confident air", "nervous glance"
❌ Body parts: "eyes", "face", "hands", "mouth"  
❌ Actions: "breathing", "heartbeat", "running", "walking"
❌ Objects without qualities: "sword", "tea", "house", "door"

QUALITY RULES:
• Each term must describe HOW something affects the senses
• Use specific sensory vocabulary (acrid, thunderous, silky, blazing, putrid)
• 2-3 words maximum per term
• Only extract 3-5 terms per passage (be highly selective)

Return: {"items":[{"term":"Acrid Smoke","category":"smell","notes":"brief context"}]}"""

USER_TEMPLATE = """Extract ONLY genuine sensory experiences that characters physically perceive through their 5 senses.

SETTING: {locale}, {era} period, {register} context, {weather} conditions

FOCUS ON:
• What characters SMELL (scents, odors, aromas)
• What they HEAR (sound qualities, not just "voice" or "sound")  
• What they TASTE (actual flavors in mouth)
• What they FEEL physically (textures, temperatures, pressures)
• What they SEE (visual qualities, not just objects)

PASSAGE:
{passage}

Extract 3-5 high-quality sensory terms with specific descriptive vocabulary."""

# ------------------ Context-Aware Vocabulary ------------------
REGISTER_SPECIFIC_VOCABULARY = {
    "court": {
        "scent": ["perfumed oils", "aromatic incense", "exotic fragrances", "sandalwood", "precious aromatics"],
        "audio": ["ceremonial bells", "hushed whispers", "formal announcements", "silk rustling", "jade chimes"],
        "flavor": ["exotic delicacies", "aged wines", "honeyed treats", "rare spices", "imperial teas"],
        "texture": ["silk brocade", "polished jade", "smooth lacquer", "soft furs", "precious metals"],
        "temperature": ["warm pavilions", "cool marble", "heated chambers"],
        "luminosity": ["golden glow", "pearl luminescence", "jeweled brilliance", "lantern light"],
        "atmosphere": ["formal grandeur", "hushed reverence", "luxurious comfort"]
    },
    "market": {
        "scent": ["cooking spices", "fresh produce", "smoky braziers", "animal odors", "dusty goods"],
        "audio": ["haggling voices", "clattering coins", "creaking carts", "animal sounds", "vendor calls"],
        "flavor": ["street food", "fresh fruits", "coarse bread", "simple wines", "common teas"],
        "texture": ["rough hemp", "worn leather", "coarse fabric", "wooden goods", "metal tools"],
        "temperature": ["dusty heat", "cool shade", "warm crowds"],
        "luminosity": ["flickering lanterns", "dusty sunbeams", "shadowy stalls", "bright daylight"],
        "atmosphere": ["bustling energy", "crowded chaos", "commercial bustle"]
    },
    "monastic": {
        "scent": ["burning incense", "medicinal herbs", "aged wood", "paper scrolls", "stone dampness"],
        "audio": ["chanted prayers", "temple bells", "rustling robes", "turning pages", "meditation silence"],
        "flavor": ["simple teas", "medicinal broths", "plain rice", "herbal remedies"],
        "texture": ["worn stone", "smooth prayer beads", "rough hemp robes", "soft paper", "cold metal"],
        "temperature": ["cool halls", "warm meditation rooms", "chilled mornings"],
        "luminosity": ["candlelight", "dawn glow", "sacred flames", "filtered sunlight"],
        "atmosphere": ["sacred silence", "contemplative peace", "spiritual weight"]
    },
    "military": {
        "scent": ["weapon oil", "leather armor", "campfire smoke", "horse sweat", "blood"],
        "audio": ["clashing weapons", "shouted commands", "marching feet", "armor clanking", "battle cries"],
        "flavor": ["soldier rations", "strong wine", "metallic taste", "dried meat"],
        "texture": ["rough armor", "sharp blades", "coarse rope", "hard ground", "leather straps"],
        "temperature": ["cold steel", "heated battles", "campfire warmth"],
        "luminosity": ["gleaming weapons", "flickering campfires", "dawn light", "torch flames"],
        "atmosphere": ["tense alertness", "disciplined order", "battle fury"]
    },
    "festival": {
        "scent": ["festive foods", "flower garlands", "burning offerings", "crowd odors", "celebratory incense"],
        "audio": ["joyful music", "crowd cheers", "festival drums", "laughter", "ceremonial songs"],
        "flavor": ["festive delicacies", "sweet treats", "celebratory wines", "special foods"],
        "texture": ["colorful fabrics", "smooth decorations", "soft flowers", "polished ornaments"],
        "temperature": ["warm celebrations", "cool evening air", "heated crowds"],
        "luminosity": ["bright lanterns", "colorful displays", "festive lights", "fireworks"],
        "atmosphere": ["joyful celebration", "festive energy", "communal excitement"]
    },
    "garden": {
        "scent": ["blooming flowers", "fresh earth", "pond water", "growing plants", "morning dew"],
        "audio": ["rustling leaves", "flowing water", "bird songs", "gentle breezes", "insect buzzing"],
        "flavor": ["fresh fruits", "flower nectar", "herb flavors", "clean water"],
        "texture": ["smooth stones", "soft moss", "rough bark", "flowing water", "delicate petals"],
        "temperature": ["cool shade", "warm sunshine", "gentle breezes", "cool water"],
        "luminosity": ["dappled sunlight", "morning glow", "reflecting water", "flower colors"],
        "atmosphere": ["peaceful tranquility", "natural harmony", "serene beauty"]
    },
    "maritime": {
        "scent": ["salt air", "tar rope", "fish odors", "wet wood", "ocean spray"],
        "audio": ["creaking wood", "lapping waves", "wind in sails", "rope sounds", "seabird calls"],
        "flavor": ["salty spray", "preserved fish", "ship rations", "brackish water"],
        "texture": ["rough rope", "wet wood", "salt-crusted surfaces", "rolling deck", "cold water"],
        "temperature": ["ocean winds", "sun-heated deck", "cold spray", "humid air"],
        "luminosity": ["sun on water", "misty horizons", "lantern light", "starlit nights"],
        "atmosphere": ["open vastness", "maritime rhythm", "oceanic power"]
    }
}

ERA_SPECIFIC_FOCUS = {
    "Song": {
        "priority_categories": ["scent", "flavor", "texture", "luminosity"],
        "cultural_emphasis": ["tea culture", "silk textiles", "porcelain", "scholarly pursuits"],
        "common_sensory_experiences": ["tea ceremonies", "silk clothing", "ink and paper", "garden aesthetics"]
    },
    "Heian": {
        "priority_categories": ["scent", "texture", "atmosphere", "luminosity"],
        "cultural_emphasis": ["court refinement", "seasonal awareness", "textile aesthetics", "incense culture"],
        "common_sensory_experiences": ["incense blending", "silk layering", "moon viewing", "poetry composition"]
    },
    "Mughal": {
        "priority_categories": ["scent", "flavor", "luminosity", "texture"],
        "cultural_emphasis": ["architectural grandeur", "garden culture", "textile luxury", "culinary sophistication"],
        "common_sensory_experiences": ["garden pavilions", "marble architecture", "spiced cuisine",
                                       "jeweled decoration"]
    },
    "Chimú": {
        "priority_categories": ["texture", "luminosity", "scent", "temperature"],
        "cultural_emphasis": ["metallurgy", "textile arts", "maritime culture", "desert environment"],
        "common_sensory_experiences": ["metal working", "cotton textiles", "ocean proximity", "desert conditions"]
    }
}


# ------------------ LLM Interface Functions ------------------
def chat_json(system: str, user: str, model: str = MODEL_ID, temperature: float = 0.2) -> Dict:
    """
    Creates and sends a chat completion request with robust JSON parsing.
    """
    logger = logging.getLogger("sense-ingest")
    url = f"{OPENAI_API_BASE.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
    }
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("POST %s model=%s user_len=%d", url, model, len(user))

    t0 = time.perf_counter()
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    dt = time.perf_counter() - t0
    logger.info("LLM call %.2fs", dt)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]

    # More robust JSON extraction and parsing
    try:
        # First try: parse as-is
        return json.loads(content)
    except json.JSONDecodeError:
        # Second try: extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?})\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Third try: find the first complete JSON object
        json_match = re.search(r'\{.*}', content, flags=re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Fourth try: fix common JSON issues
        try:
            # Fix duplicate "items" keys by merging arrays
            if '"items":' in content:
                # Extract all items arrays
                all_items = []
                for match in re.finditer(r'"items":\s*(\[.*?])', content):
                    try:
                        items_array = json.loads(match.group(1))
                        if isinstance(items_array, list):
                            all_items.extend(items_array)
                    except json.JSONDecodeError:
                        continue

                if all_items:
                    return {"items": all_items}

            # Remove trailing commas
            fixed_content = re.sub(r',(\s*[}\]])', r'\1', content)
            # Extract JSON object
            json_match = re.search(r'\{.*}', fixed_content, flags=re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

        # Fifth try: return a safe fallback
        logger.warning(f"Failed to parse JSON response: {content[:200]}...")
        return {"items": []}  # Safe fallback - empty items list


# ------------------ Prompt Generation ------------------
def get_contextual_vocabulary(locale: str, era: str, register: str) -> Dict[str, List[str]]:
    """Return vocabulary appropriate for specific historical context."""
    # Base vocabulary from register
    base_vocab = REGISTER_SPECIFIC_VOCABULARY.get(register, {})

    # Era-specific modifications
    era_focus = ERA_SPECIFIC_FOCUS.get(era, {})
    priority_cats = era_focus.get("priority_categories", [])

    # Enhance vocabulary for priority categories
    enhanced_vocab = {}
    for category, terms in base_vocab.items():
        if category in priority_cats:
            # Add more terms for priority categories
            enhanced_vocab[category] = terms + [f"refined {term}" for term in terms[:3]]
        else:
            enhanced_vocab[category] = terms

    return enhanced_vocab


def get_enhanced_prompts(locale: str, era: str, register: str) -> Tuple[str, str]:
    """Generate context-aware prompts based on historical setting."""
    # Get contextual vocabulary
    vocab = get_contextual_vocabulary(locale, era, register)
    era_focus = ERA_SPECIFIC_FOCUS.get(era, {})
    cultural_emphasis = era_focus.get("cultural_emphasis", [])

    # Build context-specific examples
    context_examples = []
    if register in REGISTER_SPECIFIC_VOCABULARY:
        reg_vocab = REGISTER_SPECIFIC_VOCABULARY[register]
        for category, terms in reg_vocab.items():
            if terms and category in ["scent", "audio", "flavor", "texture", "luminosity"]:
                # Map to our 5 categories
                cat_map = {"scent": "smell", "audio": "sound", "flavor": "taste",
                           "texture": "touch", "luminosity": "sight"}
                mapped_cat = cat_map.get(category, category)
                if mapped_cat in ["smell", "sound", "taste", "touch", "sight"]:
                    context_examples.append(f"{mapped_cat.upper()}: \"{terms[0]}\"")

    # Enhanced system prompt with cultural context
    enhanced_system = f"""You are a specialist in extracting PHYSICAL SENSORY EXPERIENCES from historical texts set in {locale} during the {era} period.

HISTORICAL CONTEXT: {locale}, {era} era, {register} setting
CULTURAL FOCUS: {', '.join(cultural_emphasis) if cultural_emphasis else 'authentic historical sensory experiences'}

EXTRACT ONLY these 5 types with period-appropriate vocabulary:
1. SMELL: Scents/odors - "aromatic incense", "smoky braziers", "fragrant oils"
2. SOUND: Audio qualities - "ceremonial bells", "distant chiming", "rustling silk"  
3. TASTE: Flavors in mouth - "bitter tea", "sweet wine", "herbal remedies"
4. TOUCH: Physical sensations - "silk brocade", "rough hemp", "smooth jade"
5. SIGHT: Visual qualities - "golden glow", "flickering lanterns", "brilliant bronze"

{f"PREFERRED VOCABULARY FOR THIS SETTING: {', '.join(context_examples[:3])}" if context_examples else ""}

NEVER extract:
❌ Emotions: "angry look", "confident air", "nervous glance"
❌ Body parts: "eyes", "face", "hands", "mouth"  
❌ Actions: "breathing", "heartbeat", "running", "walking"
❌ Objects without qualities: "sword", "tea", "house", "door"

QUALITY RULES:
• Each term must describe HOW something affects the senses
• Use vocabulary appropriate to {era} period {locale}
• Prefer terms authentic to {register} settings
• 2-3 words maximum per term
• Only extract 3-5 terms per passage (be highly selective)

Return: {{"items":[{{"term":"Aromatic Incense","category":"smell","notes":"brief context"}}]}}"""

    # Enhanced user template with cultural guidance
    enhanced_user = f"""Extract ONLY genuine sensory experiences from this {era} period text set in {locale}.

HISTORICAL SETTING: {locale}, {era} period, {register} context
CULTURAL GUIDANCE: Focus on sensory experiences authentic to this time and place.

PRIORITIZE THESE SENSORY AREAS for {era} culture:
{f"• {', '.join(era_focus.get('priority_categories', []))}" if era_focus.get('priority_categories') else "• All sensory categories equally"}

FOCUS ON:
• What characters SMELL (incense, cooking, nature, architectural scents)
• What they HEAR (ceremonial sounds, natural acoustics, period-specific audio)  
• What they TASTE (period foods, drinks, medicines)
• What they FEEL physically (period textiles, materials, temperatures)
• What they SEE (period lighting, colors, architectural elements)

PASSAGE:
{{passage}}

Extract 3-5 high-quality sensory terms authentic to {era} period {locale}."""

    return enhanced_system, enhanced_user


# ------------------ High-Level LLM Interface ------------------
def extract_sensory_terms_standard(chunk: str, locale: str = "", era: str = "", register: str = "common",
                                   weather: str = "any") -> List[Dict]:
    """
    Standard sensory extraction using base prompts.

    :param chunk: Text chunk to process
    :param locale: Historical locale context
    :param era: Historical era context
    :param register: Social register context
    :param weather: Weather context
    :return: List of extracted sensory terms
    """
    logger = logging.getLogger("sense-ingest")

    # Use standard prompts
    obj = chat_json(SYSTEM_PROMPT, USER_TEMPLATE.format(
        locale=locale, era=era, register=register, weather=weather, passage=chunk
    ))

    items = obj.get("items", []) if isinstance(obj, dict) else []

    # Basic cleanup
    validated = []
    for item in items:
        term = (item.get("term", "") or "").strip()
        category = (item.get("category", "") or "").strip().lower()
        notes = (item.get("notes", "") or "").strip()

        if term and category:
            validated.append({
                "term": term.title(),
                "category": category,
                "notes": notes
            })

    logger.debug(f"Standard extraction: {len(validated)}/{len(items)} terms")
    return validated


def extract_sensory_terms_context_aware(chunk: str, locale: str, era: str, register: str = "common",
                                        weather: str = "any") -> List[Dict]:
    """
    Context-aware sensory extraction using enhanced prompts.

    :param chunk: Text chunk to process
    :param locale: Historical locale context
    :param era: Historical era context
    :param register: Social register context
    :param weather: Weather context
    :return: List of extracted sensory terms
    """
    logger = logging.getLogger("sense-ingest")

    # Get enhanced prompts for this cultural context
    system_prompt, user_prompt = get_enhanced_prompts(locale, era, register)

    # Get LLM response using context-aware prompts
    obj = chat_json(system_prompt, user_prompt.format(
        locale=locale, era=era, register=register, weather=weather, passage=chunk
    ))

    items = obj.get("items", []) if isinstance(obj, dict) else []

    # Basic cleanup
    validated = []
    for item in items:
        term = (item.get("term", "") or "").strip()
        category = (item.get("category", "") or "").strip().lower()
        notes = (item.get("notes", "") or "").strip()

        if term and category:
            validated.append({
                "term": term.title(),
                "category": category,
                "notes": notes
            })

    logger.debug(f"Context-aware extraction for {era} {locale} ({register}): {len(validated)}/{len(items)} terms")
    return validated


def extract_sensory_terms(chunk: str, locale: str = "", era: str = "", register: str = "common",
                          weather: str = "any") -> List[Dict]:
    """
    Main extraction function that chooses between standard and context-aware extraction.

    :param chunk: Text chunk to process
    :param locale: Historical locale context
    :param era: Historical era context
    :param register: Social register context
    :param weather: Weather context
    :return: List of extracted sensory terms
    """
    # Check if we have enough context for cultural enhancement
    if locale and era and register != "common":
        return extract_sensory_terms_context_aware(chunk, locale, era, register, weather)
    else:
        return extract_sensory_terms_standard(chunk, locale, era, register, weather)