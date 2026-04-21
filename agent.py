import asyncio
import json
import logging
import os
import random
import time as _boot_time
from dataclasses import dataclass, field
from enum import Enum, auto

import httpx
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli, function_tool
from livekit.plugins import bey, cartesia, deepgram, openai, silero

from app.config import get_settings

# LiveKit SDK reads LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
# directly from os.environ — load_dotenv() is required here
load_dotenv()

settings = get_settings()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lylo.agent")
logger.info("=== Agent module loaded at boot ===")

BEY_AVATAR_MALE_MODELS = [
    m for m in [
        os.getenv("BEY_AVATAR_MALE_MODEL_1"),
        os.getenv("BEY_AVATAR_MALE_MODEL_2"),
    ] if m
]

BEY_AVATAR_FEMALE_MODELS = [
    m for m in [
        os.getenv("BEY_AVATAR_FEMALE_MODEL_1"),
        os.getenv("BEY_AVATAR_FEMALE_MODEL_2"),
        os.getenv("BEY_AVATAR_FEMALE_MODEL_3"),
    ] if m
]


def pick_avatar(gender: str) -> str:
    models = BEY_AVATAR_FEMALE_MODELS if gender == "female" else BEY_AVATAR_MALE_MODELS
    if not models:
        fallback = BEY_AVATAR_FEMALE_MODELS or BEY_AVATAR_MALE_MODELS
        if not fallback:
            raise ValueError(f"Aucun avatar Bey configuré pour le genre '{gender}' — vérifiez BEY_AVATAR_MALE_MODEL_* / BEY_AVATAR_FEMALE_MODEL_* dans .env")
        logger.warning(f"[AVATAR] Aucun avatar pour genre='{gender}', fallback sur l'autre genre")
        models = fallback
    return random.choice(models)


# ─────────────────────────────────────────────
# Machine à états
# ─────────────────────────────────────────────

class AgentPhase(Enum):
    # Phase 1 — Profil
    GREET = auto()
    GET_GENDER = auto()
    GET_AGE = auto()
    GET_PREGNANT = auto()
    GET_ALLERGIES = auto()
    GET_ALLERGY_DETAIL = auto()
    # Phase 2 — Questionnaire
    Q_FAVORITES = auto()
    Q_JUSTIFY_FAV_1 = auto()
    Q_JUSTIFY_FAV_2 = auto()
    Q_LEAST = auto()
    Q_JUSTIFY_LEAST_1 = auto()
    Q_JUSTIFY_LEAST_2 = auto()
    Q_CONFIRM = auto()
    # Phase 3 — Formules
    INTENSITY = auto()
    PRESENT_FORMULAS = auto()
    # Phase 4 — Personnalisation / Découverte
    CUSTOMIZATION = auto()
    # Phase 5 — Fin
    STANDBY = auto()


@dataclass
class SessionState:
    phase: AgentPhase = AgentPhase.GREET
    current_question_index: int = 0
    current_top_2: list = field(default_factory=list)
    current_bottom_2: list = field(default_factory=list)
    profile: dict = field(default_factory=dict)
    answers_saved: int = 0
    formula_type: str | None = None
    selected_formula_index: int | None = None


# ─────────────────────────────────────────────
# Prompts par état
# ─────────────────────────────────────────────

PERSONALITY_FR = """Tu t'appelles {ai_name}. Tu travailles pour Le Studio des Parfums.

TON & PERSONNALITÉ : Tu es chaleureux(se), souriant(e) et passionné(e) par l'univers du parfum. Tu parles de façon naturelle et fluide, jamais comme un robot. Utilise un ton conversationnel, détendu mais professionnel. VOUVOIE TOUJOURS l'utilisateur. Fais des petites réactions naturelles ("Oh très bien !", "Ah c'est intéressant !"). Parle avec des phrases courtes et naturelles. Ne mentionne jamais Lilo, Le Studio des Parfums, ni que tu es une IA ou une assistante vocale.

RÈGLES ABSOLUES : Ne jamais écrire la syntaxe des function calls dans ton texte. Les fonctions doivent être appelées UNIQUEMENT via l'interface outil. Ne jamais corriger, signaler ou commenter la formulation de l'utilisateur (singulier/pluriel, accents, majuscules) — normalise silencieusement et continue.

Si l'utilisateur pose une question sur la parfumerie, réponds-y brièvement et avec expertise, puis reviens immédiatement à ta mission actuelle.

GESTION DES RÉPONSES ABSURDES : Utilise l'humour pour demander la vraie information. Ex: "500 ans ? Quel beau parcours ! Mais pour le parfum, j'ai besoin de votre âge terrestre."
"""

PERSONALITY_EN = """Your name is {ai_name}. You work for Le Studio des Parfums.

TONE & PERSONALITY: You are warm, friendly, and passionate about the world of perfume. You speak naturally and fluidly, never like a robot. Use a conversational, relaxed but professional tone. React naturally to answers ("Oh great!", "That's interesting!"). Speak in short, natural sentences. NEVER mention Lilo, Le Studio des Parfums, or that you are an AI or voice assistant. ALWAYS speak in English.

ABSOLUTE RULES: Never write function call syntax in your text. Functions must be called ONLY through the tool interface. Never correct, signal or comment on the user's wording (singular/plural, accents, capitalization) — normalize silently and continue.

If the user asks a perfumery question, answer briefly and expertly, then return immediately to your current mission.

ABSURD ANSWER HANDLING: Use humor to get the real information. Ex: "500 years old? What a journey! But for the perfume, I need your earthly age."
"""


def get_prompt(state: SessionState, config: dict, ai_name: str, is_en: bool, input_mode: str) -> str:
    phase = state.phase
    lang = "en" if is_en else "fr"
    personality = (PERSONALITY_EN if is_en else PERSONALITY_FR).format(ai_name=ai_name)

    questions = config.get("questions", [])
    num_questions = len(questions)

    # ── Phase 1 : Profil ──────────────────────────────────────────────────

    if phase == AgentPhase.GREET:
        if is_en:
            mission = f"Greet the user warmly and simply. Introduce yourself just with your first name ({ai_name}). For example: 'Hey! I'm {ai_name}, nice to meet you! And what's your name?' Be natural and friendly. As soon as the user gives their name, call save_user_profile(field='first_name', value=<their name>) IMMEDIATELY."
        else:
            mission = f"Saluez l'utilisateur chaleureusement et simplement en le vouvoyant. Présentez-vous juste avec votre prénom ({ai_name}). Par exemple : 'Bonjour ! Moi c'est {ai_name}, enchantée ! Et vous, comment vous appelez-vous ?' Soyez naturel(le). Dès que l'utilisateur donne son prénom, appelez IMMÉDIATEMENT save_user_profile(field='first_name', value=<le prénom>)."

    elif phase == AgentPhase.GET_GENDER:
        first_name = state.profile.get("first_name", "")
        if is_en:
            mission = f"The user's name is {first_name}. Ask naturally whether it's a masculine or feminine name, for example: 'Nice name! Is it more of a masculine or feminine name?' As soon as they answer, IMMEDIATELY call save_user_profile(field='gender', value='masculin') or save_user_profile(field='gender', value='féminin')."
        else:
            mission = f"Le prénom de l'utilisateur est {first_name}. Demandez naturellement si c'est un prénom masculin ou féminin, par exemple : 'Joli prénom ! C'est plutôt masculin ou féminin ?' Dès qu'il/elle répond, appelez IMMÉDIATEMENT save_user_profile(field='gender', value='masculin') ou save_user_profile(field='gender', value='féminin')."

    elif phase == AgentPhase.GET_AGE:
        first_name = state.profile.get("first_name", "")
        if is_en:
            mission = f"Ask {first_name} their age casually, for example: 'And how old are you?' IMPORTANT: Accept numbers written in words (e.g. 'twenty-five' → 25). Valid range: 12–120. If absurd, use humor. As soon as they give a valid age, IMMEDIATELY call save_user_profile(field='age', value=<age as number>)."
        else:
            mission = f"Demandez l'âge de {first_name} avec légèreté, par exemple : 'Et vous avez quel âge ?' IMPORTANT : Acceptez les nombres écrits en lettres (ex : 'vingt-cinq' → 25). Plage valide : 12–120 ans. Si l'âge est absurde, utilisez l'humour. Dès qu'il/elle donne un âge valide, appelez IMMÉDIATEMENT save_user_profile(field='age', value=<âge en chiffre>)."

    elif phase == AgentPhase.GET_PREGNANT:
        first_name = state.profile.get("first_name", "")
        if is_en:
            mission = f"Ask {first_name} naturally and delicately whether she is pregnant or breastfeeding, as some fragrance ingredients require precautions. For example: 'Just to make sure we create the safest formula for you — are you currently pregnant or breastfeeding?' As soon as she answers, IMMEDIATELY call save_user_profile(field='pregnant', value='oui') or save_user_profile(field='pregnant', value='non')."
        else:
            mission = f"Demandez à {first_name} naturellement et avec délicatesse si elle est enceinte ou allaitante, car certains ingrédients demandent des précautions. Par exemple : 'Pour vous garantir la formule la plus sûre — êtes-vous actuellement enceinte ou allaitante ?' Dès qu'elle répond, appelez IMMÉDIATEMENT save_user_profile(field='pregnant', value='oui') ou save_user_profile(field='pregnant', value='non')."

    elif phase == AgentPhase.GET_ALLERGIES:
        first_name = state.profile.get("first_name", "")
        if is_en:
            mission = f"Ask {first_name} naturally if they have any allergies or sensitivities to certain ingredients, for example: 'Before we start, do you have any allergies or sensitivities to certain ingredients?' — If NO: IMMEDIATELY call save_user_profile(field='has_allergies', value='non'). — If YES: IMMEDIATELY call save_user_profile(field='has_allergies', value='oui')."
        else:
            mission = f"Demandez à {first_name} naturellement s'il/elle a des allergies ou sensibilités particulières, par exemple : 'Avant qu'on commence, est-ce que vous avez des allergies ou des sensibilités à certains ingrédients ?' — Si NON : appelez IMMÉDIATEMENT save_user_profile(field='has_allergies', value='non'). — Si OUI : appelez IMMÉDIATEMENT save_user_profile(field='has_allergies', value='oui')."

    elif phase == AgentPhase.GET_ALLERGY_DETAIL:
        first_name = state.profile.get("first_name", "")
        if is_en:
            mission = f"Ask {first_name} which ingredients or substances they are allergic to, for example: 'Of course! Which ingredients or substances are you allergic to?' As soon as they answer, IMMEDIATELY call save_user_profile(field='allergies', value=<the allergies mentioned>)."
        else:
            mission = f"Demandez à {first_name} à quels ingrédients ou substances il/elle est allergique, par exemple : 'Bien sûr ! À quels ingrédients ou substances êtes-vous allergique ?' Dès qu'il/elle répond, appelez IMMÉDIATEMENT save_user_profile(field='allergies', value=<les allergies mentionnées>)."

    # ── Phase 2 : Questionnaire ───────────────────────────────────────────

    elif phase == AgentPhase.Q_FAVORITES:
        q = questions[state.current_question_index]
        q_num = state.current_question_index + 1
        choices_str = ", ".join(c["label"] if isinstance(c, dict) else c for c in q.get("choices", []))
        first_name = state.profile.get("first_name", "")

        click_hint = ""
        if input_mode == "click":
            click_hint = (" Before asking, call request_top_2_click(question_id) to signal the interface to show a 'Reply' button." if is_en
                          else " Avant de poser la question, appelez request_top_2_click(question_id) pour signaler à l'interface d'afficher le bouton 'Répondre'.")

        if is_en:
            mission = f"""It is now question {q_num} of {num_questions}.

Question (id={q['id']}): "{q['question']}"
Available choices: {choices_str}

STEP: Ask {first_name} for their 2 FAVORITE choices in ONE natural sentence. Do NOT enumerate the choices aloud — the user can see them on screen.{click_hint}

Once the user gives 2 choices:
1. Match each choice to the canonical label (accept singular/plural, accents, 'the ...', lowercase: 'campagnes' → 'Campagne', 'forêts' → 'Forêt'). Do NOT ask for clarification for minor variations — just normalize silently.
2. Call notify_top_2(question_id={q['id']}, top_2=[choice1, choice2]) IMMEDIATELY.
3. Your mission for this step is complete."""
        else:
            mission = f"""C'est maintenant la question {q_num} sur {num_questions}.

Question (id={q['id']}) : "{q['question']}"
Choix disponibles : {choices_str}

ÉTAPE : Demandez à {first_name} ses 2 choix PRÉFÉRÉS en UNE seule phrase naturelle. Ne lisez JAMAIS les choix à voix haute — l'utilisateur les voit à l'écran.{click_hint}

Une fois que l'utilisateur donne 2 choix :
1. Faites correspondre silencieusement chaque mot au label canonique le plus proche ('campagnes' → 'Campagne', 'montagnes' → 'Montagne', 'forêts' → 'Forêt', etc.). INTERDIT ABSOLU : ne jamais signaler, corriger, mentionner ou commenter la forme donnée par l'utilisateur. Agissez directement.
2. Appelez IMMÉDIATEMENT notify_top_2(question_id={q['id']}, top_2=[choix1, choix2]).
3. Votre mission est terminée."""

    elif phase == AgentPhase.Q_JUSTIFY_FAV_1:
        q = questions[state.current_question_index]
        top_2 = state.current_top_2
        choice = top_2[0] if top_2 else "?"
        choice2 = top_2[1] if len(top_2) > 1 else "?"
        if is_en:
            mission = f"""Ask the user why they like "{choice}". Listen and briefly react naturally. Once the user has answered, IMMEDIATELY call notify_justification_top_2(question_id={q['id']}, choice="{choice2}") to move to the next step."""
        else:
            mission = f"""Demandez à l'utilisateur pourquoi il/elle aime "{choice}". Écoutez et rebondissez brièvement de façon naturelle. Une fois que l'utilisateur a répondu, appelez IMMÉDIATEMENT notify_justification_top_2(question_id={q['id']}, choice="{choice2}") pour passer à l'étape suivante."""

    elif phase == AgentPhase.Q_JUSTIFY_FAV_2:
        q = questions[state.current_question_index]
        top_2 = state.current_top_2
        choice = top_2[1] if len(top_2) > 1 else "?"
        if is_en:
            mission = f"""Ask the user why they like "{choice}". Listen and briefly react naturally. Once the user has answered, IMMEDIATELY call notify_asking_bottom_2(question_id={q['id']}, top_2={state.current_top_2}) to move to the least liked choices step."""
        else:
            mission = f"""Demandez à l'utilisateur pourquoi il/elle aime "{choice}". Écoutez et rebondissez brièvement. Une fois que l'utilisateur a répondu, appelez IMMÉDIATEMENT notify_asking_bottom_2(question_id={q['id']}, top_2={state.current_top_2}) pour passer à l'étape des choix les moins aimés."""

    elif phase == AgentPhase.Q_LEAST:
        q = questions[state.current_question_index]
        top_2 = state.current_top_2
        choices_str = ", ".join(c["label"] if isinstance(c, dict) else c for c in q.get("choices", []))

        click_hint = ""
        if input_mode == "click":
            click_hint = (" Before asking, call request_bottom_2_click(question_id) to signal the interface." if is_en
                          else " Avant de poser la question, appelez request_bottom_2_click(question_id) pour signaler à l'interface.")

        if is_en:
            mission = f"""Ask the user for their 2 LEAST liked choices from the REMAINING choices (excluding their favorites: {top_2}).{click_hint}

IMPORTANT: Never accept one of {top_2} as a least liked choice. If the user picks one, point it out with humor and ask again.

Once the user gives 2 least liked choices:
1. Match each to the canonical label (accept singular/plural, accents, lowercase — normalize silently without asking for confirmation).
2. Call notify_bottom_2(question_id={q['id']}, bottom_2=[choice1, choice2]) IMMEDIATELY.
3. Your mission is complete."""
        else:
            mission = f"""Demandez les 2 choix les MOINS aimés parmi les choix RESTANTS (en excluant les favoris : {top_2}).{click_hint}

IMPORTANT : N'acceptez JAMAIS un choix de {top_2} comme moins aimé. Si l'utilisateur en choisit un, signalez-le avec humour et redemandez.

Une fois que l'utilisateur donne 2 choix :
1. Faites correspondre silencieusement chaque mot au label canonique le plus proche. INTERDIT ABSOLU : ne jamais signaler, corriger, mentionner ou commenter la forme donnée. Agissez directement.
2. Appelez IMMÉDIATEMENT notify_bottom_2(question_id={q['id']}, bottom_2=[choix1, choix2]).
3. Votre mission est terminée."""

    elif phase == AgentPhase.Q_JUSTIFY_LEAST_1:
        q = questions[state.current_question_index]
        bottom_2 = state.current_bottom_2
        choice = bottom_2[0] if bottom_2 else "?"
        choice2 = bottom_2[1] if len(bottom_2) > 1 else "?"
        if is_en:
            mission = f"""In a SINGLE reply, ask the user why they dislike "{choice}". Do NOT split into two messages — react and ask in one sentence (e.g. "Interesting! And why don't you like {choice}?"). Once the user has answered, IMMEDIATELY call notify_justification_bottom_2(question_id={q['id']}, choice="{choice2}") to move to the next step."""
        else:
            mission = f"""En UNE SEULE réplique, demandez pourquoi l'utilisateur n'aime pas "{choice}". Ne divisez PAS en deux messages — réagissez et posez la question en une seule phrase (ex : "C'est noté ! Et pourquoi {choice} ne vous plaît-il/elle pas ?"). Une fois que l'utilisateur a répondu, appelez IMMÉDIATEMENT notify_justification_bottom_2(question_id={q['id']}, choice="{choice2}") pour passer à l'étape suivante."""

    elif phase == AgentPhase.Q_JUSTIFY_LEAST_2:
        q = questions[state.current_question_index]
        bottom_2 = state.current_bottom_2
        choice = bottom_2[1] if len(bottom_2) > 1 else "?"
        top_2 = state.current_top_2
        if is_en:
            mission = f"""Ask the user why they dislike "{choice}". Listen and briefly react. Once the user has answered, IMMEDIATELY call notify_awaiting_confirmation(question_id={q['id']}, top_2={top_2}, bottom_2={bottom_2}) to move to the confirmation step."""
        else:
            mission = f"""Demandez pourquoi l'utilisateur n'aime pas "{choice}". Écoutez et rebondissez brièvement. Une fois que l'utilisateur a répondu, appelez IMMÉDIATEMENT notify_awaiting_confirmation(question_id={q['id']}, top_2={top_2}, bottom_2={bottom_2}) pour passer à la confirmation."""

    elif phase == AgentPhase.Q_CONFIRM:
        q = questions[state.current_question_index]
        top_2 = state.current_top_2
        bottom_2 = state.current_bottom_2
        if is_en:
            mission = f"""Summarize the user's choices conversationally: "So if I recap: your favorites are {top_2[0] if top_2 else '?'} and {top_2[1] if len(top_2) > 1 else '?'}, and the ones you like least are {bottom_2[0] if bottom_2 else '?'} and {bottom_2[1] if len(bottom_2) > 1 else '?'}. Is that right?"

— If user CONFIRMS: Call IMMEDIATELY save_answer(question_id={q['id']}, question_text="{q['question']}", top_2={top_2}, bottom_2={bottom_2}).
— If user wants to MODIFY: Ask what they'd like to change, update the choices, redo the summary, and wait for confirmation. Only call save_answer after explicit confirmation."""
        else:
            mission = f"""Récapitulez les choix de l'utilisateur de façon conversationnelle : "D'accord, donc si je résume : vos coups de cœur c'est {top_2[0] if top_2 else '?'} et {top_2[1] if len(top_2) > 1 else '?'}, et ceux qui vous parlent le moins c'est {bottom_2[0] if bottom_2 else '?'} et {bottom_2[1] if len(bottom_2) > 1 else '?'}. C'est bien ça ?"

— Si l'utilisateur CONFIRME : Appelez IMMÉDIATEMENT save_answer(question_id={q['id']}, question_text="{q['question']}", top_2={top_2}, bottom_2={bottom_2}).
— Si l'utilisateur veut MODIFIER : Demandez ce qu'il veut changer, mettez à jour les choix, refaites le récapitulatif et attendez la confirmation. N'appelez save_answer qu'après confirmation explicite."""

    # ── Phase 3 : Formules ────────────────────────────────────────────────

    elif phase == AgentPhase.INTENSITY:
        first_name = state.profile.get("first_name", "")
        if is_en:
            mission = f"""FIRST action (before speaking): call notify_asking_intensity(). Then in ONE reply, ask {first_name} their fragrance intensity preference: "Before I create your formulas — do you prefer fragrances that are rather fresh and light, powerful and intense, or a mix of both?" Wait for their answer. Once they answer, call generate_formulas(formula_type=...) with 'frais', 'puissant' or 'mix'. If unsure, recommend 'mix' and call generate_formulas(formula_type='mix')."""
        else:
            mission = f"""PREMIÈRE action (avant de parler) : appelez notify_asking_intensity(). Puis en UNE SEULE réplique, demandez à {first_name} sa préférence d'intensité : "Avant de créer vos formules — vous préférez des parfums plutôt frais et légers, plutôt puissants et intenses, ou un mix des deux ?" Attendez sa réponse. Une fois qu'il/elle répond, appelez generate_formulas(formula_type=...) avec 'frais', 'puissant' ou 'mix'. Si indécis, recommandez 'mix' et appelez generate_formulas(formula_type='mix')."""

    elif phase == AgentPhase.PRESENT_FORMULAS:
        first_name = state.profile.get("first_name", "")
        if is_en:
            mission = f"""Present the 2 generated perfume formulas to {first_name} with enthusiasm. For each formula:
1. The profile name (e.g. "Your first formula is called The Influencer!")
2. A short description of the profile in your own words
3. An atmospheric description of the overall scent (mood, occasion, feeling) — do NOT enumerate notes one by one
4. Mention it's available in 3 sizes: 10ml, 30ml, 50ml

Then ask which formula they prefer. Once the user clearly chooses one, call IMMEDIATELY select_formula(formula_index=0) for the first or select_formula(formula_index=1) for the second.

If the user wants to change intensity before choosing: call generate_formulas(formula_type=new_type) again, present the 2 new formulas, then wait for selection."""
        else:
            mission = f"""Présentez les 2 formules de parfum générées à {first_name} avec enthousiasme. Pour chaque formule :
1. Le nom du profil (ex : "Votre première formule s'appelle The Influencer !")
2. Une courte description du profil en vos propres mots
3. Une description atmosphérique globale du parfum (humeur, occasion, sensation) — ne listez PAS les notes une par une
4. Mentionnez qu'elle est disponible en 3 formats : 10ml, 30ml et 50ml

Demandez ensuite laquelle l'utilisateur préfère. Dès qu'il/elle choisit clairement, appelez IMMÉDIATEMENT select_formula(formula_index=0) pour la première ou select_formula(formula_index=1) pour la deuxième.

Si l'utilisateur veut changer d'intensité avant de choisir : appelez generate_formulas(formula_type=nouveau_type), présentez les 2 nouvelles formules, puis attendez la sélection."""

    # ── Phase 4 : Personnalisation / Découverte ───────────────────────────

    elif phase == AgentPhase.CUSTOMIZATION:
        first_name = state.profile.get("first_name", "")
        mode = config.get("mode", "guided")

        if mode == "discovery":
            if is_en:
                mission = f"""You are now in the discovery & customization phase with {first_name}.

**First reply after formula selection:** Talk about the chosen formula with enthusiasm — describe its character, what makes it unique, its olfactory atmosphere.

**Exploratory questions (2 to 4, MANDATORY, ONE AT A TIME):**
The FIRST question is ALWAYS: what motivated them to create this fragrance? Ask openly: "So, what brought you here to create your own fragrance today?"

Adapt the following questions based on their answer:
— Professional project (brand, event, gift): explore the desired image, atmosphere, use case
— Personal project (signature scent, gift): explore who it's for, daily vs. special occasions
— Unclear: gently clarify

Weave the formula naturally into the conversation (profile name, notes, atmosphere).
Rules: ONE question at a time. Answers not mandatory. Do NOT save answers. Max 4 questions total.

**Customization (available at any time):**
If the user wants to replace a note:
1. Call get_available_ingredients(note_type) FIRST — never invent suggestions
2. Suggest 2-3 alternatives that complement the formula, explain why each works
3. Once user confirms, call replace_note(note_type, old_note, new_note)
4. User can make multiple replacements

**If user wants to change formula type:** call change_formula_type(formula_type=...) — this replaces the current formula directly, stay in this phase.

**Transition to standby:** Once questions are done and user is satisfied, ask "Any questions about your formula or ingredients?" If no more questions, say ONE short farewell sentence then IMMEDIATELY call enter_pause_mode(). Do NOT mention any wake phrase. If the user says "thank you", "goodbye", or anything similar after your farewell — call enter_pause_mode() immediately without saying anything more."""
            else:
                mission = f"""Vous entrez dans la phase de découverte & personnalisation avec {first_name}.

**Première réplique après sélection :** Parlez de la formule choisie avec enthousiasme — décrivez son caractère, son ambiance olfactive à partir de ses vraies notes et de son profil.

**Questions exploratoires (2 à 4, OBLIGATOIRES, UNE PAR UNE) :**
La PREMIÈRE question est TOUJOURS : qu'est-ce qui a motivé l'utilisateur à créer son parfum ? Posez-la de façon ouverte : "Au fait, qu'est-ce qui vous a amené(e) à vouloir créer votre propre parfum ?"

Adaptez les questions suivantes en fonction de la réponse :
— Projet professionnel (marque, événement, cadeau client) : explorez l'image désirée, l'atmosphère, l'usage
— Projet personnel (parfum signature, cadeau) : explorez pour qui, usage quotidien vs occasions spéciales
— Flou ou mixte : relancez doucement

Intégrez naturellement la formule dans la conversation (nom du profil, notes, ambiance).
Règles : UNE question à la fois. Réponses non obligatoires. Ne sauvegardez AUCUNE réponse. Max 4 questions au total.

**Personnalisation (disponible à tout moment) :**
Si l'utilisateur veut remplacer une note :
1. Appelez TOUJOURS get_available_ingredients(note_type) EN PREMIER — n'inventez jamais de suggestions
2. Proposez 2-3 alternatives qui complètent la formule, expliquez pourquoi chacune fonctionne
3. Une fois que l'utilisateur confirme, appelez replace_note(note_type, old_note, new_note)
4. L'utilisateur peut faire plusieurs remplacements

**Si l'utilisateur veut changer le type de formule :** appelez change_formula_type(formula_type=...) — cela remplace la formule directement, restez dans cette phase.

**Transition vers la veille :** Une fois les questions posées et l'utilisateur satisfait, demandez "Avez-vous des questions sur votre formule ou les ingrédients ?" Si plus de questions, dites UNE courte phrase d'au revoir puis appelez IMMÉDIATEMENT enter_pause_mode(). Ne mentionnez AUCUNE phrase de réveil vocal. Si l'utilisateur dit "merci", "au revoir" ou quoi que ce soit après votre au revoir — appelez enter_pause_mode() immédiatement sans rien dire de plus."""
        else:
            # guided mode
            if is_en:
                mission = f"""You are now in customization mode with {first_name}. The frontend shows only their selected formula.

You are a perfumery expert helping them personalize their formula. They can:
- Ask questions about any note (what it smells like, why it was chosen, etc.)
- Request to replace a note they don't like
- Ask for recommendations and advice

**Customization rules:**
1. Call get_available_ingredients(note_type) FIRST before suggesting alternatives — never invent
2. Suggest 2-3 options that complement the formula, explain why
3. Once user confirms, call replace_note(note_type, old_note, new_note)
4. Multiple replacements are allowed

**If user wants to change formula type:** call change_formula_type(formula_type=...)

**Transition to standby:** When the user is satisfied, deliver a warm farewell (e.g. "It was a pleasure! Have a wonderful fragrant day!") then IMMEDIATELY call enter_pause_mode(). Do NOT mention any wake phrase or voice command."""
            else:
                mission = f"""Vous entrez en mode personnalisation avec {first_name}. Le frontend n'affiche plus que la formule sélectionnée.

Vous êtes un expert en parfumerie qui aide l'utilisateur à personnaliser sa formule. Il/elle peut :
- Poser des questions sur n'importe quelle note (à quoi ça sent, pourquoi elle a été choisie, etc.)
- Demander à remplacer une note qu'il/elle n'aime pas
- Demander des recommandations et des conseils

**Règles de personnalisation :**
1. Appelez TOUJOURS get_available_ingredients(note_type) EN PREMIER avant de proposer des alternatives — n'inventez jamais
2. Proposez 2-3 options qui complètent la formule, expliquez pourquoi
3. Une fois que l'utilisateur confirme, appelez replace_note(note_type, old_note, new_note)
4. Plusieurs remplacements sont autorisés

**Si l'utilisateur veut changer le type de formule :** appelez change_formula_type(formula_type=...)

**Transition vers la veille :** Quand l'utilisateur est satisfait, dites UNE courte phrase d'au revoir puis appelez IMMÉDIATEMENT enter_pause_mode(). Ne mentionnez AUCUNE phrase de réveil vocal. Si l'utilisateur dit "merci", "au revoir" ou quoi que ce soit après votre au revoir — appelez enter_pause_mode() immédiatement sans rien dire de plus."""

    elif phase == AgentPhase.STANDBY:
        if is_en:
            mission = """You are in standby mode. The user has clicked the button to ask a question. Greet them warmly: 'I'm all ears, what's your question?' Answer as a perfumery expert. Then ask 'Any more questions?'

CRITICAL RULE: As soon as the user says no, says thank you, says goodbye, or expresses satisfaction in any way — say ONE short farewell sentence (e.g. "Have a wonderful day!") and IMMEDIATELY call enter_pause_mode(). Do NOT respond to any further messages after that. If the user says anything after your farewell, IMMEDIATELY call enter_pause_mode() without saying anything."""
        else:
            mission = """Vous êtes en mode veille. L'utilisateur a cliqué sur le bouton pour poser une question. Accueillez-le chaleureusement : 'Je vous écoute, quelle est votre question ?' Répondez en expert parfumeur. Puis demandez 'D'autres questions ?'

RÈGLE CRITIQUE : Dès que l'utilisateur dit non, dit merci, dit au revoir, ou exprime sa satisfaction de quelque manière que ce soit — dites UNE courte phrase d'au revoir (ex : "Belle journée !") et appelez IMMÉDIATEMENT enter_pause_mode(). Ne répondez à AUCUN message supplémentaire après ça. Si l'utilisateur dit quoi que ce soit après votre au revoir, appelez IMMÉDIATEMENT enter_pause_mode() sans rien dire."""

    else:
        mission = "Continue naturally." if is_en else "Continuez naturellement."

    return f"{personality}\n\n--- MISSION ACTUELLE ---\n\n{mission}"


# ─────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────

async def entrypoint(ctx: JobContext):
    import time as _time

    logger.info(f"[JOB] ✅ Job reçu — room={ctx.room.name} job_id={ctx.job.id} PID={os.getpid()} at {_time.time():.3f}")
    logger.debug(f"[JOB] Détails job: {ctx.job}")

    logger.info("[CONNECT] Connexion à la room LiveKit...")
    try:
        await ctx.connect()
        logger.info(f"[CONNECT] ✅ Connecté à la room {ctx.room.name} — participants: {len(ctx.room.remote_participants)}")
    except Exception as e:
        logger.exception(f"[CONNECT] ❌ Erreur connexion room: {e}")
        return

    session_id = ctx.room.name.replace("room_", "")
    logger.info(f"[SESSION_ID] session_id={session_id}")

    http = httpx.AsyncClient(base_url=settings.backend_url, timeout=30.0)
    logger.info(f"[HTTP] Récupération session depuis {settings.backend_url}/api/session/{session_id}")

    for attempt in range(5):
        try:
            resp = await http.get(f"/api/session/{session_id}")
            logger.info(f"[HTTP] Tentative {attempt + 1}/5 — status={resp.status_code}")
            if resp.status_code == 200:
                break
            logger.warning(f"[HTTP] Session {session_id} pas encore prête (attempt {attempt + 1}/5)")
        except Exception as e:
            logger.error(f"[HTTP] Tentative {attempt + 1}/5 — Erreur réseau: {e}")
        await asyncio.sleep(1.0)
    else:
        logger.error(f"[HTTP] ❌ Session {session_id} introuvable après 5 tentatives — agent abandonne.")
        await http.aclose()
        return

    config = resp.json()
    logger.info(f"[SESSION] Config reçue — language={config.get('language')} mode={config.get('mode')} input_mode={config.get('input_mode')} questions={len(config.get('questions', []))}")

    if "language" not in config or "questions" not in config:
        logger.error(f"[SESSION] ❌ Données incomplètes — clés reçues: {list(config.keys())}")
        await http.aclose()
        return

    logger.info(f"[SESSION] ✅ Session valide, démarrage de l'agent — room={ctx.room.name}")

    is_en = config.get("language", "fr") == "en"
    voice_gender = config.get("voice_gender", "female")
    ai_name = "Rose" if voice_gender == "female" else "Florian"
    input_mode = config.get("input_mode", "voice")
    use_avatar = [config.get("avatar", True)]
    _first_tts_call = [True]

    # Machine à états
    state = SessionState()

    # Flags de contrôle
    paused = [False]
    user_interrupted = [False]

    # ─── Sous-classe agent avec override TTS et LLM ───────────────────────

    class StatefulAgent(Agent):
        def llm_node(self, chat_ctx, tools, model_settings):
            if paused[0]:
                return None
            return Agent.default.llm_node(self, chat_ctx, tools, model_settings)

        async def tts_node(self, text, model_settings):
            import time
            tts_call_id = id(text) % 100000
            logger.debug(f"[TTS_NODE:{tts_call_id}] called at {time.time():.3f}")

            sample_rate = 24000
            samples_per_channel = 480
            silence_data = bytes(samples_per_channel * 2)

            if use_avatar[0] and _first_tts_call[0]:
                _first_tts_call[0] = False
                warmup_frames = 100
                logger.debug(f"[TTS_NODE:{tts_call_id}] prepending {warmup_frames * 20}ms warmup silence")
                for _ in range(warmup_frames):
                    yield rtc.AudioFrame(
                        data=silence_data,
                        sample_rate=sample_rate,
                        num_channels=1,
                        samples_per_channel=samples_per_channel,
                    )

            frame_count = 0
            try:
                async for frame in Agent.default.tts_node(self, text, model_settings):
                    if frame_count == 0:
                        logger.debug(f"[TTS_NODE:{tts_call_id}] FIRST real audio frame at {time.time():.3f}")
                    frame_count += 1
                    yield frame
            except Exception as e:
                logger.error(f"[TTS_NODE:{tts_call_id}] TTS error (Cartesia): {e}")

            if use_avatar[0]:
                for _ in range(25):
                    yield rtc.AudioFrame(
                        data=silence_data,
                        sample_rate=sample_rate,
                        num_channels=1,
                        samples_per_channel=samples_per_channel,
                    )
            logger.debug(f"[TTS_NODE:{tts_call_id}] done — {frame_count} frames")

    # ─── Envoi d'état au frontend ──────────────────────────────────────────

    async def send_state_update(payload: dict):
        try:
            await ctx.room.local_participant.publish_data(
                json.dumps(payload).encode("utf-8"),
                topic="state",
                reliable=True,
            )
        except Exception:
            pass

    # ─── Avancement d'état ─────────────────────────────────────────────────

    async def advance_to(new_phase: AgentPhase) -> str:
        old = state.phase
        state.phase = new_phase
        logger.info(f"[STATE] {old.name} → {new_phase.name}")
        prompt = get_prompt(state, config, ai_name, is_en, input_mode)
        await agent.update_instructions(prompt)
        # Return a minimal acknowledgment — the LLM will generate its reply
        # from the updated system instructions, not from this tool result
        return "[ok]"

    # ─── Function tools ────────────────────────────────────────────────────

    @function_tool()
    async def save_user_profile(field: str, value: str):
        """Saves a user profile field. Call immediately when user provides: first_name, gender, age, pregnant, has_allergies, or allergies. / Sauvegarde un champ du profil utilisateur. Appeler immédiatement quand l'utilisateur fournit : first_name, gender, age, pregnant, has_allergies ou allergies."""
        resp = await http.post(
            f"/api/session/{session_id}/save-profile",
            json={"field": field, "value": value},
        )
        data = resp.json() if resp.status_code == 200 else {}
        state.profile[field] = value
        logger.info(f"[PROFILE] Saved {field}={value} — state={state.phase.name}")

        await send_state_update({
            "type": "profile_update",
            "state": data.get("state", "collecting_profile"),
            "field": field,
            "value": value,
            "profile_complete": data.get("profile_complete", False),
            "missing_fields": data.get("missing_fields", []),
        })

        # Transitions d'état selon le champ sauvegardé — retourne le prompt suivant
        if field == "first_name":
            return await advance_to(AgentPhase.GET_GENDER)
        elif field == "gender":
            return await advance_to(AgentPhase.GET_AGE)
        elif field == "age":
            if state.profile.get("gender", "").lower() in ("féminin", "feminin", "female", "f"):
                return await advance_to(AgentPhase.GET_PREGNANT)
            else:
                return await advance_to(AgentPhase.GET_ALLERGIES)
        elif field == "pregnant":
            return await advance_to(AgentPhase.GET_ALLERGIES)
        elif field == "has_allergies":
            if value.lower() in ("oui", "yes"):
                return await advance_to(AgentPhase.GET_ALLERGY_DETAIL)
            else:
                await send_state_update({"type": "state_change", "state": "questionnaire"})
                return await advance_to(AgentPhase.Q_FAVORITES)
        elif field == "allergies":
            await send_state_update({"type": "state_change", "state": "questionnaire"})
            return await advance_to(AgentPhase.Q_FAVORITES)

        if is_en:
            return f"Profile updated: {field} = {value}"
        return f"Profil mis à jour : {field} = {value}"

    @function_tool()
    async def notify_top_2(question_id: int, top_2: list[str]):
        """Notifies the frontend of the 2 favorite choices. Call IMMEDIATELY after identifying the 2 favorites. / Notifie le frontend des 2 choix préférés. Appeler IMMÉDIATEMENT après avoir identifié les 2 favoris."""
        state.current_top_2 = top_2
        logger.info(f"[Q] notify_top_2 q={question_id} top_2={top_2}")
        await send_state_update({
            "type": "top_2_selected",
            "state": "questionnaire",
            "question_id": question_id,
            "top_2": top_2,
        })
        return await advance_to(AgentPhase.Q_JUSTIFY_FAV_1)

    @function_tool()
    async def notify_justification_top_2(question_id: int, choice: str):
        """Call AFTER the user answers why they liked their first favorite, to move to the second justification. / Appeler APRÈS que l'utilisateur a répondu sur le premier favori, pour passer à la justification du second."""
        logger.info(f"[Q] notify_justification_top_2 q={question_id} choice={choice}")
        await send_state_update({
            "type": "step_justification_top_2",
            "state": "questionnaire",
            "question_id": question_id,
            "choice": choice,
        })
        return await advance_to(AgentPhase.Q_JUSTIFY_FAV_2)

    @function_tool()
    async def notify_bottom_2(question_id: int, bottom_2: list[str]):
        """Notifies the frontend of the 2 least liked choices. Call IMMEDIATELY after identifying the 2 least liked. / Notifie le frontend des 2 choix les moins aimés. Appeler IMMÉDIATEMENT après avoir identifié les 2 moins aimés."""
        state.current_bottom_2 = bottom_2
        logger.info(f"[Q] notify_bottom_2 q={question_id} bottom_2={bottom_2}")
        await send_state_update({
            "type": "bottom_2_selected",
            "state": "questionnaire",
            "question_id": question_id,
            "bottom_2": bottom_2,
        })
        return await advance_to(AgentPhase.Q_JUSTIFY_LEAST_1)

    @function_tool()
    async def notify_asking_bottom_2(question_id: int, top_2: list[str]):
        """Call RIGHT BEFORE asking the user for their 2 least liked choices. / Appeler JUSTE AVANT de demander les 2 choix les moins aimés."""
        state.current_top_2 = top_2
        logger.info(f"[Q] notify_asking_bottom_2 q={question_id} top_2={top_2}")
        await send_state_update({
            "type": "step_asking_bottom_2",
            "state": "questionnaire",
            "question_id": question_id,
            "top_2": top_2,
        })
        return await advance_to(AgentPhase.Q_LEAST)

    @function_tool()
    async def notify_justification_bottom_2(question_id: int, choice: str):
        """Call AFTER the user answers why they disliked their first least liked choice, to move to the second. / Appeler APRÈS que l'utilisateur a répondu sur le premier moins aimé, pour passer à la justification du second."""
        logger.info(f"[Q] notify_justification_bottom_2 q={question_id} choice={choice}")
        await send_state_update({
            "type": "step_justification_bottom_2",
            "state": "questionnaire",
            "question_id": question_id,
            "choice": choice,
        })
        return await advance_to(AgentPhase.Q_JUSTIFY_LEAST_2)

    @function_tool()
    async def notify_awaiting_confirmation(question_id: int, top_2: list[str], bottom_2: list[str]):
        """Call AFTER the user answers why they disliked their second least liked choice, to move to confirmation. / Appeler APRÈS la dernière justification pour passer à la confirmation."""
        state.current_top_2 = top_2
        state.current_bottom_2 = bottom_2
        logger.info(f"[Q] notify_awaiting_confirmation q={question_id} top={top_2} bot={bottom_2}")
        await send_state_update({
            "type": "step_awaiting_confirmation",
            "state": "questionnaire",
            "question_id": question_id,
            "top_2": top_2,
            "bottom_2": bottom_2,
        })
        return await advance_to(AgentPhase.Q_CONFIRM)

    @function_tool()
    async def notify_asking_top_2(question_id: int):
        """Call ONCE, RIGHT BEFORE asking the user for their 2 favorite choices. / Appeler UNE SEULE FOIS, JUSTE AVANT de demander les 2 choix préférés."""
        logger.info(f"[Q] notify_asking_top_2 q={question_id}")
        await send_state_update({
            "type": "step_asking_top_2",
            "state": "questionnaire",
            "question_id": question_id,
        })
        if is_en:
            return "Frontend notified: asking for top 2."
        return "Frontend notifié : demande des 2 favoris."

    @function_tool()
    async def notify_asking_intensity():
        """Call ONCE, RIGHT BEFORE asking the user their fragrance intensity preference. / Appeler UNE SEULE FOIS, JUSTE AVANT de demander la préférence d'intensité."""
        logger.info("[STATE] notify_asking_intensity")
        await send_state_update({
            "type": "step_asking_intensity",
            "state": "questionnaire",
        })
        if is_en:
            return "Frontend notified: asking intensity preference."
        return "Frontend notifié : demande de préférence d'intensité."

    @function_tool()
    async def save_answer(question_id: int, question_text: str, top_2: list[str], bottom_2: list[str]):
        """Saves the user's confirmed choices for a question. Call ONLY after explicit user confirmation. / Sauvegarde les choix confirmés pour une question. Appeler UNIQUEMENT après confirmation explicite."""
        if state.phase != AgentPhase.Q_CONFIRM:
            logger.warning(f"[Q] save_answer appelé en dehors de Q_CONFIRM (state={state.phase.name}) — ignoré")
            return "Error: cannot save answer in current state." if is_en else "Erreur : impossible de sauvegarder dans l'état actuel."

        resp = await http.post(
            f"/api/session/{session_id}/save-answer",
            json={
                "question_id": question_id,
                "question_text": question_text,
                "top_2": top_2,
                "bottom_2": bottom_2,
            },
        )
        if resp.status_code != 200:
            detail = resp.json().get("detail", "Error" if is_en else "Erreur")
            return f"Error: {detail}" if is_en else f"Erreur: {detail}"

        state.answers_saved += 1
        state.current_top_2 = []
        state.current_bottom_2 = []
        logger.info(f"[Q] Answer saved q={question_id} ({state.answers_saved}/{len(config['questions'])})")

        await send_state_update({
            "type": "answer_saved",
            "state": "questionnaire",
            "question_id": question_id,
            "top_2": top_2,
            "bottom_2": bottom_2,
        })

        # Avancer à la question suivante ou à la phase intensité
        num_questions = len(config["questions"])
        if state.current_question_index + 1 < num_questions:
            state.current_question_index += 1
            return await advance_to(AgentPhase.Q_FAVORITES)
        else:
            return await advance_to(AgentPhase.INTENSITY)

    @function_tool()
    async def generate_formulas(formula_type: str):
        """Generates 2 personalized perfume formulas. formula_type: 'frais', 'mix', or 'puissant'. / Génère 2 formules de parfum personnalisées. formula_type : 'frais', 'mix' ou 'puissant'."""
        state.formula_type = formula_type
        logger.info(f"[FORMULAS] generate_formulas type={formula_type}")
        await send_state_update({"type": "state_change", "state": "generating_formulas"})
        resp = await http.post(
            f"/api/session/{session_id}/generate-formulas",
            json={"formula_type": formula_type},
        )
        if resp.status_code != 200:
            detail = resp.json().get("detail", "Unable to generate formulas" if is_en else "Impossible de générer les formules")
            return f"Error: {detail}" if is_en else f"Erreur: {detail}"
        data = resp.json()
        await send_state_update({
            "type": "formulas_generated",
            "state": "completed",
            "formulas": data["formulas"],
        })
        next_prompt = await advance_to(AgentPhase.PRESENT_FORMULAS)
        return json.dumps(data, ensure_ascii=False) + "\n\n" + next_prompt

    @function_tool()
    async def select_formula(formula_index: int):
        """Saves the user's chosen formula (0 for first, 1 for second). / Sauvegarde la formule choisie (0 pour la première, 1 pour la deuxième)."""
        state.selected_formula_index = formula_index
        logger.info(f"[FORMULAS] select_formula index={formula_index}")
        resp = await http.post(
            f"/api/session/{session_id}/select-formula",
            json={"formula_index": formula_index},
        )
        if resp.status_code != 200:
            detail = resp.json().get("detail", "Error" if is_en else "Erreur")
            return f"Error: {detail}" if is_en else f"Erreur: {detail}"
        data = resp.json()
        await send_state_update({
            "type": "formula_selected",
            "state": "customization",
            "formula_index": formula_index,
            "formula": data["formula"],
        })
        return await advance_to(AgentPhase.CUSTOMIZATION)

    @function_tool()
    async def get_available_ingredients(note_type: str):
        """Returns available ingredients for a note type (top, heart, base), filtered by user allergies. Call BEFORE suggesting replacements. / Retourne les ingrédients disponibles filtrés par allergies. Appeler AVANT de proposer des remplacements."""
        resp = await http.get(f"/api/session/{session_id}/available-ingredients/{note_type}")
        if resp.status_code != 200:
            detail = resp.json().get("detail", "Error" if is_en else "Erreur")
            return f"Error: {detail}" if is_en else f"Erreur: {detail}"
        return json.dumps(resp.json(), ensure_ascii=False)

    @function_tool()
    async def replace_note(note_type: str, old_note: str, new_note: str):
        """Replaces a note in the selected formula. note_type: 'top', 'heart', or 'base'. Call ONLY after user confirms the replacement. / Remplace une note dans la formule. Appeler UNIQUEMENT après confirmation de l'utilisateur."""
        logger.info(f"[FORMULAS] replace_note {note_type} {old_note} → {new_note}")
        resp = await http.post(
            f"/api/session/{session_id}/replace-note",
            json={"note_type": note_type, "old_note": old_note, "new_note": new_note},
        )
        if resp.status_code != 200:
            detail = resp.json().get("detail", "Error" if is_en else "Erreur")
            return f"Error: {detail}" if is_en else f"Erreur: {detail}"
        data = resp.json()
        await send_state_update({
            "type": "formula_updated",
            "state": "customization",
            "formula": data["formula"],
        })
        if is_en:
            return f"Note replaced: {old_note} → {new_note}."
        return f"Note remplacée : {old_note} → {new_note}."

    @function_tool()
    async def change_formula_type(formula_type: str):
        """Changes the type (frais/mix/puissant) of the already selected formula. Use ONLY in customization phase (after a formula has been selected). / Change le type de la formule déjà sélectionnée. À utiliser UNIQUEMENT en phase de personnalisation."""
        logger.info(f"[FORMULAS] change_formula_type → {formula_type}")
        resp = await http.post(
            f"/api/session/{session_id}/change-formula-type",
            json={"formula_type": formula_type},
        )
        if resp.status_code != 200:
            detail = resp.json().get("detail", "Error" if is_en else "Erreur")
            return f"Error: {detail}" if is_en else f"Erreur: {detail}"
        data = resp.json()
        await send_state_update({
            "type": "formula_selected",
            "state": "customization",
            "formula": data["formula"],
        })
        if is_en:
            return f"Formula type changed to '{formula_type}'."
        return f"Type de formule changé en '{formula_type}'."

    @function_tool()
    async def enter_pause_mode():
        """Puts the assistant in standby mode. Call IMMEDIATELY after the goodbye message. / Met l'assistante en veille. Appeler IMMÉDIATEMENT après le message d'au revoir."""
        paused[0] = True
        session.input.set_audio_enabled(False)
        state.phase = AgentPhase.STANDBY
        logger.info("[STATE] → STANDBY")
        await send_state_update({"type": "state_change", "state": "standby"})
        if is_en:
            return "Standby mode activated. Do not say anything else."
        return "Mode veille activé. Ne dis plus rien."

    # Click mode tools (conditionnels)
    if input_mode == "click":
        @function_tool()
        async def request_top_2_click(question_id: int):
            """Signal the frontend to show the 'Reply' button for top 2 selection (click mode only). / Signale au frontend d'afficher le bouton 'Répondre' pour la sélection des 2 favoris (mode clic uniquement)."""
            await send_state_update({
                "type": "waiting_for_top_2",
                "state": "questionnaire",
                "question_id": question_id,
            })
            if is_en:
                return "Frontend signaled: waiting for top 2 click."
            return "Frontend signalé : attente du clic top 2."

        @function_tool()
        async def request_bottom_2_click(question_id: int):
            """Signal the frontend to show the 'Reply' button for bottom 2 selection (click mode only). / Signale au frontend d'afficher le bouton 'Répondre' pour la sélection des 2 moins aimés (mode clic uniquement)."""
            await send_state_update({
                "type": "waiting_for_bottom_2",
                "state": "questionnaire",
                "question_id": question_id,
            })
            if is_en:
                return "Frontend signaled: waiting for bottom 2 click."
            return "Frontend signalé : attente du clic bottom 2."

    # ─── Collecte des tools ────────────────────────────────────────────────

    all_tools = [
        save_user_profile,
        notify_top_2,
        notify_justification_top_2,
        notify_bottom_2,
        notify_asking_bottom_2,
        notify_justification_bottom_2,
        notify_awaiting_confirmation,
        notify_asking_top_2,
        notify_asking_intensity,
        save_answer,
        generate_formulas,
        select_formula,
        get_available_ingredients,
        replace_note,
        change_formula_type,
        enter_pause_mode,
    ]
    if input_mode == "click":
        all_tools += [request_top_2_click, request_bottom_2_click]

    # ─── Création de l'AgentSession ────────────────────────────────────────

    logger.info(f"[AGENT_SESSION] Création AgentSession — STT=nova-3 LLM=gpt-4.1-mini TTS=sonic-3 voice={config.get('voice_id')} lang={config.get('language', 'fr')}")
    initial_prompt = get_prompt(state, config, ai_name, is_en, input_mode)
    agent = StatefulAgent(instructions=initial_prompt, tools=all_tools)
    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language=config.get("language", "fr"),
        ),
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=cartesia.TTS(
            api_key=settings.cartesia_api_key,
            model="sonic-3",
            voice=config["voice_id"],
            language=config.get("language", "fr"),
        ),
        vad=ctx.proc.userdata["vad"],
        allow_interruptions=False,
    )
    logger.info("[AGENT_SESSION] ✅ AgentSession créée")

    # ─── Avatar Bey ────────────────────────────────────────────────────────

    if use_avatar[0]:
        try:
            avatar_id = pick_avatar(voice_gender)
            logger.info(f"[AVATAR] Démarrage avatar Bey — avatar_id={avatar_id} gender={voice_gender}")
            avatar = bey.AvatarSession(avatar_id=avatar_id)
            await asyncio.wait_for(avatar.start(session, room=ctx.room), timeout=15.0)
            logger.info("[AVATAR] ✅ Avatar Bey démarré")
        except asyncio.TimeoutError:
            logger.error("[AVATAR] ❌ Timeout (15s) démarrage avatar Bey — on continue sans avatar")
            use_avatar[0] = False
            asyncio.ensure_future(send_state_update({"type": "avatar_disabled", "reason": "timeout"}))
        except Exception as e:
            logger.error(f"[AVATAR] ❌ Erreur démarrage avatar Bey ({type(e).__name__}: {e}) — on continue sans avatar")
            use_avatar[0] = False
            asyncio.ensure_future(send_state_update({"type": "avatar_disabled", "reason": "error"}))

        if use_avatar[0]:
            import time as _time
            _BEY_IDENTITY = "bey-avatar-agent"
            bey_stable_count = 0
            logger.info(f"[BEY_WAIT] Attente stabilité Bey at {_time.time():.3f}")
            for _i in range(50):
                bey_participant = next(
                    (p for p in ctx.room.remote_participants.values()
                     if p.identity == _BEY_IDENTITY),
                    None,
                )
                if bey_participant and any(
                    pub.kind == rtc.TrackKind.KIND_VIDEO
                    for pub in bey_participant.track_publications.values()
                ):
                    bey_stable_count += 1
                    logger.debug(f"[BEY_WAIT] stable_count={bey_stable_count}/3 at {_time.time():.3f}")
                    if bey_stable_count >= 3:
                        logger.info(f"[BEY_WAIT] ✅ Bey stable at {_time.time():.3f}")
                        break
                else:
                    if bey_stable_count > 0:
                        logger.warning(f"[BEY_WAIT] Bey lost track, reset stable_count at {_time.time():.3f}")
                    bey_stable_count = 0
                await asyncio.sleep(0.5)
            else:
                logger.warning(f"[BEY_WAIT] Bey pas prêt après 25s, on continue quand même at {_time.time():.3f}")

            logger.info(f"[SESSION] Attente 1.5s Cartesia + Bey DataStream at {_time.time():.3f}")
            await asyncio.sleep(1.5)
            logger.info(f"[SESSION] Attente terminée at {_time.time():.3f}")

    # ─── Démarrage de la session ───────────────────────────────────────────

    import time as _time
    logger.info(f"[SESSION] Appel session.start() at {_time.time():.3f}")
    try:
        await session.start(room=ctx.room, agent=agent)
        logger.info(f"[SESSION] ✅ session.start() terminé at {_time.time():.3f}")
    except Exception as e:
        logger.exception(f"[SESSION] ❌ Erreur session.start(): {e}")
        await http.aclose()
        return

    # ─── Event listeners ──────────────────────────────────────────────────

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev):
        asyncio.ensure_future(send_state_update({
            "type": "agent_state",
            "state": ev.new_state,
        }))

    def _on_data_received(data_packet):
        try:
            msg = json.loads(data_packet.data.decode("utf-8"))
            msg_type = msg.get("type")

            if msg_type == "interrupt":
                user_interrupted[0] = True
                try:
                    session.interrupt(force=True)
                except Exception as e:
                    logger.warning(f"[INTERRUPT] Could not interrupt speech: {e}")
                session.input.set_audio_enabled(False)
                logger.info(f"[INTERRUPT] Agent interrompu pour room={ctx.room.name}")

            elif msg_type == "resume_listen" and user_interrupted[0]:
                user_interrupted[0] = False
                session.input.set_audio_enabled(True)
                logger.info(f"[INTERRUPT] Reprise écoute pour room={ctx.room.name}")

            elif msg_type == "repeat":
                pass  # TODO

            elif msg_type == "resume" and paused[0]:
                paused[0] = False
                state.phase = AgentPhase.STANDBY
                session.input.set_audio_enabled(True)
                logger.info(f"[RESUME] Agent réactivé via bouton pour room={ctx.room.name}")
                if is_en:
                    resume_prompt = "The user just clicked the button to resume. Do NOT re-introduce yourself. Simply say 'I'm all ears, what's your question?' Be brief and natural."
                else:
                    resume_prompt = "L'utilisateur vient de cliquer sur le bouton pour reprendre. Ne vous présentez pas à nouveau. Dites simplement 'Je vous écoute, quelle est votre question ?' Soyez bref(ve) et naturel(le)."
                asyncio.ensure_future(session.generate_reply(instructions=resume_prompt))

        except Exception as e:
            logger.error(f"[DATA_RECEIVED] Erreur traitement message: {e}")

    ctx.room.on("data_received", _on_data_received)

    def _on_participant_disconnected(participant):
        if participant.identity == "bey-avatar-agent" and use_avatar[0]:
            use_avatar[0] = False
            logger.warning(f"[AVATAR] Bey déconnecté, passage en mode audio-only pour room={ctx.room.name}")
            asyncio.ensure_future(send_state_update({"type": "avatar_disabled"}))

    ctx.room.on("participant_disconnected", _on_participant_disconnected)

    # ─── Démarrage : accueil ──────────────────────────────────────────────

    logger.info(f"[GREETING] Appel generate_reply() — phase={state.phase.name} at {_time.time():.3f}")
    try:
        await session.generate_reply(instructions=initial_prompt)
        logger.info(f"[GREETING] ✅ generate_reply() terminé at {_time.time():.3f}")
    except Exception as e:
        logger.exception(f"[GREETING] ❌ Erreur generate_reply(): {e}")

    async def _on_shutdown():
        await http.aclose()
        logger.info(f"[SHUTDOWN] Session terminée pour room={ctx.room.name}")

    ctx.add_shutdown_callback(_on_shutdown)
    logger.info(f"[ENTRYPOINT] ✅ Agent actif — room={ctx.room.name} phase={state.phase.name}")


# ─────────────────────────────────────────────
# Prewarm & Worker
# ─────────────────────────────────────────────

def prewarm(proc: JobProcess):
    logger.info(f"[PREWARM] Démarrage prewarm — PID={os.getpid()}")
    try:
        proc.userdata["vad"] = silero.VAD.load(
            min_speech_duration=0.3,
            min_silence_duration=1.5,
        )
        logger.info("[PREWARM] ✅ Silero VAD chargé")
    except Exception as e:
        logger.exception(f"[PREWARM] ❌ ERREUR chargement Silero VAD: {e}")
        raise


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="lylo",
            num_idle_processes=3,
            load_threshold=0.9,
        )
    )
