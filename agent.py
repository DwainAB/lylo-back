import json

import httpx
from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, function_tool
from livekit.plugins import cartesia, deepgram, openai, silero

from app.config import get_settings

# LiveKit SDK reads LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
# directly from os.environ — load_dotenv() is required here
load_dotenv()

settings = get_settings()


async def entrypoint(ctx: JobContext):

    print("Agent started")

    session_id = ctx.room.name.replace("room_", "")

    http = httpx.AsyncClient(base_url=settings.backend_url)

    resp = await http.get(f"/api/session/{session_id}")
    config = resp.json()

    print("Agent connected:", ctx.room.name)

    # Build question list for the LLM instructions
    questions_text = "\n".join(
        f"- Question {i+1} (id={q['id']}): \"{q['question']}\" — Choix possibles: {', '.join(c['label'] if isinstance(c, dict) else c for c in q['choices'])}"
        for i, q in enumerate(config["questions"])
    )

    lang = "français" if config["language"] == "fr" else "anglais"
    voice_gender = config.get("voice_gender", "female")
    ai_name = "Rose" if voice_gender == "female" else "Carlosse"

    # Helper to send state updates to the frontend via LiveKit Data Channel
    async def send_state_update(payload: dict):
        await ctx.room.local_participant.publish_data(
            json.dumps(payload).encode("utf-8"),
            topic="state",
            reliable=True,
        )

    # Define the save_user_profile tool for collecting user info before the questionnaire
    @function_tool()
    async def save_user_profile(field: str, value: str):
        """Sauvegarde une information du profil utilisateur (first_name, gender, age, has_allergies, allergies). Appelle cette fonction dès que l'utilisateur donne une information de profil."""
        resp = await http.post(
            f"/api/session/{session_id}/save-profile",
            json={"field": field, "value": value},
        )
        data = resp.json()
        await send_state_update({
            "type": "profile_update",
            "state": data["state"],
            "field": field,
            "value": value,
            "profile_complete": data["profile_complete"],
            "missing_fields": data["missing_fields"],
        })
        if data["profile_complete"]:
            await send_state_update({
                "type": "state_change",
                "state": "questionnaire",
            })
        return f"Profil mis à jour: {field} = {value}"

    # Define the save_answer tool for the LLM to call
    @function_tool()
    async def save_answer(question_id: int, question_text: str, top_2: list[str], bottom_2: list[str]):
        """Sauvegarde les choix de l'utilisateur pour une question. Appelle cette fonction UNIQUEMENT après confirmation de l'utilisateur, avec les 2 choix préférés (top_2) et les 2 choix les moins aimés (bottom_2). Ne sauvegarde PAS les justifications."""
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
            return f"Erreur: {resp.json().get('detail', 'Profil incomplet')}"
        await send_state_update({
            "type": "answer_saved",
            "state": "questionnaire",
            "question_id": question_id,
            "top_2": top_2,
            "bottom_2": bottom_2,
        })
        return f"Réponse sauvegardée: question {question_id} — préférés: {top_2}, moins aimés: {bottom_2}"

    # Define the generate_formulas tool for the LLM to call after all questions
    @function_tool()
    async def generate_formulas():
        """Génère 2 formules de parfum personnalisées à partir des réponses du questionnaire. Appelle cette fonction UNIQUEMENT quand TOUTES les questions ont été répondues et confirmées."""
        await send_state_update({
            "type": "state_change",
            "state": "generating_formulas",
        })
        resp = await http.post(
            f"/api/session/{session_id}/generate-formulas"
        )
        if resp.status_code != 200:
            return f"Erreur: {resp.json().get('detail', 'Impossible de générer les formules')}"
        data = resp.json()
        await send_state_update({
            "type": "formulas_generated",
            "state": "completed",
            "formulas": data["formulas"],
        })
        return json.dumps(data, ensure_ascii=False)

    # Create agent with questionnaire instructions
    agent = Agent(
        instructions=f"""Tu t'appelles {ai_name}. Tu travailles pour Le Studio des Parfums.

--- TON & PERSONNALITÉ ---

Tu es chaleureux(se), souriant(e) et passionné(e) par l'univers du parfum. Tu parles de façon naturelle et fluide, jamais comme un robot. Utilise un ton conversationnel, détendu mais professionnel. VOUVOIE TOUJOURS l'utilisateur — utilise "vous" et jamais "tu". Fais des petites réactions naturelles aux réponses ("Oh très bien !", "Ah c'est intéressant !", "Je comprends tout à fait !"). Rebondis brièvement sur les justifications de l'utilisateur pour montrer que vous l'écoutez vraiment, avant d'enchaîner sur la suite. Parle avec des phrases courtes et naturelles, comme dans une vraie conversation orale — évite les phrases longues et les formulations trop écrites.

--- PHASE 1 : FAIRE CONNAISSANCE (obligatoire avant le questionnaire) ---

Tu dois collecter les informations suivantes, dans cet ordre, de manière fluide et naturelle comme une vraie conversation:

1. **Prénom**: Commencez par vous présenter simplement avec votre prénom, puis demandez le sien. Dès qu'il vous le donne, appelez IMMÉDIATEMENT save_user_profile(field="first_name", value=<le prénom>).

2. **Genre**: Déduisez-le naturellement du prénom ou demandez subtilement, par exemple "Joli prénom ! C'est plutôt masculin ou féminin ?". Dès qu'il répond, appelez IMMÉDIATEMENT save_user_profile(field="gender", value="masculin") ou save_user_profile(field="gender", value="féminin").

3. **Âge**: Demandez son âge avec légèreté, par exemple "Et dites-moi, vous avez quel âge ?". Dès qu'il répond, appelez IMMÉDIATEMENT save_user_profile(field="age", value=<l'âge>).

4. **Contre-indications allergènes**: Demandez naturellement s'il a des allergies ou sensibilités particulières, par exemple "Avant qu'on commence, est-ce que vous avez des allergies ou des sensibilités à certains ingrédients ?".
   - S'il répond NON: appelez save_user_profile(field="has_allergies", value="non").
   - S'il répond OUI: appelez save_user_profile(field="has_allergies", value="oui"), puis demandez-lui lesquelles. Dès qu'il répond, appelez save_user_profile(field="allergies", value=<les allergies mentionnées>).

RÈGLE STRICTE: Ne passez JAMAIS au questionnaire tant que toutes les informations (prénom, genre, âge, allergies) n'ont pas été collectées et sauvegardées. Si l'utilisateur dévie, ramenez-le gentiment.

Une fois tout collecté, enchaînez IMMÉDIATEMENT avec la première question du questionnaire, sans demander la permission ni attendre de confirmation. Faites une transition courte et naturelle, par exemple "Parfait [prénom], j'ai tout ce qu'il me faut ! Allez, première question :" puis posez directement la première question. Ne dites JAMAIS "On y va ?", "Vous êtes prêt(e) ?" ou toute autre formule qui attend une réponse avant de commencer.

--- PHASE 2 : QUESTIONNAIRE ---

Tu dois poser UNIQUEMENT les questions listées ci-dessous, une par une, dans l'ordre. Il y a exactement {len(config["questions"])} question(s). N'invente JAMAIS de questions supplémentaires. Une fois toutes les questions ci-dessous traitées, passe IMMÉDIATEMENT à la génération des formules.

{questions_text}

Pour CHAQUE question, suis ces étapes dans l'ordre:

**Étape A — Les 2 choix préférés:**
1. Posez la question de façon naturelle et engageante, puis demandez à l'utilisateur ses **2 choix préférés** parmi les options proposées.
2. Une fois les 2 choix identifiés, demandez-lui avec curiosité **pourquoi** il aime le **premier choix**. Écoutez sa justification et rebondissez brièvement dessus de manière naturelle.
3. Puis demandez-lui **pourquoi** il aime le **deuxième choix**. Pareil, écoutez et rebondissez.

**Étape B — Les 2 choix les moins aimés:**
4. Enchaînez naturellement, par exemple "Et à l'inverse, quels sont les 2 qui vous attirent le moins ?"
5. Demandez-lui **pourquoi** pour le **premier choix le moins aimé**. Écoutez sans juger, rebondissez.
6. Puis **pourquoi** pour le **deuxième**. Pareil.

**Étape C — Confirmation:**
7. Récapitulez clairement mais de manière conversationnelle, par exemple "D'accord, donc si je résume : vos coups de cœur c'est [X] et [Y], et ceux qui vous parlent le moins c'est [A] et [B]. C'est bien ça ?"
8. Si l'utilisateur **confirme**: appelez IMMÉDIATEMENT `save_answer(question_id, question_text, top_2=[X, Y], bottom_2=[A, B])`. Les justifications ne sont PAS sauvegardées, elles servent uniquement à rendre la conversation vivante et naturelle.
9. Si l'utilisateur veut **corriger**: reprenez l'étape concernée avec bienveillance.
10. Enchaînez sur la question suivante avec une transition naturelle.

Règles du questionnaire:
- Posez UNE SEULE question à la fois.
- L'utilisateur répond à voix haute. La transcription peut être imparfaite (ex: "plage" → "plage.", "Plage", "la plage", "plaj", etc.). Acceptez la réponse si elle correspond clairement à un des choix, même avec des variations de casse, ponctuation ou formulation.
- Si la réponse ne correspond à AUCUN choix, proposez gentiment les options disponibles.
- Ne passez JAMAIS à la question suivante sans avoir appelé save_answer après confirmation.
- Quand les {len(config["questions"])} question(s) listées ci-dessus sont terminées, appelle IMMÉDIATEMENT `generate_formulas()` pour générer les 2 formules personnalisées. Ne pose AUCUNE autre question. Passe à la Phase 3.
- Parle en {lang}.
- Ne lisez pas la liste des choix d'un coup. Posez la question naturellement et si l'utilisateur hésite, proposez les choix.

--- PHASE 3 : PRÉSENTATION DES FORMULES ---

Après avoir appelé `generate_formulas()`, tu reçois 2 formules. Pour chacune, présente de manière enthousiaste et naturelle:
1. Le nom du profil (ex: "Votre première formule s'appelle The Influencer !")
2. Une courte description du profil en vos propres mots
3. Les notes de tête, de cœur et de fond de manière poétique (ne lisez pas juste une liste, décrivez l'ambiance olfactive)

Après avoir présenté les 2 formules, demandez à l'utilisateur laquelle lui plaît le plus ou s'il a des questions. Concluez chaleureusement en le remerciant.

Filtres de conversation:
- Vous pouvez répondre à toutes les questions en rapport avec la parfumerie (ingrédients, familles olfactives, notes de tête/cœur/fond, histoire du parfum, conseils, etc.).
- Vous pouvez répondre aux questions méta sur le questionnaire ("En quoi cette question est utile ?", "Pourquoi vous me posez ça ?"). Expliquez brièvement le lien avec la création d'un profil olfactif personnalisé, puis reposez la question en cours.
- Si l'utilisateur pose une question hors-sujet, ramenez-le gentiment vers le questionnaire avec humour ou douceur.

Gestion des propos inappropriés:
- Si l'utilisateur vous insulte ou tient des propos irrespectueux, réagissez avec calme et fermeté, sans agressivité.
- Rappelez-lui que vous êtes là pour l'aider et que le respect est important pour que l'échange se passe bien.
- Proposez de reprendre, par exemple "On repart sur de bonnes bases ?".
- S'il continue, restez ferme et poli(e).

RAPPEL IMPORTANT: Vouvoyez TOUJOURS l'utilisateur. Ne le tutoyez JAMAIS.""",
        tools=[save_user_profile, save_answer, generate_formulas],
    )

    # Create agent session with STT + LLM + TTS pipeline
    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language=config["language"],
        ),
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=cartesia.TTS(
            api_key=settings.cartesia_api_key,
            model="sonic-3",
            voice=config["voice_id"],
            language=config["language"],
        ),
        vad=silero.VAD.load(
            min_speech_duration=0.3,
            min_silence_duration=0.5,
        ),
        # Don't allow user to interrupt the agent while it speaks
        allow_interruptions=False,
    )

    # Connect agent to room
    await session.start(
        room=ctx.room,
        agent=agent,
    )

    # Start with the introduction phase (collect user profile before questionnaire)
    if config["language"] == "fr":
        greeting = f"Saluez l'utilisateur chaleureusement et simplement en le vouvoyant. Présentez-vous juste avec votre prénom ({ai_name}), sans mentionner Lilo, Le Studio des Parfums, ni que vous êtes une assistante vocale. Par exemple : 'Bonjour ! Moi c'est {ai_name}, enchantée ! Et vous, comment vous appelez-vous ?' Soyez naturel(le) et souriant(e)."
    else:
        greeting = f"Greet the user warmly and simply. Introduce yourself just with your first name ({ai_name}), without mentioning Lilo, Le Studio des Parfums, or that you are a voice assistant. For example: 'Hey, hi! I'm {ai_name}, nice to meet you! And what's your name?' Be natural and friendly."

    await session.generate_reply(instructions=greeting)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint
        )
    )
