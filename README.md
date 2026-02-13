# Lilo Backend

Backend de l'application Lilo pour Le Studio des Parfums — une experience de creation de parfum personnalisee guidee par un agent vocal IA.

## Architecture

Le projet est compose de 3 services :

- **Backend API** (FastAPI) — Gere les sessions, le profil utilisateur, les reponses au questionnaire et la generation de formules de parfum.
- **Agent vocal** (LiveKit Agents) — Se connecte a une room LiveKit et guide l'utilisateur par la voix a travers un questionnaire olfactif. Utilise Deepgram (STT), OpenAI (LLM) et Cartesia (TTS).
- **Redis** — Stocke l'etat des sessions, les profils et les reponses.

## Flux general

1. Le frontend demarre une session via l'API (`POST /api/session/start`)
2. Le backend cree la session avec les questions et retourne un token LiveKit
3. L'agent vocal rejoint la room LiveKit et commence la conversation
4. L'agent collecte le profil utilisateur (prenom, genre, age, allergies) puis pose les questions du questionnaire
5. Les reponses sont sauvegardees dans Redis via l'API backend
6. Une fois le questionnaire termine, l'agent declenche la generation de 2 formules de parfum personnalisees
7. Les formules sont envoyees au frontend via le Data Channel LiveKit

## Pre-requis

- Python 3.13+
- Docker Desktop (pour Redis)
- Un fichier `.env` a la racine du projet avec les cles API necessaires (LiveKit, Deepgram, Cartesia, OpenAI)

## Lancer le projet en local (developpement)

3 terminaux sont necessaires :

**Terminal 1** — Lancer Redis via Docker Desktop (ou en ligne de commande) :

```bash
docker run -d -p 6379:6379 redis:7-alpine
```

**Terminal 2** — Lancer le serveur API :

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

**Terminal 3** — Lancer l'agent vocal :

```bash
python3 agent.py dev
```

Le backend API tourne sur `http://localhost:8000`. Le flag `--reload` permet le rechargement automatique quand tu modifies le code.

## Build et deploiement (mise en ligne)

### Avec Docker Compose

Docker Compose permet de lancer les 3 services d'un coup (Redis + API + Agent) :

```bash
# Build et lancement
docker compose up -d --build
```

Le flag `-d` lance les conteneurs en arriere-plan, et `--build` force la reconstruction des images.

Pour arreter les services :

```bash
docker compose down
```

### Build de l'image Docker seule

```bash
docker build -t lilo-backend .
```

### Lancer un service individuellement

```bash
# Seulement le backend API
docker run -p 8000:8000 --env-file .env lilo-backend

# Seulement l'agent vocal
docker run --env-file .env lilo-backend python agent.py start
```

## Variables d'environnement

| Variable | Description |
|---|---|
| `LIVEKIT_URL` | URL du serveur LiveKit |
| `LIVEKIT_API_KEY` | Cle API LiveKit |
| `LIVEKIT_API_SECRET` | Secret API LiveKit |
| `DEEPGRAM_API_KEY` | Cle API Deepgram (STT) |
| `CARTESIA_API_KEY` | Cle API Cartesia (TTS) |
| `OPENAI_API_KEY` | Cle API OpenAI (LLM) |
| `VOICE_FR_FEMALE` | ID voix Cartesia FR feminine |
| `VOICE_FR_MALE` | ID voix Cartesia FR masculine |
| `VOICE_EN_FEMALE` | ID voix Cartesia EN feminine |
| `VOICE_EN_MALE` | ID voix Cartesia EN masculine |
| `BACKEND_URL` | URL du backend API (defaut: `http://localhost:8000`) |
| `REDIS_URL` | URL Redis (defaut: `redis://localhost:6379`) |
