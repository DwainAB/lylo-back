from livekit import api

from app.config import get_settings


def create_token(identity: str, room: str) -> str:
    settings = get_settings()

    token = api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
    token = token.with_identity(identity)
    token = token.with_grants(
        api.VideoGrants(
            room_join=True,
            room=room,
            can_publish=True,
            can_subscribe=True,
        )
    )

    return token.to_jwt()
