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


async def delete_room(room_name: str) -> bool:
    """Delete a LiveKit room. Returns True if successful."""
    settings = get_settings()
    lkapi = api.LiveKitAPI(
        settings.livekit_url,
        settings.livekit_api_key,
        settings.livekit_api_secret,
    )
    try:
        await lkapi.room.delete_room(api.DeleteRoomRequest(room=room_name))
        return True
    except Exception as e:
        print(f"Failed to delete LiveKit room {room_name}: {e}")
        return False
    finally:
        await lkapi.aclose()
