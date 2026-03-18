from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LiveKit
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str

    # Deepgram
    deepgram_api_key: str

    # Cartesia
    cartesia_api_key: str

    # Voices
    voice_fr_female: str
    voice_fr_male: str
    voice_en_female: str
    voice_en_male: str

    # OpenAI
    openai_api_key: str

    # Backend
    backend_url: str = "http://localhost:8000"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # SMTP (email sending — OVH MX Plan)
    smtp_host: str = "ssl0.ovh.net"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    internal_email: str = ""

    # Database (MySQL)
    db_host: str
    db_port: int = 3306
    db_name: str
    db_user: str
    db_password: str

    @property
    def voice_mapping(self) -> dict[str, dict[str, str]]:
        return {
            "fr": {"female": self.voice_fr_female, "male": self.voice_fr_male},
            "en": {"female": self.voice_en_female, "male": self.voice_en_male},
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
