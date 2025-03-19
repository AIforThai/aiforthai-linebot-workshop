from pydantic_settings import BaseSettings, SettingsConfigDict


class Configs(BaseSettings):
    AIFORTHAI_APIKEY: str
    LINE_CHANNEL_ACCESS_TOKEN: str
    LINE_CHANNEL_SECRET: str

    # Basic NLP VARIABLES
    URL: str
    WAV_FILE: str
    DIR_FILE: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore",str_strip_whitespace=True)
