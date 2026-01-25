from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_title: str = "Company File Analyzer (MVP)"
    data_dir: str = "data"
    db_path: str = "data/app.db"

    redis_url: str = "redis://127.0.0.1:6379/0"

    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b"

settings = Settings()
