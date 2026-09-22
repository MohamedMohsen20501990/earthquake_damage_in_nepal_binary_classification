from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    
    model_config= SettingsConfigDict(env_file=".env")
    
    """tells Pydantic Settings where to look for environment variables
    Think of it as saying:
    "When you create Settings(), also load variables from the .env file."""
    DATABASE_USERNAME: str
    DATABASE_PASSWORD: str
    
settings=Settings()    