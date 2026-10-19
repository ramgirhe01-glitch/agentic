import os
import certifi
from dotenv import load_dotenv

load_dotenv()

from pydantic_settings import BaseSettings,SettingsConfigDict
from pydantic import Field
# --- AUTOMATIC SSL FIX FOR CONDA/PYTHON ENVIRONMENTS ---

ssl_cert = os.environ.get('SSL_CERT_FILE')
if ssl_cert and not os.path.exists(ssl_cert):
    os.environ['SSL_CERT_FILE'] = certifi.where()
elif not ssl_cert:
    os.environ['SSL_CERT_FILE'] = certifi.where()
# --------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env')

    pinecone_api_key: str = Field(default_factory=lambda: os.getenv("PINECONE_API_KEY"," "))
    tavily_api_key: str = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY"," "))
    pinecone_index_name: str = Field(default_factory=lambda: os.getenv("PINECONE_INDEX_NAME"," "))
    ollama_chat_model: str = Field(default_factory=lambda: os.getenv("OLLAMA_CHAT_MODEL"," "))
    ollama_embedding_model: str = Field(default_factory=lambda: os.getenv("OLLAMA_EMBEDDING_MODEL"," "))
    api_token: str = Field(default_factory=lambda: os.getenv("API_TOKEN"," "))
    allowed_origins: str = Field(default_factory=lambda: os.getenv("ALLOWED_ORIGINS").split(","))

settings = Settings()