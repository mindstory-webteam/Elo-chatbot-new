"""
Configuration settings for the elo Platform.
Supports multiple providers for LLM, embeddings, vector store, database, storage, and cache.
"""
import secrets
from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache
from typing import Literal, Optional, List


class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "elo"
    DEBUG: bool = False
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    SITE_URL: str = "http://localhost:8000"
    
    # ===========================================
    # Security Settings
    # ===========================================
    # CORS - comma-separated list of allowed origins
    CORS_ORIGINS: str = (
        "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000,"
        "http://localhost:8015,http://127.0.0.1:8015,"
        "http://localhost:8012,http://127.0.0.1:8012,"
        # Default HTTP port (80): browsers send Origin without an explicit port
        "http://127.0.0.1,http://localhost"
    )
    CORS_ALLOW_CREDENTIALS: bool = True
    # Match loopback on any port (CORS Origin is exact; http://127.0.0.1 does not match :8012).
    # Set to empty string in .env to disable.
    CORS_ORIGIN_REGEX: str = r"^https?://127\.0\.0\.1(?::\d+)?$"
    
    # Trusted hosts for production
    TRUSTED_HOSTS: str = "localhost,127.0.0.1"
    
    # Security headers
    ENABLE_SECURITY_HEADERS: bool = True
    # Full Content-Security-Policy for non-API HTML (/, /app, static pages). Empty = omit CSP header.
    # Configure in .env — see .env.example.
    CONTENT_SECURITY_POLICY: str = ""
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS_ORIGINS into a list."""
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def cors_origin_regex(self) -> Optional[str]:
        """Optional regex for allowed Origins (e.g. 127.0.0.1 with any port)."""
        if not self.CORS_ORIGIN_REGEX or not str(self.CORS_ORIGIN_REGEX).strip():
            return None
        return str(self.CORS_ORIGIN_REGEX).strip()

    @property
    def content_security_policy_non_api(self) -> Optional[str]:
        """CSP for HTML routes; None when unset so the header is not sent."""
        raw = (self.CONTENT_SECURITY_POLICY or "").strip()
        return raw if raw else None
    
    @property
    def trusted_hosts_list(self) -> List[str]:
        """Parse TRUSTED_HOSTS into a list."""
        if self.TRUSTED_HOSTS == "*":
            return ["*"]
        return [host.strip() for host in self.TRUSTED_HOSTS.split(",") if host.strip()]
    
    # ===========================================
    # LLM Provider Configuration
    # ===========================================
    LLM_PROVIDER: Literal["ollama", "openai", "anthropic", "azure"] = "ollama"
    LLM_MODEL: str = "llama3.1:8b"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 1000
    # Ollama: cap context window for faster attention (omit by setting 0 or use model default if unset)
    OLLAMA_NUM_CTX: int = 4096
    
    # Provider-specific API keys
    OPENAI_API_KEY: Optional[str] = None
    # Point the OpenAI-compatible client at a different provider. Leave unset
    # for OpenAI itself. Examples:
    #   Google Gemini: https://generativelanguage.googleapis.com/v1beta/openai/
    #   Mistral:       https://api.mistral.ai/v1
    #   Groq (chat only, no embeddings): https://api.groq.com/openai/v1
    OPENAI_BASE_URL: Optional[str] = None

    # Embeddings can come from a DIFFERENT provider than chat. Useful because
    # some chat providers (DeepSeek, Groq) have no embeddings endpoint at all.
    # Both fall back to the OPENAI_* values above when unset.
    EMBEDDINGS_API_KEY: Optional[str] = None
    EMBEDDINGS_BASE_URL: Optional[str] = None

    @property
    def embeddings_api_key(self) -> Optional[str]:
        return self.EMBEDDINGS_API_KEY or self.OPENAI_API_KEY

    @property
    def embeddings_base_url(self) -> Optional[str]:
        if self.EMBEDDINGS_API_KEY:
            # A dedicated embeddings key implies a dedicated endpoint; don't
            # silently inherit the chat provider's URL.
            return self.EMBEDDINGS_BASE_URL or None
        return self.EMBEDDINGS_BASE_URL or self.OPENAI_BASE_URL or None
    ANTHROPIC_API_KEY: Optional[str] = None
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_DEPLOYMENT: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # ===========================================
    # Embeddings Provider Configuration
    # ===========================================
    EMBEDDINGS_PROVIDER: Literal["ollama", "openai", "huggingface"] = "huggingface"
    EMBEDDINGS_MODEL: str = "all-MiniLM-L6-v2"
    
    # ===========================================
    # Vector Store Provider Configuration
    # ===========================================
    VECTOR_STORE_PROVIDER: Literal["faiss", "chroma", "pinecone", "qdrant"] = "faiss"
    
    # FAISS settings
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    
    # Chroma settings
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "elo_docs"
    
    # Pinecone settings
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    
    # Qdrant settings
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "elo"
    
    # ===========================================
    # Database Provider Configuration
    # ===========================================
    DATABASE_PROVIDER: Literal["postgresql"] = "postgresql"
    
    # PostgreSQL settings
    # Either set POSTGRES_URL to a full DSN, or set the discrete parts below.
    POSTGRES_URL: Optional[str] = None
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "elo"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    
    # Connection pool
    POSTGRES_POOL_MIN_SIZE: int = 1
    POSTGRES_POOL_MAX_SIZE: int = 10
    POSTGRES_COMMAND_TIMEOUT: int = 60
    
    # Apply pending migrations automatically on startup.
    # Leave enabled for local/dev; disable in production and run
    # `python -m app.database.migrate` as an explicit deploy step.
    POSTGRES_AUTO_MIGRATE: bool = True
    
    @property
    def postgres_dsn(self) -> str:
        """Full PostgreSQL DSN, built from the discrete parts if not given directly."""
        if self.POSTGRES_URL and self.POSTGRES_URL.strip():
            return self.POSTGRES_URL.strip()
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
    
    @property
    def postgres_database_name(self) -> str:
        """Database name, parsed out of POSTGRES_URL when that is what's configured."""
        if self.POSTGRES_URL and self.POSTGRES_URL.strip():
            from urllib.parse import urlparse
            path = urlparse(self.POSTGRES_URL.strip()).path or ""
            return path.lstrip("/") or self.POSTGRES_DB
        return self.POSTGRES_DB
    
    # ===========================================
    # Storage Provider Configuration
    # ===========================================
    STORAGE_PROVIDER: Literal["local", "s3", "gcs"] = "local"
    
    # Local storage settings
    LOCAL_STORAGE_PATH: str = "./data/uploads"
    
    # S3 settings
    S3_BUCKET: Optional[str] = None
    S3_REGION: Optional[str] = None
    S3_ACCESS_KEY: Optional[str] = None
    S3_SECRET_KEY: Optional[str] = None
    
    # GCS settings
    GCS_BUCKET: Optional[str] = None
    GCS_CREDENTIALS_PATH: Optional[str] = None
    
    # ===========================================
    # Cache Provider Configuration
    # ===========================================
    CACHE_PROVIDER: Literal["memory", "redis"] = "memory"
    
    # Redis settings
    REDIS_URL: Optional[str] = None
    CACHE_TTL: int = 300  # Default TTL in seconds
    
    # ===========================================
    # Auth settings
    # ===========================================
    JWT_SECRET: str = "CHANGE-THIS-SECRET-IN-PRODUCTION"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_HOURS: int = 24
    
    # Admin credentials (created on first run if no admin exists)
    # Set ADMIN_PASSWORD to empty string to disable auto-creation
    ADMIN_EMAIL: str = "admin@example.com"
    ADMIN_PASSWORD: str = ""
    
    # Password policy
    MIN_PASSWORD_LENGTH: int = 8
    REQUIRE_PASSWORD_COMPLEXITY: bool = True
    
    @property
    def is_jwt_secret_secure(self) -> bool:
        """Check if JWT secret is properly configured."""
        insecure_defaults = [
            "your-super-secret-key-change-in-production",
            "CHANGE-THIS-SECRET-IN-PRODUCTION",
            "secret",
            "changeme",
            ""
        ]
        return (
            self.JWT_SECRET not in insecure_defaults and
            len(self.JWT_SECRET) >= 32
        )
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.ENVIRONMENT == "production" and not self.DEBUG
    
    # ===========================================
    # RAG settings
    # ===========================================
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    # Fewer chunks = less prompt text = faster LLM (raise for richer answers)
    RETRIEVAL_K: int = 3
    # Vector search returns k * oversample candidates before grading (2 = faster than 4)
    RAG_RETRIEVAL_OVERSAMPLE: int = 2
    # Max characters per retrieved chunk in the chat prompt (lower = faster)
    RAG_CONTEXT_CHUNK_MAX_CHARS: int = 900
    # Recent messages included in prompt; each message truncated for speed
    CHAT_HISTORY_MAX_MESSAGES: int = 4
    CHAT_HISTORY_MESSAGE_MAX_CHARS: int = 350
    # When true, load history, Q&A check, and vector retrieval in parallel (faster RAG path).
    # Set false if Q&A hits are common and you want to skip wasted retrieval work.
    CHAT_SPECULATIVE_PREFETCH: bool = True
    
    # ===========================================
    # Crawler settings
    # ===========================================
    MAX_PAGES: int = 100
    CRAWL_DELAY: float = 1.0
    
    # ===========================================
    # Rate limiting
    # ===========================================
    RATE_LIMIT_REQUESTS: int = 20
    RATE_LIMIT_WINDOW: int = 60
    
    # ===========================================
    # Conversation settings
    # ===========================================
    CONVERSATION_WINDOW_SIZE: int = 10

    # ===========================================
    # Backward Compatibility Aliases
    # ===========================================
    @property
    def POSTGRES_DSN(self) -> str:
        """Uppercase alias for postgres_dsn."""
        return self.postgres_dsn
    
    @property
    def SECRET_KEY(self) -> str:
        """Backward compatibility alias for JWT_SECRET."""
        return self.JWT_SECRET
    
    @property
    def OLLAMA_MODEL(self) -> str:
        """Backward compatibility alias for LLM_MODEL."""
        return self.LLM_MODEL
    
    @property
    def OLLAMA_EMBEDDING_MODEL(self) -> str:
        """Backward compatibility alias for EMBEDDINGS_MODEL."""
        return self.EMBEDDINGS_MODEL
    
    class Config:
        env_file = ".env"
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
