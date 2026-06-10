"""
src/config.py
─────────────────────────────────────────────────────────────────────────────
Centralised project configuration via pydantic-settings.

All values are read from environment variables (or a .env file).
Import `settings` wherever you need config — never read os.environ directly.

Usage:
    from src.config import settings
    print(settings.data_raw_dir)
"""

from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── CDS / ERA5 ──────────────────────────────────────────────────────────
    cds_api_url: str = Field(
        default="https://cds.climate.copernicus.eu/api/v2",
        description="Copernicus CDS API endpoint",
    )
    cds_api_key: str = Field(
        default="",
        description="CDS API key in format uid:key",
    )

    # ── Data directories ────────────────────────────────────────────────────
    data_raw_dir: Path = Field(default=Path("data/raw"))
    data_interim_dir: Path = Field(default=Path("data/interim"))
    data_processed_dir: Path = Field(default=Path("data/processed"))

    # ── MLflow ──────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field(default="mlruns")

    # ── Risk model ──────────────────────────────────────────────────────────
    risk_haircut_alpha: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Portfolio haircut coefficient (0 = no adjustment, 1 = full)",
    )
    weight_flood: float = Field(default=0.40, ge=0.0, le=1.0)
    weight_heat: float = Field(default=0.35, ge=0.0, le=1.0)
    weight_cyclone: float = Field(default=0.25, ge=0.0, le=1.0)

    # ── ERA5 parameters ─────────────────────────────────────────────────────
    era5_variables: list[str] = Field(
        default=["2m_temperature", "total_precipitation", "10m_wind_speed"]
    )
    era5_year_start: int = Field(default=1990, ge=1950, le=2024)
    era5_year_end: int = Field(default=2023, ge=1950, le=2024)

    # ── Logging ─────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    # ── Validators ──────────────────────────────────────────────────────────
    @field_validator("era5_variables", mode="before")
    @classmethod
    def parse_era5_variables(cls, v: str | list) -> list[str]:
        """Accept comma-separated string or list from env."""
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "Settings":
        total = self.weight_flood + self.weight_heat + self.weight_cyclone
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Hazard weights must sum to 1.0, got {total:.3f}. "
                "Adjust WEIGHT_FLOOD, WEIGHT_HEAT, WEIGHT_CYCLONE in .env"
            )
        return self

    @model_validator(mode="after")
    def year_range_valid(self) -> "Settings":
        if self.era5_year_start >= self.era5_year_end:
            raise ValueError(
                f"ERA5_YEAR_START ({self.era5_year_start}) must be "
                f"< ERA5_YEAR_END ({self.era5_year_end})"
            )
        return self

    def ensure_dirs(self) -> None:
        """Create all data directories if they don't exist."""
        for d in [self.data_raw_dir, self.data_interim_dir, self.data_processed_dir]:
            d.mkdir(parents=True, exist_ok=True)


# Module-level singleton — import this everywhere
settings = Settings()
