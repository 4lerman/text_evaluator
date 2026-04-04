from pydantic import BaseModel, Field, field_validator

from app.config import config


class EvaluateRequest(BaseModel):
    """Request model for evaluation."""

    text: str = Field(
        ...,
        min_length=10,
        description="The candidate's full answer to evaluate.",
    )

    @field_validator("text")
    @classmethod
    def check_max_length(cls, v: str) -> str:
        """Reject text that would produce an unbounded number of Prometheus calls.

        Args:
            v: Raw text value.

        Returns:
            The validated text string.

        Raises:
            ValueError: If text exceeds MAX_TEXT_LENGTH.
        """
        if len(v) > config.MAX_TEXT_LENGTH:
            raise ValueError(
                f"Text is too long ({len(v)} chars). Maximum is {config.MAX_TEXT_LENGTH} characters."
            )
        return v
