"""Error taxonomy for provider calls.

The split that matters is retryable vs not. Retrying a 400 wastes quota we cannot get
back; not retrying a 503 throws away a review for no reason. Every provider failure is
sorted into one of these before it reaches the retry logic.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for anything that went wrong talking to a provider."""

    retryable: bool = False

    def __init__(self, message: str, *, provider: str = "", model: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model


class TransientProviderError(ProviderError):
    """5xx, connection resets, timeouts. Worth retrying."""

    retryable = True


class RateLimitedError(ProviderError):
    """HTTP 429. Retryable, but only after the server-suggested delay."""

    retryable = True

    def __init__(self, message: str, *, retry_after: float = 5.0, **kw: str) -> None:
        super().__init__(message, **kw)
        self.retry_after = retry_after


class PermanentProviderError(ProviderError):
    """4xx that will not change on retry: bad model id, malformed request, bad key."""

    retryable = False


class QuotaExhaustedError(ProviderError):
    """Our own daily budget for this provider is spent. Not the provider's fault."""

    retryable = False


class StructuredOutputError(ProviderError):
    """The model replied, but not with anything we could parse into the schema."""

    retryable = False
