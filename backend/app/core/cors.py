"""
CORS that follows the sites registered in the dashboard.

The stock CORSMiddleware takes a fixed origin list from CORS_ORIGINS, so every
new customer site had to be added to .env by hand followed by a restart --
easy to forget, and the symptom (widget loads, chat silently fails) gives no
hint as to the cause.

This subclass keeps CORS_ORIGINS working for first-party origins (dashboard,
landing page) and additionally allows the origin of any site registered in the
database, since a registered site is by definition allowed to embed the widget.
The set is cached briefly so the check costs nothing per request.
"""
import time
from typing import Set
from urllib.parse import urlparse

from fastapi.middleware.cors import CORSMiddleware
from loguru import logger


class SiteAwareCORSMiddleware(CORSMiddleware):
    """CORSMiddleware that also trusts the origins of registered sites."""

    # Seconds to cache the site origin set. Short enough that a newly added
    # site starts working almost immediately, long enough to avoid per-request
    # database traffic.
    CACHE_TTL = 30

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._site_origins: Set[str] = set()
        self._fetched_at: float = 0.0

    @staticmethod
    def _origin_of(url: str) -> str:
        """Reduce a stored site URL to a bare scheme://host[:port] origin."""
        try:
            parsed = urlparse(url if "://" in url else f"https://{url}")
            if not parsed.hostname:
                return ""
            port = f":{parsed.port}" if parsed.port else ""
            return f"{parsed.scheme}://{parsed.hostname}{port}"
        except Exception:
            return ""

    async def _refresh_site_origins(self) -> Set[str]:
        now = time.monotonic()
        if self._site_origins and (now - self._fetched_at) < self.CACHE_TTL:
            return self._site_origins

        try:
            from app.database import get_database
            db = await get_database()
            sites = await db.list_sites()

            origins: Set[str] = set()
            for site in sites:
                origin = self._origin_of(site.get("url") or "")
                if not origin:
                    continue
                origins.add(origin)
                # Accept the www / bare-domain counterpart too: a site stored as
                # example.com is routinely embedded on www.example.com.
                host = urlparse(origin).hostname or ""
                scheme = urlparse(origin).scheme
                if host.startswith("www."):
                    origins.add(f"{scheme}://{host[4:]}")
                else:
                    origins.add(f"{scheme}://www.{host}")

            self._site_origins = origins
            self._fetched_at = now
        except Exception as e:
            # Never let a database hiccup break CORS for configured origins.
            logger.warning(f"Could not refresh site origins for CORS: {e}")

        return self._site_origins

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            origin = headers.get(b"origin", b"").decode("latin-1")
            if origin and origin not in self.allow_origins:
                site_origins = await self._refresh_site_origins()
                if origin in site_origins:
                    # Allow just this request's origin, rather than mutating
                    # shared state, so concurrent requests stay independent.
                    self.allow_origins = list(self.allow_origins) + [origin]
                    logger.debug(f"CORS: allowing registered site origin {origin}")
        await super().__call__(scope, receive, send)