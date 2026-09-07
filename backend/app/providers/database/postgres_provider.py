"""
PostgreSQL implementation of the database provider.

Wraps the PostgresDB client so the provider architecture and the direct
``app.database`` accessor share a single implementation. PostgresDB already
covers the overwhelming majority of BaseDatabaseProvider's surface with matching
signatures, so this class inherits it and only fills in the handful of
provider-specific operations.
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from app.database.postgres import PostgresDB
from .base import BaseDatabaseProvider


class PostgresProvider(PostgresDB, BaseDatabaseProvider):
    """PostgreSQL implementation of the database provider."""

    # ===========================================
    # Site configuration
    # ===========================================

    async def get_site_config(self, site_id: str) -> Optional[Dict]:
        """Get site configuration."""
        site = await self.get_site(site_id)
        if not site:
            return None
        return {
            "config": site.get("config", {}),
            "appearance": site.get("appearance", {}),
            "behavior": site.get("behavior", {})
        }

    async def update_site_config(self, site_id: str, config: Dict) -> bool:
        """Update site configuration."""
        if not await self.get_site(site_id):
            return False
        return await self.update_site(site_id, dict(config))

    # ===========================================
    # Documents
    # ===========================================

    async def save_document(self, doc_data: Dict) -> str:
        """Save document metadata. Returns document_id."""
        doc = dict(doc_data)
        doc_id = doc.get("doc_id") or doc.get("document_id") or str(uuid.uuid4())
        doc["doc_id"] = doc_id
        doc.setdefault("uploaded_at", datetime.utcnow())
        await self.create_document(doc)
        return doc_id

    # ===========================================
    # Analytics
    # ===========================================

    async def get_analytics_overview(
        self,
        site_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """Get analytics overview data."""
        clauses: List[str] = []
        params: List = []

        if site_id:
            params.append(site_id)
            clauses.append(f"site_id = ${len(params)}")
        if start_date:
            params.append(start_date)
            clauses.append(f"created_at >= ${len(params)}")
        if end_date:
            params.append(end_date)
            clauses.append(f"created_at <= ${len(params)}")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT COUNT(*) AS total_conversations,
                       COALESCE(SUM(jsonb_array_length(messages)), 0) AS total_messages
                FROM conversations {where}
                """,
                *params,
            )

        return {
            "total_conversations": row["total_conversations"] if row else 0,
            "total_messages": int(row["total_messages"]) if row else 0,
            "site_id": site_id
        }
