"""Async Pipedrive CRM API wrapper."""

import asyncio
from typing import Any

import httpx
from loguru import logger


class PipedriveClient:
    """Async wrapper around the Pipedrive REST API v1."""

    def __init__(
        self,
        api_token: str,
        api_url: str = "https://api.pipedrive.com/v1",
    ):
        self._token = api_token
        self._base_url = api_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an authenticated API request with rate limiting."""
        client = await self._get_client()
        url = f"{self._base_url}/{endpoint.lstrip('/')}"

        request_params = {"api_token": self._token}
        if params:
            request_params.update(params)

        for attempt in range(3):
            try:
                response = await client.request(
                    method,
                    url,
                    params=request_params,
                    json=json_data,
                )
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", "2"))
                    logger.warning(f"Pipedrive rate limit hit, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"Pipedrive API error: {e.response.status_code} {e.response.text[:200]}"
                )
                raise
            except httpx.RequestError:
                if attempt < 2:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
                raise

        return {"success": False, "error": "Max retries exceeded"}

    # --- Persons ---

    async def search_persons(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search for persons by name or email."""
        result = await self._request(
            "GET",
            "persons/search",
            params={"term": query, "limit": limit},
        )
        items = result.get("data", {}).get("items", [])
        return [item.get("item", {}) for item in items]

    async def create_person(
        self,
        name: str,
        email: str | None = None,
        phone: str | None = None,
        org_id: int | None = None,
    ) -> dict[str, Any]:
        """Create a new person in Pipedrive."""
        data: dict[str, Any] = {"name": name}
        if email:
            data["email"] = [{"value": email, "primary": True}]
        if phone:
            data["phone"] = [{"value": phone, "primary": True}]
        if org_id:
            data["org_id"] = org_id
        result = await self._request("POST", "persons", json_data=data)
        return result.get("data", {})

    async def get_person(self, person_id: int) -> dict[str, Any]:
        """Get a person by ID."""
        result = await self._request("GET", f"persons/{person_id}")
        return result.get("data", {})

    # --- Organizations ---

    async def search_organizations(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search for organizations."""
        result = await self._request(
            "GET",
            "organizations/search",
            params={"term": query, "limit": limit},
        )
        items = result.get("data", {}).get("items", [])
        return [item.get("item", {}) for item in items]

    async def create_organization(self, name: str, address: str | None = None) -> dict[str, Any]:
        """Create a new organization."""
        data: dict[str, Any] = {"name": name}
        if address:
            data["address"] = address
        result = await self._request("POST", "organizations", json_data=data)
        return result.get("data", {})

    # --- Leads ---

    async def get_leads(
        self,
        limit: int = 50,
        start: int = 0,
        archived: bool = False,
    ) -> list[dict[str, Any]]:
        """Get leads with pagination."""
        result = await self._request(
            "GET",
            "leads",
            params={
                "limit": limit,
                "start": start,
                "archived_status": ("archived" if archived else "not_archived"),
            },
        )
        return result.get("data", []) or []

    async def create_lead(
        self,
        title: str,
        person_id: int | None = None,
        org_id: int | None = None,
        value: float | None = None,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new lead."""
        data: dict[str, Any] = {"title": title}
        if person_id:
            data["person_id"] = person_id
        if org_id:
            data["organization_id"] = org_id
        if value:
            data["value"] = {
                "amount": value,
                "currency": "EUR",
            }
        if labels:
            data["label_ids"] = labels
        result = await self._request("POST", "leads", json_data=data)
        return result.get("data", {})

    # --- Activities ---

    async def create_activity(
        self,
        activity_type: str,
        subject: str,
        person_id: int | None = None,
        org_id: int | None = None,
        note: str | None = None,
        done: bool = False,
    ) -> dict[str, Any]:
        """Create an activity (call, email, meeting, etc.)."""
        data: dict[str, Any] = {
            "type": activity_type,
            "subject": subject,
            "done": 1 if done else 0,
        }
        if person_id:
            data["person_id"] = person_id
        if org_id:
            data["org_id"] = org_id
        if note:
            data["note"] = note
        result = await self._request("POST", "activities", json_data=data)
        return result.get("data", {})

    # --- Deals ---

    async def get_deals(self, limit: int = 50, start: int = 0) -> list[dict[str, Any]]:
        """Get deals with pagination."""
        result = await self._request(
            "GET",
            "deals",
            params={"limit": limit, "start": start},
        )
        return result.get("data", []) or []
