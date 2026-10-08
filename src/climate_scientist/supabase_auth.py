"""Short-lived Supabase clients and per-Streamlit-session token restoration."""
from __future__ import annotations


def _field(value, name: str, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def create_client(url: str, publishable_key: str):
    """Create a fresh client for this Streamlit session; never cache auth clients globally."""
    from supabase import create_client as make_client
    from supabase.client import ClientOptions
    return make_client(
        url, publishable_key,
        options=ClientOptions(auto_refresh_token=False, persist_session=False),
    )


def session_tokens(session) -> dict:
    access_token = _field(session, "access_token")
    refresh_token = _field(session, "refresh_token")
    if not access_token or not refresh_token:
        raise ValueError("Supabase did not return both session tokens")
    return {"access_token": access_token, "refresh_token": refresh_token}


def restore_session(client, tokens: dict | None):
    """Restore/refresh tokens and ask Supabase to verify the user identity server-side."""
    if not tokens:
        return None, None
    result = client.auth.set_session(tokens["access_token"], tokens["refresh_token"])
    current_session = client.auth.get_session()
    if hasattr(current_session, "session"):
        current_session = current_session.session
    current_session = current_session or _field(result, "session")
    user_response = client.auth.get_user()
    user = _field(user_response, "user")
    if user is None or not _field(user, "id"):
        return None, None
    return user, session_tokens(current_session)
