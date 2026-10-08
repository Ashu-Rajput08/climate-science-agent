from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
from html import escape

from climate_scientist.chat_store import (add_message, create_chat, list_chats,
    load_chat, rename_chat, save_chat, save_experiment)
from climate_scientist.data import (discover_datasets, load_experiment_dataset,
                                    validate_dataset)
from climate_scientist.plotly_visualizations import (build_charts, date_bounds, MONTHS,
    SEASON_ORDER, build_coverage_map)
from climate_scientist.question_check import suggest_corrected_question
from climate_scientist.supabase_auth import create_client as create_supabase_client
from climate_scientist.supabase_auth import restore_session, session_tokens
from climate_scientist.workflow import invoke


st.set_page_config(page_title="Autonomous Climate Scientist", page_icon="🌦️", layout="wide")
st.markdown("""
<style>
.parameter-chip-list { display: flex; flex-wrap: wrap; gap: .38rem; margin: .35rem 0 .65rem; }
.parameter-chip { 
  display: inline-block; border-radius: 999px; padding: .25rem .55rem;
  font-size: .73rem; line-height: 1.25rem; font-family: ui-monospace, Consolas, monospace;
  border: 1px solid rgba(158,174,201,.22); color: #E8EDF5;
}
.parameter-chip.teal { background: rgba(48,179,159,.13); border-color: rgba(75,213,190,.34); }
.parameter-chip.violet { background: rgba(129,112,232,.14); border-color: rgba(164,149,255,.32); }
.parameter-icon { font-family: sans-serif; margin-right: .3rem; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_data_bundle():
    data = load_experiment_dataset("india_daily_multicity_2000_2024")
    return data, data.inspect(), validate_dataset(data)


@st.cache_resource
def get_analysis_data(dataset_id: str):
    if dataset_id != "india_daily_multicity_2000_2024":
        raise ValueError("This app uses only the India-wide daily city dataset.")
    return load_experiment_dataset(dataset_id)


@st.cache_resource
def get_coverage_figure():
    """Build the static coverage map once per app process, not on every chat rerun."""
    sources = [source for source in discover_datasets()
               if source.get("dataset_id") == "india_daily_multicity_2000_2024"]
    return build_coverage_map(sources)


def render_data_overview() -> None:
    sources = [source for source in discover_datasets()
               if source.get("dataset_id") == "india_daily_multicity_2000_2024"]
    if not sources:
        return
    with st.container(border=True):
        st.markdown("#### Data coverage and available parameters")
        st.caption("A quick guide to the fields available for research questions. Dataset units and time handling stay source-specific.")
        map_column, catalog_column = st.columns([1.2, 1], gap="large")
        with map_column:
            st.plotly_chart(get_coverage_figure(), width="stretch",
                            config={"displayModeBar": False}, key="india-coverage-map")
            st.caption("City markers show approximate city centers for orientation; the source file itself contains no coordinates.")
        with catalog_column:
            city_source = next((source for source in sources
                                if source.get("dataset_id") == "india_daily_multicity_2000_2024"), {})
            st.markdown(f"**India multi-city · daily · {city_source.get('location_count', 0)} cities**")
            st.caption(f"{city_source.get('rows', 0):,} daily rows · "
                       f"{city_source.get('time_start', '—')} to {city_source.get('time_end', '—')} · "
                       f"{city_source.get('location_count', 0)} cities")
            city_parameters = [name for name in city_source.get("columns", [])
                               if name not in {"city", "date"}]
            _render_parameter_chips(city_parameters, "violet")
            st.caption("Units, provider timezone, and the weather-code legend are not specified in this file.")


def _render_parameter_chips(parameters: list[str], tone: str) -> None:
    if not parameters:
        return
    def icon_for(parameter: str) -> str:
        name = parameter.casefold()
        if any(word in name for word in ("precipitation", "rain")):
            return "🌧️"
        if any(word in name for word in ("wind", "gust")):
            return "🌬️"
        if "humidity" in name:
            return "💧"
        if "pressure" in name:
            return "🧭"
        if "cloud" in name:
            return "☁️"
        if "radiation" in name or "solar" in name:
            return "☀️"
        if "weather_code" in name:
            return "⛅"
        if "temperature" in name or "apparent" in name:
            return "🌡️"
        return "📊"
    items = "".join(
        f'<span class="parameter-chip {tone}"><span class="parameter-icon">{icon_for(parameter)}</span>{escape(parameter)}</span>'
        for parameter in parameters
    )
    st.markdown(f'<div class="parameter-chip-list">{items}</div>', unsafe_allow_html=True)


def answer_question(question: str, supabase, user_id: str, chat_id: str) -> tuple[str, dict]:
    """Run the investigation and return a concise answer plus expandable provenance."""
    try:
        outcome = invoke(question)
        if outcome.get("error"):
            error = outcome["error"]
            for prefix in ("ValueError: ", "KeyError: ", "TypeError: "):
                if error.startswith(prefix):
                    error = error[len(prefix):]
                    break
            return error, {}
        record = outcome.get("result", {})
        if not record:
            return "I could not produce an analysis result. Please try a more specific question.", {}
        experiment_id = save_experiment(supabase, user_id, chat_id, record)
        metadata = {
            "experiment_id": experiment_id,
            "research_question": record["experiment"].get("research_question", question),
        }
        return record.get("answer", "Analysis complete."), metadata
    except Exception as exc:
        return str(exc), {}


def render_explorer(metadata: dict, widget_id: str, expanded: bool = False) -> None:
    analysis = metadata.get("analysis")
    if not analysis:
        return
    try:
        plan = analysis["experiment"]
        explorer = st.expander("Explore graphs and filters", expanded=expanded,
                               key=f"explorer-{widget_id}", on_change="rerun")
        if explorer.open:
            with explorer:
                data = get_analysis_data(plan.get("dataset_id", "india_daily_multicity_2000_2024"))
                min_day, max_day = date_bounds(data, plan)
                timezone_note = "the IST calendar (date-only labels; no timestamp conversion)"
                st.caption(f"Filters apply to all charts and use {timezone_note}.")
                date_value = st.date_input("Time range", value=(min_day, max_day),
                    min_value=min_day, max_value=max_day, key=f"date-{widget_id}")
                if isinstance(date_value, (tuple, list)) and len(date_value) == 2:
                    date_range = (date_value[0], date_value[1])
                elif date_value:
                    date_range = (date_value, date_value)
                else:
                    date_range = (min_day, max_day)
                selected_months = st.multiselect("Months", MONTHS, default=MONTHS,
                    key=f"months-{widget_id}")
                selected_seasons = st.multiselect("Seasons", SEASON_ORDER, default=SEASON_ORDER,
                    key=f"seasons-{widget_id}")
                chart_plan = dict(plan)
                if "city" in data.daily:
                    cities = sorted(data.daily.city.dropna().astype(str).unique().tolist())
                    default_cities = plan.get("locations") or cities
                    chart_plan["locations"] = st.multiselect("Cities", cities, default=default_cities,
                        key=f"cities-{widget_id}")
                charts = build_charts(data, chart_plan, date_range,
                                      selected_months, selected_seasons)
                chart_names = [name for name, _ in charts]
                selected_chart = st.selectbox("Chart view", ["All charts", *chart_names],
                    key=f"chart-{widget_id}")
                for index, (name, figure) in enumerate(charts):
                    if selected_chart == "All charts" or selected_chart == name:
                        st.plotly_chart(figure, width="stretch",
                            key=f"plot-{widget_id}-{index}")
    except Exception as exc:
        st.warning(f"Charts could not be displayed: {exc}")


def _setting(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.environ.get(name)


def _store_auth_session(auth_response) -> None:
    session = getattr(auth_response, "session", None)
    if session is None:
        raise ValueError("Supabase did not return an active session.")
    st.session_state["supabase_auth_tokens"] = session_tokens(session)
    st.session_state.pop("active_chat_id", None)


def render_authentication():
    """Require a verified Supabase identity before any chat or experiment is loaded."""
    project_url = _setting("SUPABASE_URL")
    publishable_key = _setting("SUPABASE_PUBLISHABLE_KEY") or _setting("SUPABASE_ANON_KEY")
    if not project_url or not publishable_key:
        st.title("Connect your Supabase project")
        st.error("Chat history is disabled until authentication and per-user database storage are configured.")
        st.markdown("Add `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY` to Streamlit Community Cloud → App settings → Secrets. For local runs, put them in `.env`.")
        st.stop()

    try:
        client = create_supabase_client(project_url, publishable_key)
    except ImportError:
        st.error(
            "The Supabase Python package is missing from the Python environment running this app. "
            "In the VS Code terminal, run .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt, "
            "then stop and restart Streamlit."
        )
        st.stop()
    except Exception as exc:
        st.error(f"Could not create a Supabase client: {exc}")
        st.stop()

    user = None
    tokens = st.session_state.get("supabase_auth_tokens")
    if tokens:
        try:
            user, refreshed_tokens = restore_session(client, tokens)
            if user and refreshed_tokens:
                st.session_state["supabase_auth_tokens"] = refreshed_tokens
        except Exception:
            # Expired/revoked tokens are discarded, never downgraded to shared file storage.
            st.session_state.pop("supabase_auth_tokens", None)
            st.session_state.pop("active_chat_id", None)
    if user:
        return client, user

    st.title("Sign in to Climate Scientist")
    st.caption("Sign in with the same email account on each of your devices to see your own chat history.")
    with st.form("climate-sign-in"):
        email = st.text_input("Email", key="auth-signin-email")
        password = st.text_input("Password", type="password", key="auth-signin-password")
        signin = st.form_submit_button("Sign in", type="primary", width="stretch")
    if signin:
        try:
            response = client.auth.sign_in_with_password({"email": email.strip(), "password": password})
            _store_auth_session(response)
            st.rerun()
        except Exception as exc:
            st.error(f"Sign-in failed: {exc}")

    with st.expander("Create an account"):
        with st.form("climate-sign-up"):
            new_email = st.text_input("Email address", key="auth-signup-email")
            new_password = st.text_input("Password (at least 8 characters)", type="password",
                                         key="auth-signup-password")
            signup = st.form_submit_button("Create account", width="stretch")
        if signup:
            if len(new_password) < 8:
                st.error("Choose a password with at least 8 characters.")
            else:
                try:
                    response = client.auth.sign_up({"email": new_email.strip(), "password": new_password})
                    if getattr(response, "session", None):
                        _store_auth_session(response)
                        st.rerun()
                    st.warning(
                        "Supabase is still requiring email confirmation, so this account cannot sign in yet. "
                        "To create accounts without email verification, turn off Confirm Email in "
                        "Supabase → Authentication → Sign In / Providers → Email, then try again."
                    )
                except Exception as exc:
                    st.error(f"Account creation failed: {exc}")
    st.stop()


supabase, authenticated_user = render_authentication()
user_id = authenticated_user.id


with st.sidebar:
    st.title("Climate Scientist")
    st.caption(f"Signed in as {getattr(authenticated_user, 'email', 'user')}")
    if st.button("Sign out", width="stretch", key="sign-out"):
        try:
            supabase.auth.sign_out()
        except Exception:
            pass
        st.session_state.pop("supabase_auth_tokens", None)
        st.session_state.pop("active_chat_id", None)
        st.rerun()
    if st.button("＋ New chat", width="stretch", type="primary"):
        chat = create_chat(supabase, user_id)
        st.session_state.active_chat_id = chat["id"]
        st.rerun()

    st.subheader("Chat history")
    try:
        chat_summaries = list_chats(supabase, user_id)
    except Exception as exc:
        st.error(f"Chat storage schema needs an update. Run the current supabase/schema.sql in the Supabase SQL Editor. Details: {exc}")
        st.stop()
    if "active_chat_id" not in st.session_state and chat_summaries:
        st.session_state.active_chat_id = chat_summaries[0]["id"]
    active_id = st.session_state.get("active_chat_id")
    if active_id and not any(item["id"] == active_id for item in chat_summaries):
        st.session_state.active_chat_id = None
        active_id = None
    for summary in chat_summaries:
        label = summary.get("title") if summary.get("title_is_custom") else "Untitled chat"
        label = label or "Untitled chat"
        if st.button(label, key=f"open-{summary['id']}", width="stretch",
                     type="secondary" if summary["id"] != active_id else "primary"):
            st.session_state.active_chat_id = summary["id"]
            st.rerun()

    active_summary = next((item for item in chat_summaries if item["id"] == active_id), None)
    if active_summary:
        rename_key = f"rename-chat-title-{active_id}"
        current_title = active_summary.get("title") if active_summary.get("title_is_custom") else ""
        if rename_key not in st.session_state:
            st.session_state[rename_key] = current_title or ""
        with st.expander("Rename selected chat"):
            with st.form(f"rename-chat-form-{active_id}"):
                st.text_input("Chat name", placeholder="Untitled chat", max_chars=64,
                              key=rename_key)
                rename_submitted = st.form_submit_button("Save name", icon=":material/edit:")
            if rename_submitted:
                try:
                    rename_chat(supabase, user_id, active_id, st.session_state[rename_key])
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Could not rename this chat: {exc}")

    st.divider()
    st.caption("India daily city records · 10 cities · date-only labels are interpreted on the IST calendar.")
    try:
        _, info, quality = get_data_bundle()
        st.metric("Daily city records", f"{info['daily_rows']:,}")
        st.caption(f"{info['metadata']['time_start']} — {info['metadata']['time_end']}")
        st.caption(f"Cities: {info['metadata']['location_count']} · duplicate city/date rows: {quality['duplicate_city_date_rows']}")
        with st.expander("Dataset details"):
            for source in [source for source in discover_datasets()
                           if source.get("dataset_id") == "india_daily_multicity_2000_2024"]:
                st.markdown(f"**{source['dataset_id']}**")
                st.caption(f"{source['rows']:,} daily rows · {source['location_count']} cities · "
                           f"{source['time_start']} to {source['time_end']}")
                st.caption("Date-only labels; units, timezone, coordinates, and provider are unspecified in the file.")
                st.caption("Fields: " + ", ".join(source["columns"]))
    except Exception as exc:
        st.error(f"Dataset unavailable: {exc}")


active_id = st.session_state.get("active_chat_id")
chat = load_chat(active_id, supabase, user_id) if active_id else None
if active_id and chat is None:
    st.session_state.pop("active_chat_id", None)
    active_id = None
st.title("Climate Scientist")
render_data_overview()

with st.container():
    if chat:
      for index, message in enumerate(chat["messages"]):
        with st.chat_message(message["role"]):
            metadata = message.get("metadata", {})
            confirmation = metadata.get("confirmation")
            widget_id = f"{chat['id']}-{index}"
            is_old_report = (message["role"] == "assistant" and not metadata
                             and message["content"].lstrip().startswith("# Scientific Investigation"))
            is_old_plot_error = (message["role"] == "assistant" and not metadata
                                 and "Axes.boxplot() got an unexpected keyword argument 'labels'" in message["content"])
            if is_old_report:
                st.caption("This saved report is from an older version. Rerun the question for a concise answer.")
            elif is_old_plot_error:
                st.warning("The previous version failed while drawing this chart.")
            else:
                st.markdown(message["content"])

            if (is_old_report or is_old_plot_error):
                previous_question = next((item["content"] for item in reversed(chat["messages"][:index])
                                          if item["role"] == "user"), None)
                if previous_question and st.button("Re-run with the improved analysis", key=f"rerun-{widget_id}"):
                    with st.spinner("Re-running with the updated answer and charts…"):
                        answer, answer_metadata = answer_question(previous_question, supabase, user_id, chat["id"])
                    add_message(chat, "assistant", answer, supabase, user_id, metadata=answer_metadata)
                    st.rerun()

            if confirmation and confirmation.get("status") == "pending":
                st.markdown(f"Did you mean **“{confirmation['suggested']}”**?")
                yes_col, no_col = st.columns(2)
                if yes_col.button("Yes, analyze this", key=f"yes-{widget_id}", type="primary"):
                    confirmation["status"] = "confirmed"
                    save_chat(chat, supabase, user_id)
                    with st.spinner("Analyzing the confirmed question…"):
                        answer, answer_metadata = answer_question(confirmation["suggested"], supabase, user_id, chat["id"])
                    add_message(chat, "assistant", answer, supabase, user_id, metadata=answer_metadata)
                    st.rerun()
                if no_col.button("No, use my original wording", key=f"no-{widget_id}"):
                    confirmation["status"] = "original"
                    save_chat(chat, supabase, user_id)
                    with st.spinner("Analyzing your original question…"):
                        answer, answer_metadata = answer_question(confirmation["original"], supabase, user_id, chat["id"])
                    add_message(chat, "assistant", answer, supabase, user_id, metadata=answer_metadata)
                    st.rerun()
            elif message["role"] == "assistant":
                render_explorer(metadata, widget_id, expanded=False)
    else:
        st.info("Try: “How does precipitation vary by season?” or “Is temperature related to rainfall?”")


prompt = st.chat_input("Ask a climate research question…")
if prompt and prompt.strip():
    if not active_id:
        chat = create_chat(supabase, user_id)
        active_id = chat["id"]
        st.session_state.active_chat_id = active_id
    else:
        chat = load_chat(active_id, supabase, user_id)
        if chat is None:
            chat = create_chat(supabase, user_id)
            active_id = chat["id"]
            st.session_state.active_chat_id = active_id

    question = prompt.strip()
    add_message(chat, "user", question, supabase, user_id)
    suggestion = suggest_corrected_question(question)
    if suggestion:
        add_message(chat, "assistant", "I noticed a possible spelling correction.", supabase, user_id, metadata={
            "confirmation": {"original": question, "suggested": suggestion, "status": "pending"}
        })
        st.rerun()
    else:
        with st.spinner("Planning and running the scientific analysis…"):
            answer, answer_metadata = answer_question(question, supabase, user_id, chat["id"])
        add_message(chat, "assistant", answer, supabase, user_id, metadata=answer_metadata)
        st.rerun()
