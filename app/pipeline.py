"""JobSearchCompanion LLM pipeline (Google Gemini).

Three calls, all Gemini:

  1. research_company(job_description)        -> ResearchResult (notes + citations)
                                                 uses Grounding with Google Search
  2. rewrite_cover_letter(...)  -> async stream of the tailored cover letter text
  3. review_cv(...)             -> async stream of CV feedback text

Calls 2 & 3 are independent given the research notes; the web app runs them
concurrently. The model call surface lives only here, so swapping providers
later is a contained change.

Self-check:  GEMINI_API_KEY=... python -m app.pipeline
Runs the whole pipeline on a real job description and asserts research produces
grounded citations and the writing calls produce non-empty output. This is the
make-or-break validation of research quality.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from google.genai import types

# override=True: this project's .env is authoritative even if GEMINI_API_KEY
# happens to already be set in the shell/OS environment (e.g. from another tool).
load_dotenv(override=True)

# Model is env-configurable so a retired/renamed ID is a config change, not a
# code change. The self-check prints the account's available models so we can
# confirm the exact current ID (Gemini model IDs move faster than release notes).
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

_client_singleton: genai.Client | None = None


def _client() -> genai.Client:
    global _client_singleton
    if _client_singleton is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _client_singleton = genai.Client(api_key=key)
    return _client_singleton


# --- prompts -----------------------------------------------------------------

_RESEARCH_SYSTEM = (
    "You are a job-search research assistant. Use Google Search to gather "
    "current, factual information about the hiring company and the role. Only "
    "state things you can support from the search results; if something isn't "
    "available, say so rather than inventing it. Prefer concrete details "
    "(product names, dates, recent announcements) over generic praise."
)

_RESEARCH_PROMPT = """\
Here is a job description:

<job_description>
{job_description}
</job_description>

1. Identify the hiring company and the role.
2. Search the web for: what the company does and its main product(s); notable
   news or announcements in roughly the last 12 months; its mission, values,
   culture, and communication tone; the team or product area this role sits in;
   and anything a strong applicant should reflect in their application.
3. Write concise, structured notes (grouped bullet points) that a cover-letter
   writer and a CV reviewer can use to tailor an application to THIS company and
   role. Keep it factual and specific. Output only the notes.
"""

_COVER_SYSTEM = (
    "You are an expert cover-letter writer. Rewrite the applicant's cover "
    "letter so it is specifically tailored to the target role and company, "
    "drawing on the company research and the applicant's real experience from "
    "their CV. Stay truthful: never invent experience the CV does not support. "
    "Preserve the applicant's genuine intent and voice where reasonable. Output "
    "ONLY the finished cover letter as plain text — no preamble, no commentary, "
    "no markdown headers."
)

_COVER_PROMPT = """\
<job_description>
{job_description}
</job_description>

<company_research>
{research_notes}
</company_research>

<applicant_cv_latex>
{cv_tex}
</applicant_cv_latex>

<applicant_original_cover_letter>
{cover_letter}
</applicant_original_cover_letter>

Rewrite the cover letter, tailored to this role and company. Ground specifics in
the CV and the research. Return only the cover letter text.
"""

_CV_SYSTEM = (
    "You are an expert technical recruiter and CV reviewer. Give specific, "
    "actionable feedback to improve the CV for the target role, grounded in the "
    "job description and the company research. Be honest and concrete. Do NOT "
    "rewrite the whole CV — give prioritized recommendations the applicant can "
    "act on."
)

_CV_PROMPT = """\
<job_description>
{job_description}
</job_description>

<company_research>
{research_notes}
</company_research>

<applicant_cv_latex>
{cv_tex}
</applicant_cv_latex>

Review this CV for the target role. Structure your answer as:
1. A brief overall assessment (2-3 sentences).
2. Prioritized, specific recommendations — what to add, cut, rephrase, or
   reorder — with concrete examples drawn from the CV.
3. Keyword / skill gaps versus the job description.
"""


# --- results -----------------------------------------------------------------


@dataclass
class Citation:
    title: str
    uri: str


@dataclass
class ResearchResult:
    notes: str
    citations: list[Citation]


def _response_text(resp: types.GenerateContentResponse) -> str:
    try:
        return resp.text or ""
    except (ValueError, AttributeError):
        # Blocked / empty candidate — surface as empty rather than raising.
        return ""


def _extract_citations(resp: types.GenerateContentResponse) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[str] = set()
    for cand in resp.candidates or []:
        meta = getattr(cand, "grounding_metadata", None)
        if not meta:
            continue
        for chunk in meta.grounding_chunks or []:
            web = getattr(chunk, "web", None)
            if not web or not web.uri or web.uri in seen:
                continue
            seen.add(web.uri)
            citations.append(Citation(title=web.title or web.domain or web.uri, uri=web.uri))
    return citations


# --- calls -------------------------------------------------------------------


async def research_company(job_description: str) -> ResearchResult:
    """Research the company/role with Google Search grounding."""
    resp = await _client().aio.models.generate_content(
        model=MODEL,
        contents=_RESEARCH_PROMPT.format(job_description=job_description),
        config=types.GenerateContentConfig(
            system_instruction=_RESEARCH_SYSTEM,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    return ResearchResult(notes=_response_text(resp), citations=_extract_citations(resp))


async def _stream(system: str, prompt: str) -> AsyncIterator[str]:
    async for chunk in await _client().aio.models.generate_content_stream(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system),
    ):
        if chunk.text:
            yield chunk.text


def rewrite_cover_letter(
    *, job_description: str, cv_tex: str, cover_letter: str, research_notes: str
) -> AsyncIterator[str]:
    return _stream(
        _COVER_SYSTEM,
        _COVER_PROMPT.format(
            job_description=job_description,
            research_notes=research_notes,
            cv_tex=cv_tex,
            cover_letter=cover_letter,
        ),
    )


def review_cv(
    *, job_description: str, cv_tex: str, research_notes: str
) -> AsyncIterator[str]:
    return _stream(
        _CV_SYSTEM,
        _CV_PROMPT.format(
            job_description=job_description,
            research_notes=research_notes,
            cv_tex=cv_tex,
        ),
    )


# --- self-check --------------------------------------------------------------

_SAMPLE_JD = """\
Senior Backend Engineer — Stripe (Payments Infrastructure)

Stripe builds economic infrastructure for the internet. We're looking for a
Senior Backend Engineer to join our Payments Infrastructure team, building the
APIs and systems that move money reliably at massive scale. You'll design
fault-tolerant distributed services in a language like Go, Java, or Ruby, own
systems end to end, and collaborate closely with product teams. We value
users first, rigorous thinking, and clear written communication.
"""

_SAMPLE_CV_TEX = r"""
\documentclass{article}
\begin{document}
\textbf{Jane Doe} — Backend Engineer \\
Email: jane@example.com

\section*{Experience}
\textbf{Backend Engineer, Acme Corp} (2021--present) \\
Built and operated REST APIs in Python/Flask serving 2M requests/day.
Reduced p99 latency 40\% by adding caching and query optimization.

\section*{Skills}
Python, Flask, PostgreSQL, Redis, Docker, AWS.
\end{document}
"""

_SAMPLE_COVER = """\
Dear Hiring Manager,

I am writing to apply for the Backend Engineer role. I have three years of
experience building web APIs in Python and enjoy working on reliable systems.
I would welcome the chance to contribute to your team.

Sincerely,
Jane Doe
"""


async def _selfcheck() -> None:
    failures: list[str] = []

    print(f"Model: {MODEL}\n")
    print("=== Gemini models available to this key (generateContent) ===")
    try:
        for m in _client().models.list():
            actions = getattr(m, "supported_actions", None) or []
            if "generateContent" in actions and "gemini" in (m.name or ""):
                print(f"  {m.name}")
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        print(f"  (could not list models: {exc})")

    print("\n=== 1. RESEARCH (Google Search grounding) ===")
    research = await research_company(_SAMPLE_JD)
    print(research.notes)
    print("\n--- citations ---")
    for c in research.citations:
        print(f"  - {c.title} :: {c.uri}")

    if not research.notes.strip():
        failures.append("research notes are empty")
    if not research.citations:
        failures.append("research returned no grounding citations — search may not be active")

    print("\n=== 2 & 3. COVER LETTER + CV FEEDBACK (concurrent stream) ===")

    async def _collect(stream: AsyncIterator[str]) -> str:
        return "".join([chunk async for chunk in stream])

    cover, feedback = await asyncio.gather(
        _collect(
            rewrite_cover_letter(
                job_description=_SAMPLE_JD,
                cv_tex=_SAMPLE_CV_TEX,
                cover_letter=_SAMPLE_COVER,
                research_notes=research.notes,
            )
        ),
        _collect(
            review_cv(
                job_description=_SAMPLE_JD,
                cv_tex=_SAMPLE_CV_TEX,
                research_notes=research.notes,
            )
        ),
    )

    print("\n----- TAILORED COVER LETTER -----\n" + cover)
    print("\n----- CV FEEDBACK -----\n" + feedback)

    if not cover.strip():
        failures.append("cover letter is empty")
    if not feedback.strip():
        failures.append("cv feedback is empty")

    print("\n=== VERDICT ===")
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
    else:
        print("  PASS: research is grounded and all three calls produced output.")
    assert not failures, failures


if __name__ == "__main__":
    asyncio.run(_selfcheck())
