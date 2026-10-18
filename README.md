# NGO Social Impact & Volunteer Triage PoC 🌍🤖

A Proof-of-Concept (PoC) LLM automation tool built with **Python, Streamlit, pypdf, and Gemini 2.5 Flash** to translate the conceptual findings from the **InAmigos Foundation AI for Social Impact Challenge** into working software.

## Key Engineering Features
- **Constrained JSON Output:** Uses Pydantic `response_schema` with `gemini-2.5-flash` to guarantee deterministic JSON responses for UI rendering.
- **Unstructured PDF Parsing:** Extracts volunteer resumes/bios using `pypdf` and matches them to 5 core NGO wings (*Education, Animal Welfare, Community Support, Accessibility, Awareness*).
- **OWASP LLM01 & LLM02 Hardening:** Implements XML-tag input delimiters (`<user_data>`), length bounding, and deterministic JSON fallback handling so malformed responses never crash the UI.