
import streamlit as st
import requests
import os
import re


def render_policy_text(text):
    """Display currency literally while retaining Markdown layout and citations."""
    # Dollar signs in hotel fees are currency, not paired LaTeX delimiters.
    escaped = re.sub(r"(?<!\\)\$", lambda match: "\\$", text)
    st.markdown(escaped)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:7860").rstrip("/")


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Aurora Stays",
    page_icon="🏨",
    layout="wide"
)


# ---------------------------------------------------------
# Sidebar  -  Model parameters
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ Parameters")

    k = st.slider(
        "k (chunks to retrieve)",
        min_value=1,
        max_value=10,
        value=4,
        help="Number of relevant chunks fetched from the vector store."
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Higher values make the output more creative; lower values make it more focused."
    )

    max_tokens = st.slider(
        "Max tokens",
        min_value=64,
        max_value=2048,
        value=512,
        step=64,
        help="Maximum number of tokens in the generated answer."
    )


# ---------------------------------------------------------
# Main area
# ---------------------------------------------------------

st.title("🏨 Aurora Stays")
st.write("Internal staff POC: ask about AuroraStay hotel policies and verify the supporting sources.")


# ---------------------------------------------------------
# Response type selection (two options only)
# ---------------------------------------------------------

response_type = st.radio(
    "Response mode",
    ["Answer only", "Answer + Relevant Chunks"],
    index=1,
    horizontal=True
)


# ---------------------------------------------------------
# Single or batch input; both use the same deployed RAG endpoint.
# ---------------------------------------------------------

input_mode = st.radio(
    "Input mode", ["Single question", "Batch questions"], horizontal=True
)
if input_mode == "Single question":
    query_text = st.text_input(
        "Your question",
        placeholder="e.g., What is the pet fee at ASH-CHI Loop for a four-night stay?"
    )
else:
    query_text = st.text_area(
        "Questions (one per line)",
        height=140,
        placeholder=(
            "What is the pet fee at ASH-CHI Loop for a four-night stay?\n"
            "How many people can stay in a Double Queen room?\n"
            "Does a member discount automatically mean I can cancel for free?"
        ),
    )
    st.caption("Submit up to 10 questions together. Each is processed in order with the same settings.")


def request_answer(question, endpoint, settings):
    """Keep a failed request local to its question so the batch can continue."""
    try:
        response = requests.post(
            f"{BACKEND_URL}{endpoint}",
            json={"query": question, **settings},
            timeout=(10, 90),
        )
        if response.status_code != 200:
            return {"question": question, "error": f"RAG API returned HTTP {response.status_code}."}
        result = response.json()
        if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
            raise ValueError("Missing answer")
        chunks = result.get("relevant_chunks", [])
        if not isinstance(chunks, list) or not all(isinstance(chunk, str) for chunk in chunks):
            raise ValueError("Invalid sources")
        return {"question": question, "answer": result["answer"], "chunks": chunks}
    except requests.exceptions.RequestException:
        return {"question": question, "error": "Unable to connect to the RAG API. Check the backend service."}
    except (ValueError, TypeError):
        return {"question": question, "error": "The RAG API returned an invalid response."}


if st.button("Run batch" if input_mode == "Batch questions" else "Submit", type="primary"):
    questions = [line.strip() for line in query_text.splitlines() if line.strip()]
    if not questions:
        st.warning("Please enter at least one question.")
    elif len(questions) > 10:
        st.warning("Please submit no more than 10 questions at a time.")
    else:
        endpoint = "/v1/answer" if response_type == "Answer only" else "/v1/answer_with_relevant_chunks"
        settings = {
            "k": k, "temperature": temperature, "max_tokens": max_tokens,
            "model_name": "gpt-4o-mini", "top_p": 1,
        }
        results = []
        with st.spinner("Processing questions..."):
            for question in questions:
                results.append(request_answer(question, endpoint, settings))
        # Retain completed answers across widget reruns without issuing more requests.
        st.session_state["inference_results"] = {
            "mode": input_mode, "response_type": response_type,
            "settings": settings, "results": results,
        }

saved = st.session_state.get("inference_results")
if saved:
    results = saved["results"]
    successes = sum("answer" in result for result in results)
    st.subheader("Batch results" if saved["mode"] == "Batch questions" else "Result")
    st.caption(
        f"{successes}/{len(results)} requests succeeded | "
        f"K={saved['settings']['k']} | Temperature={saved['settings']['temperature']} | "
        f"Max tokens={saved['settings']['max_tokens']} | {saved['response_type']}"
    )
    for number, result in enumerate(results, start=1):
        if saved["mode"] == "Batch questions":
            st.markdown(f"**Question {number}**")
        st.write(result["question"])
        if "error" in result:
            st.error(result["error"])
            continue
        st.subheader("Answer")
        render_policy_text(result["answer"])
        if saved["response_type"] == "Answer + Relevant Chunks":
            for source_number, chunk in enumerate(result["chunks"], start=1):
                with st.expander(f"Question {number} - Source {source_number}"):
                    render_policy_text(chunk)
        st.divider()
