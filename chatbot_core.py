import os
from langchain_openai import ChatOpenAI

_db = None

def set_db(db):
    global _db
    _db = db

def generate_reply(
    user_text: str,
    extra_context: str = "",
    retrieval_query: str = "",
    use_knowledge: bool = True,
) -> str:
    context = (extra_context or "")[:12000]

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
    )

    prompt = f"""You are an assistant for South London College.

Rules:
- Use ONLY the provided CONTEXT.
- If the answer is not in the context, say: "not listed here".
- Use bullet points where possible.

CONTEXT:
{context}

USER:
{user_text}

ASSISTANT:
"""

    resp = llm.invoke(prompt)
    return getattr(resp, "content", str(resp))
