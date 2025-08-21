from contextvars import ContextVar

current_p_emb = ContextVar("current_p_emb", default=None)
