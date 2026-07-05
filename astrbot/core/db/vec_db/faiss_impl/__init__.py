from astrbot.core.db.vec_db.faiss_impl.vec_db import FaissVecDB


def __getattr__(name: str):
    if name == "FaissVecDB":
        from .vec_db import FaissVecDB

        return FaissVecDB
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FaissVecDB"]
