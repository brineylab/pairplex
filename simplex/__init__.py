from .config import SimplexConfig
__all__=["SimplexConfig","run","score"]
def run(*a,**k):
    from .run import run as r; return r(*a,**k)
def score(*a,**k):
    from .scoring import score as s; return s(*a,**k)
