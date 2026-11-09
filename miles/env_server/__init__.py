"""Miles-owned distributed CPU environment execution, without PrimeRL dependencies."""

from importlib import import_module

_EXPORTS = {
    "EnvClient": ("client", "EnvClient"),
    "EnvClientPool": ("client_pool", "EnvClientPool"),
    "materialize_episode": ("payload", "materialize_episode"),
    "discard_episode": ("payload", "discard_episode"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    value = getattr(  # config-access-exempt: lazy exports use the fixed module/attribute registry above
        import_module(f"miles.env_server.{module}"), attribute
    )
    globals()[name] = value
    return value
