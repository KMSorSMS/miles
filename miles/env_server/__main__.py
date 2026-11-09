"""Serve a CPU environment pool or probe its actual worker readiness."""

import argparse
import asyncio
import json
import logging
from pathlib import Path

import tomllib


def _config(path: Path, *, address: str | None, payload_advertised_host: str | None):
    # Startup-only dependency: health probes must not import environment plugins.
    from miles.env_server.config import EnvServerConfig

    with path.open("rb") as file:
        data = json.load(file) if path.suffix == ".json" else tomllib.load(file)
    # Explicit CLI deployment values win over the file before validation. This
    # supports pod IP discovery without creating a second runtime configuration.
    if address is not None:
        data.setdefault("serve", {})["address"] = address
    if payload_advertised_host is not None:
        data.setdefault("serve", {}).setdefault("payload", {})["advertised_host"] = payload_advertised_host
    return EnvServerConfig.model_validate(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path, help="JSON or TOML EnvServerConfig")
    mode.add_argument("--health-check", metavar="ADDRESS", help="Exit 0 only when server workers are ready")
    parser.add_argument("--health-timeout", type=float, default=2, help="Readiness probe deadline in seconds")
    parser.add_argument("--address", help="Override serve.address before startup")
    parser.add_argument("--payload-advertised-host", help="Override serve.payload.advertised_host, e.g. this pod's IP")
    arguments = parser.parse_args()
    if arguments.health_timeout <= 0:
        parser.error("--health-timeout must be positive")
    if arguments.health_check is not None:
        if arguments.address is not None or arguments.payload_advertised_host is not None:
            parser.error("Deployment overrides apply only with --config")
        from miles.env_server.health import ready  # avoid all environment runtime imports

        raise SystemExit(0 if asyncio.run(ready(arguments.health_check, timeout=arguments.health_timeout)) else 1)
    config = _config(
        arguments.config, address=arguments.address, payload_advertised_host=arguments.payload_advertised_host
    )
    logging.basicConfig(level=config.log_level)
    from miles.env_server.pool import serve_env  # startup-only process/runtime machinery

    serve_env(config)


if __name__ == "__main__":
    main()
