# syntax=docker/dockerfile:1.7

FROM ghcr.io/prefix-dev/pixi:0.77.1 AS build

WORKDIR /app

COPY pyproject.toml pixi.lock ./
COPY src ./src
COPY index-tts ./index-tts

RUN --mount=type=cache,target=/root/.cache/rattler \
    pixi install --environment runtime --locked
RUN printf '#!/usr/bin/env bash\n' > /entrypoint.sh \
    && pixi shell-hook --environment runtime --shell bash >> /entrypoint.sh \
    && printf '\nexec inference-index-tts "$@"\n' >> /entrypoint.sh \
    && chmod 0755 /entrypoint.sh
RUN mv /app/.pixi/envs/runtime/lib/python3.11/site-packages/nvidia /runtime-nvidia \
    && mv /app/.pixi/envs/runtime/lib/python3.11/site-packages /runtime-site-packages \
    && mkdir -p /app/.pixi/envs/runtime/lib/python3.11/site-packages

FROM ghcr.io/prefix-dev/pixi:0.77.1 AS runtime

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates gcc libc6-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

LABEL org.opencontainers.image.source="https://github.com/moeru-ai/inference-index-tts"

COPY --from=build /app/.pixi/envs/runtime /app/.pixi/envs/runtime
COPY --from=build /runtime-site-packages /app/.pixi/envs/runtime/lib/python3.11/site-packages
COPY --from=build /runtime-nvidia /app/.pixi/envs/runtime/lib/python3.11/site-packages/nvidia
COPY --from=build /app/src /app/src
COPY --from=build /app/index-tts /app/index-tts
COPY --from=build /entrypoint.sh /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
