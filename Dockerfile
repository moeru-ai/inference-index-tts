# syntax=docker/dockerfile:1.7

FROM ghcr.io/prefix-dev/pixi:0.77.1 AS build

WORKDIR /app

COPY pyproject.toml pixi.lock ./
COPY src ./src
COPY index-tts ./index-tts

RUN --mount=type=cache,target=/root/.cache/rattler pixi install --locked
RUN printf '#!/usr/bin/env bash\n' > /entrypoint.sh \
    && pixi shell-hook --shell bash >> /entrypoint.sh \
    && printf '\nexec inference-index-tts "$@"\n' >> /entrypoint.sh \
    && chmod 0755 /entrypoint.sh

FROM ghcr.io/prefix-dev/pixi:0.77.1 AS runtime

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default
COPY --from=build /app/src /app/src
COPY --from=build /app/index-tts /app/index-tts
COPY --from=build /entrypoint.sh /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
