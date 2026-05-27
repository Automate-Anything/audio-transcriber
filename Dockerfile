# Multi-stage build: pull pre-compiled static ffmpeg binaries (no apt-get,
# no compile) and copy them into a clean Node image. Cuts build time by
# ~80s vs `apt-get install ffmpeg`.
FROM mwader/static-ffmpeg:7.0 AS ffmpeg

FROM node:20-slim
COPY --from=ffmpeg /ffmpeg  /usr/local/bin/ffmpeg
COPY --from=ffmpeg /ffprobe /usr/local/bin/ffprobe

WORKDIR /app

COPY package*.json ./
RUN npm install --omit=dev --no-audit --no-fund

COPY . .

ENV NODE_ENV=production
ENV PORT=3000
EXPOSE 3000

CMD ["node", "server.js"]
