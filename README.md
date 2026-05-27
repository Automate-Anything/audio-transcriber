# Transcribe

Upload any audio or video file of any size. The server compresses it, splits at silence boundaries, sends each chunk to OpenAI Whisper, and returns a single concatenated transcript.

**Each user provides their own OpenAI API key.** Keys are stored in the user's browser (localStorage), sent to the server only as an `Authorization: Bearer` header for the duration of a job, used for the OpenAI call, then wiped from server memory. No server-side key needed.

## What the server does

| Step | Tool | What |
|---|---|---|
| 1 | `ffprobe` | Read duration |
| 2 | `ffmpeg` | Re-encode to mono 16kHz MP3 @ 24kbps **and** run `silencedetect` in one pass |
| 3 | Node | Walk the silence list, pick split points near 18-min targets, only at silence midpoints in the 14–22 min window |
| 4 | `ffmpeg` | Split with `-c copy` at those exact times (instant, no re-encode) |
| 5 | OpenAI Whisper | Transcribe each chunk sequentially using the user's key, concatenate text |

All temp files live under `/tmp` and are deleted after the job finishes.

One external API: `api.openai.com`. No databases. No object storage. No additional services.

## Local development

Requires Node 20+ and `ffmpeg` + `ffprobe` installed locally.

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install ffmpeg

# Then:
npm install
npm start
```

Open <http://localhost:3000>, paste your OpenAI key once, and use the app.

## Deploy to Render (recommended — only host that fits cleanly)

Vercel won't work: their serverless function bundles cap at 50MB and ffmpeg is ~80MB. You need a host that runs a long-lived Docker container with native ffmpeg. Render does this on the free tier.

**Steps:**

1. Push this directory to a GitHub repo.
2. [dashboard.render.com](https://dashboard.render.com) → **New** → **Web Service**.
3. Connect the GitHub repo. Render detects the `Dockerfile` automatically.
4. Settings:
   - **Runtime**: Docker
   - **Plan**: Free (sleeps after 15 min idle) or Starter ($7/mo, always on)
   - **Region**: closest to you
   - **Health Check Path**: `/health`
5. **No environment variables needed.** Each user adds their own key in the browser.
6. Click **Create Web Service**. Initial build takes ~3–5 min (Docker has to install ffmpeg).

You get a URL like `https://audio-transcribe-xxxx.onrender.com`. Anyone can open it, paste their own OpenAI key, and use the app.

## Optional: split-deploy with Cloudflare Pages (frontend) + Render (API)

You can host the static frontend on Cloudflare Pages (free, edge-cached, custom domain) and keep only the API on Render. The frontend already knows to call `CENTRAL_API` (a constant in `public/index.html`) absolutely, so cross-origin works out of the box. The Render server has CORS open.

**Setup:**

1. The repo's `public/` directory contains everything Cloudflare needs (`index.html` + `_headers` for security headers).
2. In Cloudflare Pages → Create application → Connect to Git → pick this repo.
3. Build settings:
   - **Framework preset:** None
   - **Build command:** *(leave blank)*
   - **Build output directory:** `public`
4. Save and deploy. Cloudflare gives you a `*.pages.dev` URL.
5. (Optional) Add your custom domain in Pages → Custom domains → enter domain. Cloudflare handles DNS + cert.

**Update `CENTRAL_API` in `public/index.html`** to your actual Render URL if it differs from the default (`https://audio-transcriber-ccy7.onrender.com`). Cloudflare Pages will auto-redeploy on every push to `main`.

**Notes:**

- Frontend is now always-on and globally edge-cached. First API request still wakes the sleeping Render service (~30s on free tier).
- Two URLs work after split: your custom domain (Cloudflare) and the Render URL. The Render-served frontend is unchanged so it's still a valid fallback if Cloudflare ever has issues.
- All API calls (transcribe, pairing) hit Render. Same rate limits apply.
- Cloudflare Pages free tier includes 500 builds/month and unlimited bandwidth — generous for personal use.

## Key handling, in detail

- **In the browser**: saved to `localStorage` so the user pastes it once per browser.
- **On the wire**: sent as `Authorization: Bearer sk-...` over HTTPS (Render provides certs).
- **On the server**: held in memory inside the in-memory job state for the duration of that job's processing only. Never logged. Never written to disk. Cleared when the job finishes (or after 1 hour of inactivity by the cleanup sweep).
- **What an operator (you) can see**: zero. Keys never touch disk or logs. The Node process holds them in RAM transiently.

This is fine for trusted small-team use. For a public app with strangers, add HTTPS-only enforcement (Render does this) and consider rate-limiting per source IP.

## Free tier note

On the Render free plan, the service sleeps after 15 minutes of no requests. First request after a sleep takes ~30 seconds to wake. Long transcription jobs are fine once awake — the client polls every 2 seconds, which keeps the service alive.

## Other hosts that work

Anywhere that can run a long-lived Docker container: **Fly.io**, **Railway**, **DigitalOcean App Platform**, **Hetzner**, your own VPS. Same `Dockerfile` works on all of them.

## Architecture notes

- **Memory stays low**: multer streams uploads to disk, ffmpeg reads/writes disk only, OpenAI uploads stream from disk. A 1GB WAV uses ~30MB of RAM total.
- **Smart split** uses `silencedetect` at -30dB, 0.4s min pause, then nearest-silence-to-target inside the 14–22 min window. UI shows `8 of 9 cuts aligned to silence`.
- **Jobs are in-memory**: server restart loses pending jobs. For production scale, swap the `jobs` Map for Redis.
- **No persistence of audio or transcripts.** Files in `/tmp`, deleted after job completes.

## Tuning

Constants at the top of `server.js`:

- `TARGET_CHUNK` / `MIN_CHUNK` / `MAX_CHUNK` — chunk size window (seconds)
- `SILENCE_DB` — quieter than this is considered silence (default `-30`)
- `SILENCE_MIN_SEC` — pause must last at least this long (default `0.4`)
- `OUT_SAMPLE_RATE` / `OUT_BITRATE` — output encoding params

## Files

```
audio-transcribe/
  server.js              ← Express + ffmpeg pipeline
  public/index.html      ← single-file frontend
  Dockerfile             ← installs ffmpeg, runs Node
  render.yaml            ← Render service config
  package.json
```
