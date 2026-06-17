# Spec: Burn captions into video (Transloadit, BYOK)

Status: **Phase A shipped** (styled `.ass` export). Phases B–D in progress.

## Goal
Let a user take their video plus the captions the app already generates
(original / multi-speaker / translated, as styled **ASS** from Phase A) and
produce a **downloadable video with the captions burned in**. Long/large videos
are supported via Transloadit's managed transcode. The user brings their own
Transloadit key (BYOK) and pays their own bill — consistent with the existing
OpenAI/AssemblyAI BYOK model, so there is no owner-cost or abuse exposure.

## End-to-end flow
1. User has a transcribed **video** and (optionally) a translation.
2. User clicks **"Burn captions into video"**, picks original vs translated.
3. Browser builds the styled ASS (`cuesToASS`, Phase A — done).
4. Browser signs the Transloadit assembly params (HMAC-SHA384, user's secret).
5. Browser uploads the video (resumable) + the ASS to Transloadit.
6. Transloadit runs `/video/subtitle` → burns the ASS into the video (re-encode).
7. Browser polls assembly status → progress bar.
8. On completion, Transloadit returns the result URL → user downloads the
   captioned video. (Results hosted temporarily by Transloadit; R2 optional later.)

## The Transloadit Assembly (inline instructions)
```jsonc
{
  "steps": {
    ":original": { "robot": "/upload/handle" },           // uploaded video
    "subs":      { "robot": "/upload/handle" },            // uploaded .ass (separate field)
    "burned": {
      "robot": "/video/subtitle",
      "use": { "steps": [ {"name": ":original", "as": "video"},
                          {"name": "subs",      "as": "subtitles"} ] },
      "subtitles_type": "burned",
      "preset": "empty",                                   // keep source resolution/bitrate
      "ffmpeg_stack": "v6",
      "ffmpeg": { "c:v": "libx264", "crf": 20, "c:a": "aac", "b:a": "128k" }
    },
    "exported": { "robot": "/file/serve", "use": "burned" } // temp hosting; swap for /s3/store (R2) later
  }
}
```
The ASS already encodes per-speaker colours + bold name labels, so
`/video/subtitle` in `burned` mode renders them onto the frame. Start with
inline assembly instructions (no per-user Template setup); move to a stored
Template later if params should be locked at Transloadit.

## Phases
**Phase B — BYOK key + signing (buildable without a live account)**
- Transloadit key panel (Auth Key + Auth Secret), stored like the existing
  OpenAI/AssemblyAI keys.
- `signAssembly(params)` via Web Crypto `HMAC-SHA384` → hex signature. Pure
  client-side; no backend change (matches the app's in-browser BYOK model).
- Lightweight "key looks valid" check.

**Phase C — Upload + assembly + poll**
- Resumable upload (tus) of the video so multi-GB / 1hr+ files survive flaky
  connections. Persist the active `assembly_ssl_url` (like the existing
  active-job resume) so a tab reload doesn't lose the job.
- Create assembly (signed) → upload video + ASS → poll status
  (`ASSEMBLY_EXECUTING` → `ASSEMBLY_COMPLETED`) → capture result URL.

**Phase D — UI + progress + download**
- "Burn captions into video" action, shown only when: source is a video,
  timestamps exist, and a Transloadit key is set.
- Choose original vs translated captions; sensible style defaults
  (bottom-centre, scaled font) with room to expose font/size/position later.
- Reuse existing progress-bar/status patterns; friendly Transloadit error
  mapping; download the finished video.

## Implementation gotchas
1. **CSP / `public/_headers`:** strict CSP. Add Transloadit API + upload hosts to
   `connect-src`, and **self-host the tus/upload client** (like `lamejs`) — a CDN
   script would violate `script-src 'self'`.
2. **Secret in the browser:** BYOK means the user's Transloadit secret is used
   client-side (same trust model as the OpenAI key). For public deployments use a
   scoped/limited credential. Alternative: a thin stateless backend signing
   endpoint (sign-and-discard). Start client-side; switch if needed.

## Costs / limits
- Transloadit is usage-based; 1hr+ 1080p burn-in is heavy and billed to the
  user's account. Free plan limits are tight; real use needs a paid plan.
- Upload is bound by the user's upstream bandwidth; resumable upload mitigates
  dropouts, not speed.

## Open product calls (deferrable)
- R2 now or later? Recommend temp hosting first, add R2 for persistence later.
- Style controls editable vs fixed defaults in v1? Recommend fixed defaults v1.
