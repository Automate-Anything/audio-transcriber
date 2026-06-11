// Audio transcription server.
// - Accepts any audio/video file, runs native ffmpeg to compress + smart-split,
//   then sends chunks to OpenAI Whisper.
// - User's OpenAI key arrives per request via Authorization header (never persisted).
// - "Bring your own server" via pairing code: on boot, if PAIRING_CODE env var is
//   set, this server self-registers with a central registry so the frontend can
//   auto-discover it without the user copy-pasting a URL.
// - Fast paths: small already-compressed files skip ffmpeg entirely; larger
//   already-compressed files skip the re-encode but still get smart-split.

const express = require('express');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const multer = require('multer');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');
const dns = require('dns').promises;
const net = require('net');
const OpenAI = require('openai');
const { mountMcp } = require('./mcp.js');

// ============================================================================
// Config
// ============================================================================

const PORT = process.env.PORT || 3000;
const MAX_FILE_SIZE = parseInt(process.env.MAX_FILE_SIZE || (5 * 1024 * 1024 * 1024), 10);

// Smart chunking (seconds)
const TARGET_CHUNK    = 18 * 60;
const MIN_CHUNK       = 14 * 60;
const MAX_CHUNK       = 22 * 60;
const SILENCE_DB      = '-30';
const SILENCE_MIN_SEC = '0.4';

// Audio output for transcription
const OUT_SAMPLE_RATE = 16000;
const OUT_BITRATE     = '24k';

// Fast-path: already-compressed formats Whisper accepts directly
const COMPRESSED_EXTS = ['.mp3', '.m4a', '.mp4', '.aac', '.opus', '.webm', '.ogg', '.mpga', '.mpeg'];
// If under this size AND already compressed AND under MAX_CHUNK in duration -> send as-is
const DIRECT_TO_WHISPER_MAX_BYTES = 24 * 1024 * 1024; // a hair under Whisper's 25MB limit

// ---------------------------------------------------------------------------
// Concurrency: how many CPU-heavy (ffmpeg) jobs may run at once.
//
// On our free shared Render server the box is tiny (~0.1 shared vCPU, 512MB)
// and a single ffmpeg compress already saturates it — so we process one file
// at a time and queue the rest. On a user's own (paid) server we scale to the
// instance's CPU allotment. os.cpus() is unreliable in containers (it reports
// the host, not the cgroup limit), so we trust Render's RENDER_CPU_COUNT env
// var ("0.5" Starter, "2" Pro, ...). Non-Render hosts fall back to os.cpus().
const IS_RENDER    = process.env.RENDER === 'true';
const RENDER_CPUS  = parseFloat(process.env.RENDER_CPU_COUNT || '');     // NaN if unset
const IS_FREE_PLAN = IS_RENDER && (!RENDER_CPUS || RENDER_CPUS <= 0.25); // free ≈ shared/tiny
const SERVER_PLAN  = !IS_RENDER ? 'self-hosted' : (IS_FREE_PLAN ? 'free' : 'paid');
const MAX_CONCURRENT_JOBS = (() => {
  if (IS_FREE_PLAN) return 1;
  if (IS_RENDER && RENDER_CPUS) return Math.max(1, Math.min(4, Math.floor(RENDER_CPUS)));
  // Non-Render VM: os.cpus() is accurate; leave a core for the event loop.
  return Math.max(1, Math.min(4, os.cpus().length - 1));
})();

// Simple FIFO semaphore guarding the heavy (OpenAI/ffmpeg) path. AssemblyAI
// jobs offload work to AssemblyAI's servers, so they are NOT gated here.
let activeHeavyJobs = 0;
const heavyWaiters = [];
function acquireHeavySlot() {
  if (activeHeavyJobs < MAX_CONCURRENT_JOBS) { activeHeavyJobs++; return Promise.resolve(); }
  return new Promise((resolve) => heavyWaiters.push(resolve));
}
function releaseHeavySlot() {
  const next = heavyWaiters.shift();
  if (next) { next(); return; }        // hand the slot straight to the next waiter
  activeHeavyJobs = Math.max(0, activeHeavyJobs - 1);
}

// Pairing
const PAIRING_CODE        = process.env.PAIRING_CODE || '';
const RENDER_EXTERNAL_URL = process.env.RENDER_EXTERNAL_URL || '';
const REGISTER_URL        = (process.env.REGISTER_URL || 'https://audio-transcriber-ccy7.onrender.com').replace(/\/+$/, '');
const PAIRING_TTL_MS      = 30 * 60 * 1000;
const MAX_PAIRINGS        = 10_000;      // memory cap
const MIN_CODE_LENGTH     = 12;          // forces high-entropy codes (e.g. word-word-xxxxxx)

// ============================================================================
// App
// ============================================================================

const app = express();

// Render and similar PaaS proxies sit in front of the app — needed so rate
// limiting sees real client IPs instead of the proxy's.
app.set('trust proxy', 1);

// Security headers via helmet. We use 'unsafe-inline' for scripts and styles
// because the app inlines them; the bigger wins here are X-Frame-Options
// (clickjacking), Referrer-Policy, X-Content-Type-Options, and a connect-src
// policy that prevents data exfiltration to non-HTTPS origins.
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      "default-src": ["'self'"],
      "script-src": ["'self'", "'unsafe-inline'"],
      "style-src": ["'self'", "'unsafe-inline'"],
      "img-src": ["'self'", "data:", "https://automateanything.ai"],
      // Allow blob: URLs for in-browser audio recording playback.
      "media-src": ["'self'", "blob:"],
      // Permit calls back to the app's own origin + any HTTPS endpoint (BYO
      // server feature needs this) + localhost for dev. Blocks http: outside
      // dev so audio + keys can't be exfiltrated to insecure endpoints.
      "connect-src": ["'self'", "https:", "http://localhost:*"],
      "frame-ancestors": ["'none'"],
      "base-uri": ["'self'"],
      "form-action": ["'self'"],
    },
  },
  // Render terminates TLS and adds HSTS at the proxy; doing it here can break
  // local development.
  strictTransportSecurity: false,
  // X-Powered-By header is removed by helmet's hidePoweredBy (on by default).
  crossOriginEmbedderPolicy: false, // allow audio uploads from any origin
}));

// CORS — needed so any frontend (including this app served from another origin)
// can talk to this server when used as a "bring your own server". Helmet's
// CSP above is the inbound protection; CORS controls outbound from the browser.
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.setHeader('Access-Control-Max-Age', '86400');
  if (req.method === 'OPTIONS') return res.status(204).end();
  next();
});

// Rate limiters — per-IP buckets keep one bad actor from monopolizing.
// Generous defaults; real abuse will trip them, normal use won't notice.
const pairRegisterLimiter = rateLimit({
  windowMs: 60 * 1000, max: 20,
  standardHeaders: true, legacyHeaders: false,
  message: { error: 'Too many requests, please slow down.' },
});
const pairLookupLimiter = rateLimit({
  windowMs: 60 * 1000, max: 120, // polling sends ~30/min so leave headroom
  standardHeaders: true, legacyHeaders: false,
  message: { error: 'Too many lookups, please slow down.' },
});
const jobsLimiter = rateLimit({
  windowMs: 60 * 1000, max: 240, // polling can be busy
  standardHeaders: true, legacyHeaders: false,
});
const postProcessLimiter = rateLimit({
  windowMs: 60 * 60 * 1000, max: 60, // summary/actions/translate calls per hour
  standardHeaders: true, legacyHeaders: false,
  message: { error: 'You\'ve hit the hourly limit for summary/translation on this server.' },
});

// JSON body parsing is applied per-route (not globally) so the large
// transcript bodies on /api/post-process aren't capped by the small limit
// the pairing endpoints want. /api/transcribe uses multipart (multer).
app.use(express.static('public'));

// ============================================================================
// Pairing registry (in-memory) — this server can act as the "central" server
// that user-deployed servers phone home to.
// ============================================================================

const pairings = new Map(); // code -> { url, timestamp }

setInterval(() => {
  const cutoff = Date.now() - PAIRING_TTL_MS;
  for (const [code, p] of pairings) {
    if (p.timestamp < cutoff) pairings.delete(code);
  }
}, 5 * 60 * 1000);

app.post('/api/pair/register', pairRegisterLimiter, express.json({ limit: '64kb' }), (req, res) => {
  const code = String((req.body && req.body.code) || '').trim().toLowerCase();
  const url  = String((req.body && req.body.url)  || '').trim();
  if (!code || !url) return res.status(400).json({ error: 'code and url required' });
  if (!/^https?:\/\//.test(url)) return res.status(400).json({ error: 'url must include scheme' });
  if (code.length < MIN_CODE_LENGTH) return res.status(400).json({ error: 'code too short' });
  if (code.length > 64 || url.length > 256) return res.status(400).json({ error: 'too long' });

  // Anti-clobber: if a *different* URL claimed the same code in the last minute, reject.
  const existing = pairings.get(code);
  if (existing && existing.url !== url && (Date.now() - existing.timestamp) < 60_000) {
    return res.status(409).json({ error: 'code already claimed' });
  }

  // Bound the map: evict oldest 10% when at cap to make room.
  if (pairings.size >= MAX_PAIRINGS) {
    const sorted = [...pairings.entries()].sort((a, b) => a[1].timestamp - b[1].timestamp);
    const toEvict = Math.floor(MAX_PAIRINGS * 0.1);
    for (let i = 0; i < toEvict; i++) pairings.delete(sorted[i][0]);
  }

  pairings.set(code, { url, timestamp: Date.now() });
  // Log only a truncated code so logs don't function as a pairing oracle.
  console.log(`[pair] registered ${code.slice(0, 6)}…`);
  res.json({ ok: true });
});

app.get('/api/pair/lookup/:code', pairLookupLimiter, (req, res) => {
  const code = String(req.params.code || '').trim().toLowerCase();
  const p = pairings.get(code);
  if (!p || (Date.now() - p.timestamp > PAIRING_TTL_MS)) {
    pairings.delete(code);
    return res.json({ url: null });
  }
  res.json({ url: p.url });
});

// ============================================================================
// Upload + transcribe
// ============================================================================

const uploadDir = path.join(os.tmpdir(), 'transcribe-uploads');
fs.mkdirSync(uploadDir, { recursive: true });

const upload = multer({
  storage: multer.diskStorage({
    destination: uploadDir,
    filename: (req, file, cb) => {
      const id = crypto.randomBytes(8).toString('hex');
      const ext = (path.extname(file.originalname) || '.bin').toLowerCase();
      cb(null, `${id}${ext}`);
    },
  }),
  limits: { fileSize: MAX_FILE_SIZE },
});

const jobs = new Map();

// Upload slots for the "presigned upload" pattern used by API/MCP callers who
// have a local file (no public URL). Flow: POST /api/uploads -> { uploadId,
// uploadUrl }; PUT the bytes to uploadUrl; then transcribe referencing the
// uploadId. Each slot is just a temp-file holder — no persistent storage.
const uploads = new Map(); // uploadId -> { path, ready, size, name, createdAt }
const UPLOAD_TTL_MS = 30 * 60 * 1000;

function publicBaseUrl(req) {
  // Honor proxy headers (Render/Cloudflare sit in front).
  const proto = (req.headers['x-forwarded-proto'] || req.protocol || 'https').split(',')[0].trim();
  const host = (req.headers['x-forwarded-host'] || req.headers['host'] || '').split(',')[0].trim();
  return host ? `${proto}://${host}` : '';
}

// Create an upload slot. Returns a one-time URL to PUT the file bytes to.
app.post('/api/uploads', express.json({ limit: '2kb' }), (req, res) => {
  if (shuttingDown) return res.status(503).json({ error: 'Server is updating — try again in a moment.' });
  const uploadId = crypto.randomBytes(18).toString('hex');
  const name = (req.body && req.body.fileName ? req.body.fileName.toString() : 'audio')
    .replace(/[^\w.\-]/g, '_').slice(0, 100) || 'audio';
  const tmpPath = path.join(os.tmpdir(), 'up-' + uploadId + '-' + name);
  uploads.set(uploadId, { path: tmpPath, ready: false, size: 0, name, createdAt: Date.now() });
  const base = publicBaseUrl(req);
  res.json({
    uploadId,
    uploadUrl: `${base}/api/uploads/${uploadId}`,
    method: 'PUT',
    expiresInSeconds: Math.floor(UPLOAD_TTL_MS / 1000),
    note: 'PUT the raw file bytes to uploadUrl, then call transcribe with this uploadId.',
  });
});

// Receive the bytes for a slot (raw body stream, size-capped).
app.put('/api/uploads/:id', (req, res) => {
  const slot = uploads.get(req.params.id);
  if (!slot) return res.status(404).json({ error: 'Unknown or expired uploadId.' });
  if (slot.ready) return res.status(409).json({ error: 'This upload slot was already filled.' });

  const out = fs.createWriteStream(slot.path);
  let written = 0, aborted = false;
  const fail = (status, msg) => {
    aborted = true;
    try { out.destroy(); } catch {}
    try { fs.unlinkSync(slot.path); } catch {}
    uploads.delete(req.params.id);
    if (!res.headersSent) res.status(status).json({ error: msg });
  };
  req.on('data', (chunk) => {
    if (aborted) return;
    written += chunk.length;
    if (written > MAX_FILE_SIZE) return fail(413, 'File exceeds the size limit.');
  });
  req.pipe(out);
  out.on('error', () => fail(500, 'Failed to store upload.'));
  req.on('error', () => fail(400, 'Upload stream error.'));
  out.on('finish', () => {
    if (aborted) return;
    if (written === 0) return fail(400, 'Empty upload.');
    slot.ready = true;
    slot.size = written;
    res.json({ uploadId: req.params.id, size: written, ready: true });
  });
});

// Resolve an uploadId into a fileInfo for transcription (and consume the slot
// so it can't be reused). Throws { status, message } if not usable.
function consumeUpload(uploadId) {
  const slot = uploads.get(uploadId);
  if (!slot) { const e = new Error('Unknown or expired uploadId.'); e.status = 404; throw e; }
  if (!slot.ready) { const e = new Error('Upload not finished — PUT the bytes first.'); e.status = 400; throw e; }
  uploads.delete(uploadId);
  return { path: slot.path, size: slot.size, originalname: slot.name };
}

app.get('/health', (req, res) => res.json({ ok: true }));

// Tells the client how many files this server will transcribe at once so it can
// pace uploads (and label the UI). The free shared box is 1-at-a-time; a user's
// own paid server scales with its CPU. See MAX_CONCURRENT_JOBS.
app.get('/api/capabilities', (req, res) => {
  res.json({ ok: true, maxConcurrent: MAX_CONCURRENT_JOBS, plan: SERVER_PLAN });
});

// Graceful-shutdown flag. Flipped to true on SIGTERM so new submissions get a
// clean "server updating" response instead of a half-handled request that
// dies mid-flight. In-flight polling endpoints keep working until the
// container is torn down.
let shuttingDown = false;

// Lightweight version probe. The client polls this to know when a new build
// has been deployed, so it can offer a non-intrusive "update available"
// hint. Bumped manually by us in commits.
const BUILD_VERSION = process.env.BUILD_VERSION || 'dev';
app.get('/api/version', (req, res) => res.json({ version: BUILD_VERSION, shuttingDown }));

// ============================================================================
// URL ingestion helpers (SSRF-guarded fetch of remote audio/video)
// ============================================================================
function isPrivateIp(ip) {
  if (net.isIPv4(ip)) {
    const p = ip.split('.').map(Number);
    if (p[0] === 10 || p[0] === 127 || p[0] === 0) return true;
    if (p[0] === 169 && p[1] === 254) return true;            // link-local + cloud metadata
    if (p[0] === 172 && p[1] >= 16 && p[1] <= 31) return true;
    if (p[0] === 192 && p[1] === 168) return true;
    if (p[0] === 100 && p[1] >= 64 && p[1] <= 127) return true; // CGNAT
    return false;
  }
  const lower = ip.toLowerCase();
  if (lower === '::1' || lower === '::') return true;
  if (lower.startsWith('fe80')) return true;                  // link-local
  if (lower.startsWith('fc') || lower.startsWith('fd')) return true; // unique-local
  if (lower.startsWith('::ffff:')) return isPrivateIp(lower.slice(7)); // v4-mapped
  return false;
}

// Reject anything that isn't a public http(s) URL — prevents the server from
// being tricked into fetching internal services or cloud metadata (SSRF).
async function assertPublicUrl(urlStr) {
  let u;
  try { u = new URL(urlStr); } catch { throw new Error('Invalid URL.'); }
  if (u.protocol !== 'http:' && u.protocol !== 'https:') throw new Error('Only http(s) URLs are allowed.');
  const host = u.hostname;
  if (/^(localhost|.*\.local|.*\.internal)$/i.test(host)) throw new Error('That URL host is not allowed.');
  let ips;
  if (net.isIP(host)) ips = [host];
  else {
    try { ips = (await dns.lookup(host, { all: true })).map(r => r.address); }
    catch { throw new Error('Could not resolve the URL host.'); }
  }
  for (const ip of ips) {
    if (isPrivateIp(ip)) throw new Error('That URL resolves to a private address and is not allowed.');
  }
  return u;
}

// Fetch a remote audio/video URL to a temp file. Re-validates at every redirect
// hop (so a redirect can't smuggle us to a private address), enforces a 30s
// timeout and the global size cap while streaming.
async function fetchAudioToTemp(urlStr) {
  let current = urlStr;
  let resp = null;
  for (let hop = 0; hop < 4; hop++) {
    await assertPublicUrl(current);
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 30000);
    let r;
    try {
      r = await fetch(current, {
        redirect: 'manual',
        signal: ctrl.signal,
        headers: { 'User-Agent': 'AutomateAnythingTranscribe/1.0' },
      });
    } catch (e) {
      throw new Error('Could not reach that URL.');
    } finally { clearTimeout(timer); }
    if (r.status >= 300 && r.status < 400 && r.headers.get('location')) {
      current = new URL(r.headers.get('location'), current).toString();
      continue;
    }
    resp = r; break;
  }
  if (!resp) throw new Error('Too many redirects.');
  if (!resp.ok) throw new Error(`The URL returned ${resp.status}.`);

  const declaredLen = parseInt(resp.headers.get('content-length') || '0', 10);
  if (declaredLen && declaredLen > MAX_FILE_SIZE) throw new Error('Audio at that URL exceeds the size limit.');

  // Filename + extension (guess from content-type if the URL has none)
  const u = new URL(current);
  let name = decodeURIComponent(path.basename(u.pathname)) || 'audio';
  if (!path.extname(name)) {
    const ct = (resp.headers.get('content-type') || '').split(';')[0].trim().toLowerCase();
    const extMap = {
      'audio/mpeg': '.mp3', 'audio/mp3': '.mp3', 'audio/mp4': '.m4a', 'audio/x-m4a': '.m4a',
      'audio/wav': '.wav', 'audio/x-wav': '.wav', 'audio/webm': '.webm', 'audio/ogg': '.ogg',
      'audio/aac': '.aac', 'audio/opus': '.opus',
      'video/mp4': '.mp4', 'video/quicktime': '.mov', 'video/webm': '.webm',
    };
    name += extMap[ct] || '.mp3';
  }
  const safeName = name.replace(/[^\w.\-]/g, '_').slice(0, 100);
  const tmpPath = path.join(os.tmpdir(), crypto.randomBytes(12).toString('hex') + '-' + safeName);

  const fileStream = fs.createWriteStream(tmpPath);
  let written = 0;
  try {
    const reader = resp.body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      written += value.length;
      if (written > MAX_FILE_SIZE) {
        throw new Error('Audio at that URL exceeds the size limit.');
      }
      if (!fileStream.write(Buffer.from(value))) {
        await new Promise(r => fileStream.once('drain', r));
      }
    }
    await new Promise((resolve, reject) => { fileStream.end(resolve); fileStream.on('error', reject); });
  } catch (e) {
    try { fileStream.destroy(); } catch {}
    try { fs.unlinkSync(tmpPath); } catch {}
    throw e;
  }
  if (written === 0) { try { fs.unlinkSync(tmpPath); } catch {} throw new Error('The URL returned no data.'); }
  return { path: tmpPath, size: written, originalname: name };
}

app.post('/api/transcribe', (req, res, next) => {
  if (shuttingDown) {
    return res.status(503).json({ error: 'Server is updating — try again in a moment.' });
  }
  upload.single('file')(req, res, (err) => {
    if (err) {
      if (err.code === 'LIMIT_FILE_SIZE') {
        return res.status(413).json({ error: `File exceeds max size (${MAX_FILE_SIZE} bytes).` });
      }
      return res.status(400).json({ error: err.message || 'Upload failed.' });
    }
    next();
  });
}, async (req, res) => {
  const auth = req.headers.authorization || '';
  const apiKey = auth.startsWith('Bearer ') ? auth.slice(7).trim() : '';
  if (!apiKey) {
    if (req.file) try { fs.unlinkSync(req.file.path); } catch {}
    return res.status(401).json({ error: 'Missing API key. Add yours in the app to continue.' });
  }
  if (!req.file) return res.status(400).json({ error: 'No file uploaded.' });

  const fileInfo = { path: req.file.path, size: req.file.size, originalname: req.file.originalname };
  return dispatchTranscription(res, fileInfo, apiKey, parseTranscribeOptions(req.body));
});

// ============================================================================
// URL / upload ingestion (for API / MCP callers): instead of multipart-
// uploading bytes in one request, the caller either passes a URL we fetch, or
// an uploadId from a prior /api/uploads slot. Same job pipeline afterward.
// JSON body + Bearer provider key.
// ============================================================================
app.post('/api/transcribe-url', express.json({ limit: '8kb' }), async (req, res) => {
  if (shuttingDown) return res.status(503).json({ error: 'Server is updating — try again in a moment.' });
  const auth = req.headers.authorization || '';
  const apiKey = auth.startsWith('Bearer ') ? auth.slice(7).trim() : '';
  if (!apiKey) return res.status(401).json({ error: 'Missing API key (Bearer).' });

  const audioUrl = (req.body.audioUrl || req.body.url || '').toString().trim();
  const uploadId = (req.body.uploadId || '').toString().trim();
  if (!audioUrl && !uploadId) return res.status(400).json({ error: 'Provide audioUrl or uploadId.' });

  let fileInfo;
  try {
    fileInfo = uploadId ? consumeUpload(uploadId) : await fetchAudioToTemp(audioUrl);
  } catch (e) {
    return res.status(e.status || 400).json({ error: e.message || 'Could not obtain the audio.' });
  }
  return dispatchTranscription(res, fileInfo, apiKey, parseTranscribeOptions(req.body));
});

// Parse + validate the shared transcription options from a request body
// (works for both multipart form fields and JSON).
function parseTranscribeOptions(body) {
  body = body || {};
  const rawLang = (body.language || '').toString().trim().toLowerCase();
  return {
    provider: (body.provider || 'openai').toString().toLowerCase(),
    language: /^[a-z]{2}(_[a-z]{2})?$/.test(rawLang) ? rawLang : '',
    keyterms: (body.keyterms || '').toString().trim().slice(0, 800),
    codeSwitching: body.codeSwitching === '1' || body.codeSwitching === true || body.codeSwitching === 'true',
    csLanguages: (body.languages || '').toString().trim().slice(0, 200),
  };
}

// Validate the key for the chosen provider, create the job, and start it.
// Returns the jobId. Throws { status, message } on validation failure and
// cleans up the temp file. Used directly by the MCP tools and (via the thin
// HTTP wrapper below) by the REST endpoints.
function createTranscriptionJob(fileInfo, apiKey, opts) {
  const { provider, language, keyterms, codeSwitching, csLanguages } = opts;
  const fail = (status, message) => {
    try { fs.unlinkSync(fileInfo.path); } catch {}
    const e = new Error(message); e.status = status; throw e;
  };

  if (provider === 'assemblyai') {
    if (!apiKey || apiKey.length < 20) fail(401, "That key doesn't look right for AssemblyAI.");
    return startAssemblyJob(fileInfo, apiKey, language, keyterms, codeSwitching, csLanguages);
  }

  if (!apiKey || !apiKey.startsWith('sk-') || apiKey.length < 20) {
    fail(401, "That key doesn't look right. OpenAI keys start with sk-");
  }

  const jobId = crypto.randomBytes(12).toString('hex');
  const job = {
    id: jobId,
    provider: 'openai',
    status: 'processing',
    stage: 'analyzing',
    message: 'Analyzing audio',
    progress: 0,
    transcript: '',
    utterances: null,
    segments: null,
    error: null,
    chunkInfo: { current: 0, total: 0, usedSilence: 0 },
    duration: null,
    fileSize: fileInfo.size,
    originalName: fileInfo.originalname,
    inputPath: fileInfo.path,
    workDir: null,
    silences: [],
    pathTaken: '',
    language,
    keyterms,
    createdAt: Date.now(),
    apiKey,
  };
  jobs.set(jobId, job);

  processJob(job).catch(err => {
    console.error(`[${jobId}] error:`, err);
    job.status = 'error';
    job.stage  = 'error';
    job.error  = err.message || String(err);
  });

  return jobId;
}

// Thin HTTP wrapper around createTranscriptionJob for the REST endpoints.
function dispatchTranscription(res, fileInfo, apiKey, opts) {
  try {
    const jobId = createTranscriptionJob(fileInfo, apiKey, opts);
    res.json({ jobId });
  } catch (e) {
    res.status(e.status || 500).json({ error: e.message || 'Could not start transcription.' });
  }
}

// ============================================================================
// AssemblyAI (multi-speaker / diarized) handler
// AssemblyAI's API handles files natively up to 2.2GB — no chunking needed.
// Flow: upload bytes -> submit transcript with speaker_labels -> poll until done.
// ============================================================================

function startAssemblyJob(fileInfo, apiKey, language, keyterms, codeSwitching, csLanguages) {
  const jobId = crypto.randomBytes(12).toString('hex');
  const job = {
    id: jobId,
    provider: 'assemblyai',
    status: 'processing',
    stage: 'uploading',
    message: 'Uploading audio',
    progress: 0,
    transcript: '',
    utterances: null,
    error: null,
    duration: null,
    fileSize: fileInfo.size,
    originalName: fileInfo.originalname,
    inputPath: fileInfo.path,
    workDir: null,
    chunkInfo: { current: 0, total: 0, usedSilence: 0 },
    pathTaken: 'assemblyai',
    language: language || '',
    keyterms: keyterms || '',
    codeSwitching: !!codeSwitching,
    csLanguages: csLanguages || '',
    createdAt: Date.now(),
    apiKey,
    assemblyTranscriptId: null,
  };
  jobs.set(jobId, job);

  processAssemblyJob(job).catch(err => {
    console.error(`[${jobId}] assemblyai error:`, err);
    job.status = 'error';
    job.stage  = 'error';
    job.error  = sanitizeAssemblyError(err);
  });

  return jobId;
}

async function processAssemblyJob(job) {
  // Step 1: upload the file to AssemblyAI's storage
  job.stage = 'uploading';
  job.message = 'Uploading to AssemblyAI';
  job.progress = 0.05;

  const fileStream = fs.createReadStream(job.inputPath);
  const uploadRes = await fetch('https://api.assemblyai.com/v2/upload', {
    method: 'POST',
    headers: {
      'Authorization': job.apiKey,
      'Content-Type': 'application/octet-stream',
    },
    body: fileStream,
    duplex: 'half',
  });
  if (!uploadRes.ok) {
    const errText = await safeText(uploadRes);
    throw new Error(`AssemblyAI upload failed (${uploadRes.status}): ${errText}`);
  }
  const { upload_url } = await uploadRes.json();
  if (!upload_url) throw new Error('AssemblyAI upload did not return a URL.');

  // Step 2: submit the transcript request with diarization on
  job.stage = 'submitting';
  job.message = 'Starting transcription';
  job.progress = 0.15;

  const submitBody = {
    audio_url: upload_url,
    // AssemblyAI now requires speech_models to be set explicitly (the
    // previous default was removed). This pattern follows their docs
    // recommendation: try Universal-3 Pro (best accuracy, supports
    // English/Spanish/French/German/Italian/Portuguese natively); fall
    // back to Universal-2 for any other language. Pricing is $0.21/hr
    // for U3 Pro and $0.15/hr for U2, plus $0.02/hr for diarization.
    speech_models: ['universal-3-pro', 'universal-2'],
    speaker_labels: true,
    // Punctuation and formatting are on by default; explicit for clarity.
    punctuate: true,
    format_text: true,
  };
  if (job.codeSwitching) {
    // Code-switching mode: rely on detection (never a pinned language) and
    // tell Universal-3 Pro to preserve the spoken language mix. The exact
    // prompt below is AssemblyAI's documented phrasing for enabling this.
    submitBody.language_detection = true;
    let prompt = 'The spoken language may change throughout the audio, transcribe in the original language mix (code-switching), preserving the words in the language they are spoken.';
    // Weave in the named languages as a hint when provided.
    const named = job.csLanguages
      ? job.csLanguages.split(',').map(s => s.trim()).filter(Boolean)
      : [];
    if (named.length) {
      prompt += ` The audio mixes the following languages: ${named.join(', ')}.`;
    }
    submitBody.prompt = prompt;
    // Also enable code switching for the Universal-2 fallback path so
    // non-U3Pro languages (e.g. Hebrew) still get multi-language routing.
    submitBody.language_detection_options = { code_switching: true };
  } else if (job.language) {
    // Explicit language: AssemblyAI routes to the right model and rejects
    // with a clear error if a feature isn't supported for that language
    // (better than silently dropping it).
    submitBody.language_code = job.language;
  } else {
    // No language set: detect it. Without this AssemblyAI silently defaults
    // to en_us, which would mangle non-English audio.
    submitBody.language_detection = true;
  }
  if (job.keyterms) {
    // word_boost biases recognition toward supplied terms (names, jargon).
    // Split on commas/newlines, trim, drop empties, cap the list size.
    submitBody.word_boost = job.keyterms
      .split(/[,\n]/).map(s => s.trim()).filter(Boolean).slice(0, 100);
  }

  const submitRes = await fetch('https://api.assemblyai.com/v2/transcript', {
    method: 'POST',
    headers: {
      'Authorization': job.apiKey,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(submitBody),
  });
  if (!submitRes.ok) {
    const errText = await safeText(submitRes);
    throw new Error(`AssemblyAI submit failed (${submitRes.status}): ${errText}`);
  }
  const submitData = await submitRes.json();
  if (!submitData.id) throw new Error('AssemblyAI submit did not return a transcript id.');
  job.assemblyTranscriptId = submitData.id;

  // Step 3: poll for completion. AssemblyAI's polling cadence rec: 3-5 seconds.
  job.stage = 'transcribing';
  job.message = 'Transcribing — this can take a few minutes';
  job.progress = 0.25;

  const startTime = Date.now();
  const MAX_WAIT_MS = 30 * 60 * 1000; // 30 minutes ceiling
  await pollAssemblyToCompletion(job, submitData.id, startTime, MAX_WAIT_MS);
}

// Polls an AssemblyAI transcript by ID until completion or error, populating
// the job fields. Used by both the fresh AssemblyAI flow and the reattach
// flow (where a different server instance / version takes over an existing
// transcript that's still being processed on AssemblyAI).
async function pollAssemblyToCompletion(job, transcriptId, startTime, MAX_WAIT_MS) {
  if (!startTime) startTime = Date.now();
  if (!MAX_WAIT_MS) MAX_WAIT_MS = 30 * 60 * 1000;
  job.assemblyTranscriptId = transcriptId;
  while (true) {
    if (Date.now() - startTime > MAX_WAIT_MS) {
      throw new Error('AssemblyAI took too long to finish (30+ minutes).');
    }
    await sleep(4000);
    const pollRes = await fetch(`https://api.assemblyai.com/v2/transcript/${transcriptId}`, {
      headers: { 'Authorization': job.apiKey },
    });
    if (!pollRes.ok) {
      const errText = await safeText(pollRes);
      throw new Error(`AssemblyAI poll failed (${pollRes.status}): ${errText}`);
    }
    const pollData = await pollRes.json();

    if (pollData.status === 'completed') {
      const utterances = Array.isArray(pollData.utterances)
        ? pollData.utterances.map(u => ({
            speaker: u.speaker || 'A',
            text: u.text || '',
            start_ms: typeof u.start === 'number' ? u.start : 0,
            end_ms: typeof u.end === 'number' ? u.end : 0,
            confidence: typeof u.confidence === 'number' ? u.confidence : null,
            words: Array.isArray(u.words)
              ? u.words.map(w => ({
                  text: w.text || '',
                  confidence: typeof w.confidence === 'number' ? w.confidence : null,
                }))
              : null,
          }))
        : [];
      job.utterances = utterances;
      job.transcript = pollData.text || '';
      job.duration = typeof pollData.audio_duration === 'number' ? pollData.audio_duration : null;
      job.status = 'done';
      job.stage = 'done';
      job.message = 'Done';
      job.progress = 1;
      cleanupJob(job);
      return;
    }
    if (pollData.status === 'error') {
      throw new Error(pollData.error || 'AssemblyAI returned an error.');
    }
    const elapsed = Date.now() - startTime;
    const estimatedAudioSec = Math.max(10, (job.fileSize || 100000) / 16000);
    const expectedMs = Math.min(600000, Math.max(15000, estimatedAudioSec * 100));
    const frac = Math.min(1, elapsed / expectedMs);
    job.progress = Math.min(0.95, 0.25 + frac * 0.70);
    job.stage = 'transcribing';
    job.message = 'Transcribing — this can take a few minutes';
  }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
async function safeText(res) { try { return await res.text(); } catch { return ''; } }

function sanitizeAssemblyError(err) {
  const msg = (err && err.message) || String(err) || 'Transcription failed.';
  // Strip internal stack-like text. Common patterns to surface verbatim:
  if (/401|unauthor/i.test(msg)) return 'AssemblyAI rejected the API key. Double-check it.';
  if (/insufficient|payment|quota/i.test(msg)) return 'AssemblyAI says the account is out of credit or unpaid.';
  if (/took too long/i.test(msg)) return msg;
  // Otherwise return a sanitized prefix
  return msg.length > 200 ? msg.slice(0, 200) + '…' : msg;
}

function serializeJob(job) {
  return {
    id: job.id,
    provider: job.provider || 'openai',
    status: job.status,
    stage: job.stage,
    message: job.message,
    progress: job.progress,
    transcript: job.transcript,
    utterances: job.utterances,
    segments: job.segments || null,
    // The AssemblyAI transcript ID is the "claim ticket" — once the client
    // has it, the work survives any restart of this server because the
    // transcript itself lives on AssemblyAI. See /api/assembly/reattach.
    assemblyTranscriptId: job.assemblyTranscriptId || null,
    error: job.error,
    chunkInfo: job.chunkInfo,
    duration: job.duration,
    fileSize: job.fileSize,
    originalName: job.originalName,
    pathTaken: job.pathTaken,
  };
}

app.get('/api/jobs/:id', jobsLimiter, (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: 'Job not found.' });
  res.json(serializeJob(job));
});

// ============================================================================
// Re-attach: pick up an existing AssemblyAI transcript by its ID. This is how
// in-flight Multiple-speakers jobs survive a server deploy/restart — the work
// itself runs on AssemblyAI's servers, so as long as the client kept the
// transcript ID (it persists it to localStorage from the job status), any
// version of this server can resume polling and return the result.
//
// The client supplies its own AssemblyAI key as Bearer; AssemblyAI only
// returns transcripts created with that key, so a user can never reach
// another user's transcript here.
// ============================================================================
app.post('/api/assembly/reattach', express.json({ limit: '64kb' }), (req, res) => {
  if (shuttingDown) return res.status(503).json({ error: 'Server is updating — try again in a moment.' });
  const auth = req.headers.authorization || '';
  const apiKey = auth.startsWith('Bearer ') ? auth.slice(7).trim() : '';
  if (!apiKey) return res.status(401).json({ error: 'AssemblyAI key required.' });

  const transcriptId = (req.body.transcriptId || '').toString().trim();
  // AssemblyAI transcript IDs are URL-safe lowercase alphanumerics + dashes.
  // Validate the shape so an invalid value can't be smuggled into the URL.
  if (!transcriptId || !/^[a-z0-9_-]{8,80}$/i.test(transcriptId)) {
    return res.status(400).json({ error: 'Invalid transcriptId.' });
  }

  const jobId = crypto.randomBytes(12).toString('hex');
  const job = {
    id: jobId,
    provider: 'assemblyai',
    status: 'processing',
    stage: 'transcribing',
    message: 'Reconnecting to AssemblyAI',
    progress: 0.3,
    transcript: '',
    utterances: null,
    error: null,
    duration: null,
    fileSize: Number(req.body.fileSize) || 0,
    originalName: (req.body.fileName || 'Recording').toString().slice(0, 200),
    inputPath: null,   // no local file — the audio already lives on AssemblyAI
    workDir: null,
    chunkInfo: { current: 0, total: 0, usedSilence: 0 },
    pathTaken: 'assemblyai-reattach',
    language: '',
    keyterms: '',
    codeSwitching: false,
    csLanguages: '',
    createdAt: Date.now(),
    apiKey,
    assemblyTranscriptId: transcriptId,
  };
  jobs.set(jobId, job);

  // Fire-and-forget poll, same model as the normal flow
  pollAssemblyToCompletion(job, transcriptId).catch(err => {
    console.error(`[${jobId}] reattach error:`, err);
    job.status = 'error';
    job.stage  = 'error';
    job.error  = sanitizeAssemblyError(err);
  });

  res.json({ jobId });
});

// ============================================================================
// Post-processing: summary / action items / translation
// Uses the user's OpenAI key (Bearer header). Transcript is sent in the
// request body — larger json limit applied to this route only.
// ============================================================================
const POST_PROCESS_MODELS = ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1'];

// Core summary / action-items / translation logic. Returns a result object.
// Throws Error on failure. Shared by the REST route and the MCP tool.
async function runPostProcess(apiKey, { transcript, tasks, targetLanguage, model }) {
  const text = (transcript || '').toString().trim().slice(0, 400000);
  tasks = Array.isArray(tasks) ? tasks : [];
  targetLanguage = (targetLanguage || '').toString().trim().slice(0, 40);
  model = POST_PROCESS_MODELS.includes(model) ? model : 'gpt-4o-mini';
  if (!text) { const e = new Error('No transcript provided.'); e.status = 400; throw e; }
  if (!tasks.length) { const e = new Error('No tasks requested.'); e.status = 400; throw e; }

  const client = new OpenAI({ apiKey });
  const result = {};
  const wantSummary = tasks.includes('summary');
  const wantActions = tasks.includes('actions');

  if (wantSummary || wantActions) {
    const fields = [];
    if (wantSummary) fields.push(
      '"summary": a thorough summary written as 2-4 short paragraphs (not bullet points). ' +
      'Open with one sentence stating what the conversation was and who was involved, then cover the ' +
      'main topics discussed, any decisions or conclusions reached, points of disagreement or open ' +
      'questions, and notable details, context, numbers, names, or examples that came up. Be specific ' +
      'and substantive — capture the actual content, not vague generalities. Aim for real depth while ' +
      'staying readable.'
    );
    if (wantActions) fields.push(
      '"actionItems": an array of the concrete next steps from this conversation. Apply these rules ' +
      'consistently so the same transcript always yields the same list:\n' +
      '  • INCLUDE a step only if someone explicitly committed to it, was asked/assigned to do it, or the ' +
      'group clearly agreed it should happen. Look for language like "I\'ll...", "we need to...", "can you...", ' +
      '"let\'s...", "the next step is...", "by [date]".\n' +
      '  • EXCLUDE general discussion, opinions, background, ideas mentioned in passing, and things that were ' +
      'considered but not agreed. If it was only talked about, it is NOT an action item.\n' +
      '  • Write each item as one short imperative sentence starting with a verb, self-contained. Include the ' +
      'responsible person and any deadline when the transcript states them. Do not invent or infer tasks that ' +
      'were not actually raised. Merge duplicates. Order them by the order they came up.\n' +
      '  • If there are genuinely no committed next steps, return an empty array — do not pad the list.'
    );
    const sys = 'You are a precise meeting/transcript analyst. You read carefully and capture specifics — ' +
      'names, numbers, decisions, and concrete next steps — rather than generic summaries. Respond with a ' +
      'single JSON object and nothing else.';
    const prompt =
      `Analyze the transcript below and produce a JSON object with these fields:\n` +
      fields.map(f => '- ' + f).join('\n') +
      `\n\nReturn only valid JSON. Transcript:\n"""\n${text}\n"""`;
    const completion = await client.chat.completions.create({
      model,
      messages: [{ role: 'system', content: sys }, { role: 'user', content: prompt }],
      response_format: { type: 'json_object' },
      max_tokens: 2000,
      temperature: 0,
      seed: 1234,
    });
    let parsed = {};
    try { parsed = JSON.parse(completion.choices[0].message.content); } catch {}
    if (wantSummary) result.summary = (parsed.summary || '').toString().trim();
    if (wantActions) result.actionItems = Array.isArray(parsed.actionItems)
      ? parsed.actionItems.map(s => s.toString().trim()).filter(Boolean) : [];
  }

  if (tasks.includes('translate')) {
    const lang = targetLanguage || 'English';
    const completion = await client.chat.completions.create({
      model,
      messages: [
        { role: 'system', content: `You are a translator. Translate the user's transcript into ${lang}. Preserve speaker labels and line breaks. Output only the translation, no preamble.` },
        { role: 'user', content: text },
      ],
      max_tokens: 8000,
      temperature: 0,
      seed: 1234,
    });
    result.translation = (completion.choices[0].message.content || '').trim();
    result.translationLanguage = lang;
  }

  return result;
}

function sanitizeOpenAiError(e) {
  const msg = (e && (e.message || (e.error && e.error.message))) || 'Post-processing failed.';
  if (/api key/i.test(msg) || /invalid_api_key/i.test(msg)) return 'OpenAI rejected the key.';
  if (/quota/i.test(msg) || /insufficient/i.test(msg)) return 'Your OpenAI account is out of quota for this request.';
  if (/rate limit/i.test(msg)) return 'OpenAI rate limit hit — wait a moment and retry.';
  return msg;
}

app.post('/api/post-process', postProcessLimiter, express.json({ limit: '4mb' }), async (req, res) => {
  const auth = req.headers.authorization || '';
  const apiKey = auth.startsWith('Bearer ') ? auth.slice(7).trim() : '';
  if (!apiKey.startsWith('sk-') || apiKey.length < 20) {
    return res.status(401).json({ error: 'A valid OpenAI key is required for summary, action items, and translation.' });
  }
  try {
    const result = await runPostProcess(apiKey, {
      transcript: req.body.transcript,
      tasks: req.body.tasks,
      targetLanguage: req.body.targetLanguage,
      model: req.body.model,
    });
    res.json(result);
  } catch (e) {
    res.status(e.status || 502).json({ error: e.status ? e.message : sanitizeOpenAiError(e) });
  }
});

// ============================================================================
// Cleanup
// ============================================================================

setInterval(() => {
  const cutoff = Date.now() - 60 * 60 * 1000;
  for (const [id, job] of jobs) {
    if (job.createdAt < cutoff) {
      cleanupJob(job);
      jobs.delete(id);
    }
  }
  // Expire unused upload slots (filled-but-never-transcribed, or abandoned).
  const upCutoff = Date.now() - UPLOAD_TTL_MS;
  for (const [id, slot] of uploads) {
    if (slot.createdAt < upCutoff) {
      try { if (fs.existsSync(slot.path)) fs.unlinkSync(slot.path); } catch {}
      uploads.delete(id);
    }
  }
}, 10 * 60 * 1000);

function cleanupJob(job) {
  try { if (job.inputPath && fs.existsSync(job.inputPath)) fs.unlinkSync(job.inputPath); } catch {}
  try { if (job.workDir && fs.existsSync(job.workDir)) fs.rmSync(job.workDir, { recursive: true, force: true }); } catch {}
  job.apiKey = null; // wipe key from memory
}

// ============================================================================
// Pipeline — picks one of three paths
// ============================================================================

async function processJob(job) {
  // Wait for a free CPU slot before doing any heavy ffmpeg/Whisper work. The
  // job already exists and is pollable; while it waits it simply reports
  // 'queued' so the client shows it in line. On the free plan this serializes
  // jobs to one at a time; on bigger servers it scales (see MAX_CONCURRENT_JOBS).
  if (activeHeavyJobs >= MAX_CONCURRENT_JOBS) {
    job.stage = 'queued';
    job.message = 'Waiting for a free slot on the server';
  }
  await acquireHeavySlot();
  let slotReleased = false;
  const release = () => { if (!slotReleased) { slotReleased = true; releaseHeavySlot(); } };

  try {
    job.workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tx-'));
    job.stage = 'analyzing';
    job.message = 'Reading audio metadata';
    job.duration = await ffprobeDuration(job.inputPath);
    // duration is null if even the decode-fallback couldn't determine it.
    // Don't fail outright — fall back to heuristics based on size + format.

    const ext = path.extname(job.originalName).toLowerCase();
    const isCompressed = COMPRESSED_EXTS.includes(ext);
    const isSmall      = job.fileSize <= DIRECT_TO_WHISPER_MAX_BYTES;
    // If duration is unknown but the file is compressed and small enough,
    // assume it's short enough to send direct. Whisper has its own
    // size+length limits and will reject if we're wrong.
    const isShort      = job.duration == null
                         ? (isCompressed && isSmall)
                         : (job.duration <= MAX_CHUNK);

    if (isCompressed && isSmall && isShort) {
      // PATH A: send directly to Whisper, no ffmpeg at all
      job.pathTaken = 'direct';
      job.stage = 'transcribing';
      job.message = 'Transcribing (fast path)';
      job.chunkInfo = { current: 1, total: 1, usedSilence: 0 };
      const r = await transcribeFile(job.inputPath, job.apiKey, job.language, job.keyterms);
      job.transcript = r.text;
      job.segments = r.segments;
      job.progress = 1;
      job.status = 'done';
      job.stage  = 'done';
      job.message = 'Complete';
      return;
    }

    let mediaPath;
    let splitPoints = [];

    if (isCompressed) {
      // PATH B: already compressed — detect silences only, then split with -c copy
      job.pathTaken = 'split-only';
      job.stage = 'analyzing';
      job.message = `Analyzing ${formatTime(job.duration)} for silence points`;
      await detectSilencesOnly(job);
      mediaPath = job.inputPath;
    } else {
      // PATH C: full pipeline — re-encode + detect silences in one pass
      job.pathTaken = 'full';
      job.stage = 'compressing';
      job.message = `Compressing ${formatTime(job.duration)} of audio`;
      mediaPath = path.join(job.workDir, 'compressed.mp3');
      await compressAndDetectSilences(job, mediaPath);
    }

    // Decide chunks
    let chunkPaths;
    let chunkOffsets; // start time (seconds) of each chunk within the original
    if (job.duration <= MAX_CHUNK) {
      chunkPaths = [mediaPath];
      chunkOffsets = [0];
      job.chunkInfo = { current: 0, total: 1, usedSilence: 0 };
    } else {
      job.stage = 'splitting';
      job.message = 'Splitting at silence boundaries';
      splitPoints = pickSplitPoints(job.silences, job.duration);
      const silenceAligned = splitPoints.filter(
        t => job.silences.some(s => Math.abs(s.mid - t) < 0.01)
      ).length;
      chunkPaths = await splitAudio(mediaPath, splitPoints, job.workDir, ext);
      // Chunk i starts at splitPoints[i-1] (chunk 0 starts at 0). These are
      // the offsets we add to each chunk's segment timestamps so they line
      // up with the original audio timeline.
      chunkOffsets = [0, ...splitPoints];
      job.chunkInfo = { current: 0, total: chunkPaths.length, usedSilence: silenceAligned };
    }

    // Transcribe chunks sequentially
    job.stage = 'transcribing';
    let combined = '';
    job.segments = [];
    for (let i = 0; i < chunkPaths.length; i++) {
      job.chunkInfo = { ...job.chunkInfo, current: i + 1 };
      job.progress = 0.5 + (i / chunkPaths.length) * 0.5;
      job.message = `Transcribing chunk ${i + 1} of ${chunkPaths.length}`;
      const r = await transcribeFile(chunkPaths[i], job.apiKey, job.language, job.keyterms);
      combined = combined ? `${combined} ${r.text}` : r.text;
      job.transcript = combined;
      const off = chunkOffsets[i] || 0;
      for (const seg of r.segments) {
        job.segments.push({ start: seg.start + off, end: seg.end + off, text: seg.text });
      }
    }

    job.status = 'done';
    job.stage  = 'done';
    job.progress = 1;
    job.message = 'Complete';
  } finally {
    // Free the CPU slot as soon as processing ends (success or failure) so the
    // next queued job can start — independent of the delayed temp-file cleanup.
    release();
    setTimeout(() => cleanupJob(job), 60 * 1000);
  }
}

// ============================================================================
// ffmpeg helpers
// ============================================================================

function ffprobeDuration(filePath) {
  // Fast path: read the container's format-level duration. Works for any
  // file properly muxed with a duration header (most uploads).
  return new Promise((resolve) => {
    const proc = spawn('ffprobe', [
      '-v', 'error',
      '-show_entries', 'format=duration',
      '-of', 'default=noprint_wrappers=1:nokey=1',
      filePath,
    ]);
    let out = '', err = '';
    proc.stdout.on('data', d => { out += d.toString(); });
    proc.stderr.on('data', d => { err += d.toString(); });
    proc.on('close', code => {
      if (code === 0) {
        const dur = parseFloat(out.trim());
        if (isFinite(dur) && dur > 0) return resolve(dur);
      }
      if (err) console.error(`[ffprobe] no header duration: ${err.slice(0, 300)}`);
      // Fall through to decode-based probe (handles browser-recorded webm
      // and other live-streamed containers that lack a duration header).
      ffmpegDecodeDuration(filePath).then(resolve).catch(() => resolve(null));
    });
    proc.on('error', e => {
      console.error(`[ffprobe] spawn error: ${e.message}`);
      resolve(null);
    });
  });
}

// Decode-based duration probe: ffmpeg -f null - runs the file through the
// decoder and prints the running time. Reliable but slower (proportional
// to the audio length — typically a few hundred ms for a short recording).
// Used as fallback when the container has no duration header (common for
// MediaRecorder webm output).
function ffmpegDecodeDuration(filePath) {
  return new Promise((resolve) => {
    const proc = spawn('ffmpeg', ['-i', filePath, '-f', 'null', '-']);
    let err = '';
    proc.stderr.on('data', d => { err += d.toString(); });
    proc.on('close', () => {
      // ffmpeg emits 'time=HH:MM:SS.ms' lines; the final one is the true length
      const matches = [...err.matchAll(/time=(\d{2}):(\d{2}):(\d{2}\.\d+)/g)];
      if (matches.length === 0) return resolve(null);
      const last = matches[matches.length - 1];
      const dur = (parseInt(last[1]) * 3600) + (parseInt(last[2]) * 60) + parseFloat(last[3]);
      resolve(dur > 0 ? dur : null);
    });
    proc.on('error', () => resolve(null));
  });
}

function parseSilenceLines(job, chunk) {
  for (const line of chunk.split('\n')) {
    const sM = line.match(/silence_start:\s*(-?[\d.]+)/);
    const eM = line.match(/silence_end:\s*([\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)/);
    if (sM) job._pendingSilence = { start: Math.max(0, parseFloat(sM[1])) };
    if (eM) {
      const end = parseFloat(eM[1]);
      const dur = parseFloat(eM[2]);
      if (job._pendingSilence) {
        const start = job._pendingSilence.start;
        job.silences.push({ start, end, duration: dur, mid: (start + end) / 2 });
        job._pendingSilence = null;
      } else {
        const start = Math.max(0, end - dur);
        job.silences.push({ start, end, duration: dur, mid: (start + end) / 2 });
      }
    }
  }
}

function compressAndDetectSilences(job, outputPath) {
  return new Promise((resolve, reject) => {
    const args = [
      '-i', job.inputPath,
      '-vn',
      '-ac', '1',
      '-ar', String(OUT_SAMPLE_RATE),
      '-b:a', OUT_BITRATE,
      '-c:a', 'libmp3lame',
      '-af', `silencedetect=n=${SILENCE_DB}dB:d=${SILENCE_MIN_SEC}`,
      '-progress', 'pipe:1',
      '-y',
      outputPath,
    ];
    const proc = spawn('ffmpeg', args);
    let stderrTail = '';

    proc.stderr.on('data', (data) => {
      const chunk = data.toString();
      stderrTail = (stderrTail + chunk).slice(-2000);
      parseSilenceLines(job, chunk);
    });

    proc.stdout.on('data', (data) => {
      const m = data.toString().match(/out_time_ms=(\d+)/);
      if (m && job.duration) {
        const elapsed = parseInt(m[1], 10) / 1_000_000;
        job.progress = Math.max(0, Math.min(0.5, (elapsed / job.duration) * 0.5));
      }
    });

    proc.on('close', code => {
      if (code === 0) { job.progress = 0.5; resolve(); }
      else {
        console.error(`[ffmpeg compress] exit ${code}:\n${stderrTail}`);
        reject(new Error('Audio compression failed. The file may be corrupted or in an unsupported format.'));
      }
    });
    proc.on('error', e => {
      console.error(`[ffmpeg compress] spawn error: ${e.message}`);
      reject(new Error('Audio processing tool is unavailable on this server.'));
    });
  });
}

function detectSilencesOnly(job) {
  return new Promise((resolve, reject) => {
    // Run silencedetect with null output — decode-only, no re-encode. Fast.
    const args = [
      '-i', job.inputPath,
      '-vn',
      '-af', `silencedetect=n=${SILENCE_DB}dB:d=${SILENCE_MIN_SEC}`,
      '-f', 'null',
      '-progress', 'pipe:1',
      '-y',
      '-',
    ];
    const proc = spawn('ffmpeg', args);
    let stderrTail = '';

    proc.stderr.on('data', (data) => {
      const chunk = data.toString();
      stderrTail = (stderrTail + chunk).slice(-2000);
      parseSilenceLines(job, chunk);
    });

    proc.stdout.on('data', (data) => {
      const m = data.toString().match(/out_time_ms=(\d+)/);
      if (m && job.duration) {
        const elapsed = parseInt(m[1], 10) / 1_000_000;
        job.progress = Math.max(0, Math.min(0.5, (elapsed / job.duration) * 0.5));
      }
    });

    proc.on('close', code => {
      if (code === 0) { job.progress = 0.5; resolve(); }
      else {
        console.error(`[ffmpeg silence] exit ${code}:\n${stderrTail}`);
        reject(new Error('Audio analysis failed. The file may be corrupted.'));
      }
    });
    proc.on('error', e => {
      console.error(`[ffmpeg silence] spawn error: ${e.message}`);
      reject(new Error('Audio processing tool is unavailable on this server.'));
    });
  });
}

function pickSplitPoints(silences, totalDuration) {
  const splits = [];
  let pos = 0;
  while (totalDuration - pos > MAX_CHUNK) {
    const winStart = pos + MIN_CHUNK;
    const winEnd   = pos + MAX_CHUNK;
    const target   = pos + TARGET_CHUNK;
    const inWin = silences.filter(s => s.mid >= winStart && s.mid <= winEnd);
    if (inWin.length > 0) {
      inWin.sort((a, b) => Math.abs(a.mid - target) - Math.abs(b.mid - target));
      splits.push(inWin[0].mid);
      pos = inWin[0].mid;
    } else {
      splits.push(winEnd);
      pos = winEnd;
    }
  }
  return splits;
}

function splitAudio(inputPath, splitPoints, workDir, srcExt) {
  return new Promise((resolve, reject) => {
    // Choose output extension based on input — keeps -c copy honest for the container.
    const ext = (srcExt && COMPRESSED_EXTS.includes(srcExt)) ? srcExt : '.mp3';
    const pattern = path.join(workDir, `chunk_%03d${ext}`);
    const args = [
      '-i', inputPath,
      '-vn',
      '-f', 'segment',
      '-segment_times', splitPoints.map(t => t.toFixed(3)).join(','),
      '-c', 'copy',
      '-reset_timestamps', '1',
      '-y',
      pattern,
    ];
    const proc = spawn('ffmpeg', args);
    let stderrTail = '';
    proc.stderr.on('data', d => { stderrTail = (stderrTail + d.toString()).slice(-2000); });
    proc.on('close', code => {
      if (code !== 0) {
        console.error(`[ffmpeg split] exit ${code}:\n${stderrTail}`);
        return reject(new Error('Audio splitting failed. The file may be malformed.'));
      }
      const chunks = [];
      let i = 0;
      while (true) {
        const p = path.join(workDir, `chunk_${String(i).padStart(3, '0')}${ext}`);
        if (fs.existsSync(p)) { chunks.push(p); i++; } else break;
      }
      if (chunks.length === 0) return reject(new Error('No chunks were produced from the audio.'));
      resolve(chunks);
    });
    proc.on('error', e => {
      console.error(`[ffmpeg split] spawn error: ${e.message}`);
      reject(new Error('Audio processing tool is unavailable on this server.'));
    });
  });
}

// ============================================================================
// Whisper — uses the official OpenAI SDK so multipart is encoded correctly.
// This replaces the previous form-data + native fetch combo that produced
// "Could not parse multipart form" errors.
// ============================================================================

async function transcribeFile(filePath, apiKey, language, keyterms) {
  const client = new OpenAI({ apiKey });
  try {
    const params = {
      file: fs.createReadStream(filePath),
      model: 'whisper-1',
      // verbose_json gives us segment-level timestamps (start/end seconds)
      // in addition to the text — needed for SRT/VTT export and synced
      // playback. Falls back gracefully if segments are absent.
      response_format: 'verbose_json',
    };
    // Whisper auto-detects language by default; passing a hint when the
    // user has explicitly chosen one skips that step and slightly improves
    // accuracy, especially for low-resource languages.
    if (language) params.language = language;
    // The prompt biases Whisper toward specific spellings — names, jargon,
    // acronyms. Capped to stay well under the 224-token prompt limit.
    if (keyterms) params.prompt = keyterms.slice(0, 800);
    const result = await client.audio.transcriptions.create(params);
    const text = (result && result.text || '').trim();
    const segments = Array.isArray(result.segments)
      ? result.segments.map(s => ({
          start: typeof s.start === 'number' ? s.start : 0,
          end: typeof s.end === 'number' ? s.end : 0,
          text: (s.text || '').trim(),
        }))
      : [];
    return { text, segments };
  } catch (e) {
    // Surface OpenAI's own error message when possible
    const msg = (e && (e.message || (e.error && e.error.message))) || String(e);
    throw new Error(msg);
  }
}

// ============================================================================
// Utils
// ============================================================================

function formatTime(s) {
  if (s == null || !isFinite(s)) return 'audio';
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  const mm = String(m).padStart(2, '0'), ss = String(sec).padStart(2, '0');
  return h > 0 ? `${h}:${mm}:${ss}` : `${m}:${ss}`;
}

// ============================================================================
// Self-registration (phone home)
// ============================================================================

async function registerSelf() {
  if (!PAIRING_CODE) {
    console.log('No PAIRING_CODE set — skipping self-registration (this server runs standalone).');
    return;
  }
  if (!RENDER_EXTERNAL_URL) {
    console.log('PAIRING_CODE set but RENDER_EXTERNAL_URL missing — cannot determine own URL.');
    return;
  }
  // Don't have the central server register with itself
  if (REGISTER_URL.replace(/\/+$/, '') === RENDER_EXTERNAL_URL.replace(/\/+$/, '')) {
    console.log('I am the central registry — skipping self-registration.');
    return;
  }

  const code = PAIRING_CODE.trim().toLowerCase();
  for (let attempt = 1; attempt <= 6; attempt++) {
    try {
      const res = await fetch(`${REGISTER_URL}/api/pair/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, url: RENDER_EXTERNAL_URL }),
      });
      if (res.ok) {
        console.log(`[pair] ✓ Phoned home: ${code} -> ${RENDER_EXTERNAL_URL}`);
        return;
      }
      const body = await res.text().catch(() => '');
      console.log(`[pair] attempt ${attempt} got ${res.status}: ${body.slice(0, 200)}`);
    } catch (e) {
      console.log(`[pair] attempt ${attempt} network error: ${e.message}`);
    }
    await new Promise(r => setTimeout(r, Math.min(30_000, 3_000 * attempt)));
  }
  console.log(`[pair] gave up after 6 attempts. User may need to paste server URL manually.`);
}

// ============================================================================
// Start
// ============================================================================

// Mount the MCP server (POST /mcp). Tools call the in-process service
// functions directly; provider keys arrive via connector headers.
mountMcp(app, express, {
  createTranscriptionJob,
  fetchAudioToTemp,
  consumeUpload,
  publicBaseUrl,
  createUploadSlot: (fileName) => {
    const uploadId = crypto.randomBytes(18).toString('hex');
    const name = (fileName || 'audio').toString().replace(/[^\w.\-]/g, '_').slice(0, 100) || 'audio';
    const tmpPath = path.join(os.tmpdir(), 'up-' + uploadId + '-' + name);
    uploads.set(uploadId, { path: tmpPath, ready: false, size: 0, name, createdAt: Date.now() });
    return uploadId;
  },
  jobs,
  serializeJob,
  runPostProcess,
  isShuttingDown: () => shuttingDown,
});

app.listen(PORT, () => {
  console.log(`Transcribe server listening on :${PORT}`);
  console.log('Each user supplies their own OpenAI key from the browser.');
  if (PAIRING_CODE) console.log(`Phoning home with pairing code: ${PAIRING_CODE}`);
  registerSelf();
});

// Graceful shutdown: when Render sends SIGTERM ahead of a deploy/restart,
// flip the drain flag so new submissions get a clean 'server updating'
// response. In-flight polling and reattach calls keep working until the
// container is killed. The actual Multiple-speakers jobs survive a restart
// because their work lives on AssemblyAI — the client persists the
// transcript ID and calls /api/assembly/reattach against the new instance.
function onShutdownSignal(sig) {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log(`[${sig}] draining: refusing new submissions; existing polls/reattach still served`);
}
process.on('SIGTERM', () => onShutdownSignal('SIGTERM'));
process.on('SIGINT',  () => onShutdownSignal('SIGINT'));
