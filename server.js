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
const OpenAI = require('openai');

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
const transcribeLimiter = rateLimit({
  windowMs: 60 * 60 * 1000, max: 30, // 30 transcribe jobs per hour per IP
  standardHeaders: true, legacyHeaders: false,
  message: { error: 'You\'ve hit the hourly transcription limit on this server.' },
});
const jobsLimiter = rateLimit({
  windowMs: 60 * 1000, max: 240, // polling can be busy
  standardHeaders: true, legacyHeaders: false,
});

app.use(express.json({ limit: '64kb' }));
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

app.post('/api/pair/register', pairRegisterLimiter, (req, res) => {
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

app.get('/health', (req, res) => res.json({ ok: true }));

app.post('/api/transcribe', transcribeLimiter, (req, res, next) => {
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

  // Branch on provider. Default is OpenAI Whisper (single-speaker).
  const provider = (req.body.provider || 'openai').toLowerCase();

  if (provider === 'assemblyai') {
    // AssemblyAI keys are 32-char hex-ish strings, no required prefix
    if (apiKey.length < 20) {
      try { fs.unlinkSync(req.file.path); } catch {}
      return res.status(401).json({ error: "That key doesn't look right for AssemblyAI." });
    }
    return startAssemblyJob(req, res, apiKey);
  }

  // OpenAI Whisper path (existing behavior)
  if (!apiKey.startsWith('sk-') || apiKey.length < 20) {
    try { fs.unlinkSync(req.file.path); } catch {}
    return res.status(401).json({ error: "That key doesn't look right. OpenAI keys start with sk-" });
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
    utterances: null, // diarized output (AssemblyAI only)
    error: null,
    chunkInfo: { current: 0, total: 0, usedSilence: 0 },
    duration: null,
    fileSize: req.file.size,
    originalName: req.file.originalname,
    inputPath: req.file.path,
    workDir: null,
    silences: [],
    pathTaken: '',   // 'direct' | 'split-only' | 'full'
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

  res.json({ jobId });
});

// ============================================================================
// AssemblyAI (multi-speaker / diarized) handler
// AssemblyAI's API handles files natively up to 2.2GB — no chunking needed.
// Flow: upload bytes -> submit transcript with speaker_labels -> poll until done.
// ============================================================================

function startAssemblyJob(req, res, apiKey) {
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
    fileSize: req.file.size,
    originalName: req.file.originalname,
    inputPath: req.file.path,
    workDir: null,
    chunkInfo: { current: 0, total: 0, usedSilence: 0 },
    pathTaken: 'assemblyai',
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

  res.json({ jobId });
}

async function processAssemblyJob(job) {
  // Step 1: upload the file to AssemblyAI's storage
  job.stage = 'uploading';
  job.message = 'Uploading to AssemblyAI';
  job.progress = 5;

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
  job.progress = 15;

  const submitRes = await fetch('https://api.assemblyai.com/v2/transcript', {
    method: 'POST',
    headers: {
      'Authorization': job.apiKey,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      audio_url: upload_url,
      speaker_labels: true,
      // Punctuation and formatting are on by default; explicit for clarity.
      punctuate: true,
      format_text: true,
    }),
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
  job.progress = 25;

  const startTime = Date.now();
  const MAX_WAIT_MS = 30 * 60 * 1000; // 30 minutes ceiling
  while (true) {
    if (Date.now() - startTime > MAX_WAIT_MS) {
      throw new Error('AssemblyAI took too long to finish (30+ minutes).');
    }
    await sleep(4000);
    const pollRes = await fetch(`https://api.assemblyai.com/v2/transcript/${submitData.id}`, {
      headers: { 'Authorization': job.apiKey },
    });
    if (!pollRes.ok) {
      const errText = await safeText(pollRes);
      throw new Error(`AssemblyAI poll failed (${pollRes.status}): ${errText}`);
    }
    const pollData = await pollRes.json();

    if (pollData.status === 'completed') {
      // Build the utterances array for the client. Each utterance:
      // { speaker, text, start_ms, end_ms }
      const utterances = Array.isArray(pollData.utterances)
        ? pollData.utterances.map(u => ({
            speaker: u.speaker || 'A',
            text: u.text || '',
            start_ms: typeof u.start === 'number' ? u.start : 0,
            end_ms: typeof u.end === 'number' ? u.end : 0,
          }))
        : [];
      job.utterances = utterances;
      job.transcript = pollData.text || '';
      job.duration = typeof pollData.audio_duration === 'number' ? pollData.audio_duration : null;
      job.status = 'done';
      job.stage = 'done';
      job.message = 'Done';
      job.progress = 100;
      cleanupJob(job);
      return;
    }
    if (pollData.status === 'error') {
      throw new Error(pollData.error || 'AssemblyAI returned an error.');
    }
    // queued, processing -> ease progress upward but cap before "done"
    const elapsed = Date.now() - startTime;
    job.progress = Math.min(90, 25 + Math.floor(elapsed / 1000)); // 1% per second up to 90%
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

app.get('/api/jobs/:id', jobsLimiter, (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: 'Job not found.' });
  res.json({
    id: job.id,
    provider: job.provider || 'openai',
    status: job.status,
    stage: job.stage,
    message: job.message,
    progress: job.progress,
    transcript: job.transcript,
    utterances: job.utterances,
    error: job.error,
    chunkInfo: job.chunkInfo,
    duration: job.duration,
    fileSize: job.fileSize,
    originalName: job.originalName,
    pathTaken: job.pathTaken,
  });
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
  job.workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tx-'));

  try {
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
      job.transcript = await transcribeFile(job.inputPath, job.apiKey);
      job.progress = 1;
      job.status = 'done';
      job.stage  = 'done';
      job.message = 'Complete';
      return;
    }

    let mediaPath;

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
    if (job.duration <= MAX_CHUNK) {
      chunkPaths = [mediaPath];
      job.chunkInfo = { current: 0, total: 1, usedSilence: 0 };
    } else {
      job.stage = 'splitting';
      job.message = 'Splitting at silence boundaries';
      const splitPoints = pickSplitPoints(job.silences, job.duration);
      const silenceAligned = splitPoints.filter(
        t => job.silences.some(s => Math.abs(s.mid - t) < 0.01)
      ).length;
      chunkPaths = await splitAudio(mediaPath, splitPoints, job.workDir, ext);
      job.chunkInfo = { current: 0, total: chunkPaths.length, usedSilence: silenceAligned };
    }

    // Transcribe chunks sequentially
    job.stage = 'transcribing';
    let combined = '';
    for (let i = 0; i < chunkPaths.length; i++) {
      job.chunkInfo = { ...job.chunkInfo, current: i + 1 };
      job.progress = 0.5 + (i / chunkPaths.length) * 0.5;
      job.message = `Transcribing chunk ${i + 1} of ${chunkPaths.length}`;
      const text = await transcribeFile(chunkPaths[i], job.apiKey);
      combined = combined ? `${combined} ${text}` : text;
      job.transcript = combined;
    }

    job.status = 'done';
    job.stage  = 'done';
    job.progress = 1;
    job.message = 'Complete';
  } finally {
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

async function transcribeFile(filePath, apiKey) {
  const client = new OpenAI({ apiKey });
  try {
    const result = await client.audio.transcriptions.create({
      file: fs.createReadStream(filePath),
      model: 'whisper-1',
      response_format: 'text',
    });
    return (typeof result === 'string' ? result : (result && result.text) || '').trim();
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

app.listen(PORT, () => {
  console.log(`Transcribe server listening on :${PORT}`);
  console.log('Each user supplies their own OpenAI key from the browser.');
  if (PAIRING_CODE) console.log(`Phoning home with pairing code: ${PAIRING_CODE}`);
  registerSelf();
});
