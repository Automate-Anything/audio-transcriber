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

// ============================================================================
// App
// ============================================================================

const app = express();

// CORS — needed so any frontend (including this app served from another origin)
// can talk to this server when used as a "bring your own server".
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.setHeader('Access-Control-Max-Age', '86400');
  if (req.method === 'OPTIONS') return res.status(204).end();
  next();
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

app.post('/api/pair/register', (req, res) => {
  const code = String((req.body && req.body.code) || '').trim().toLowerCase();
  const url  = String((req.body && req.body.url)  || '').trim();
  if (!code || !url) return res.status(400).json({ error: 'code and url required' });
  if (!/^https?:\/\//.test(url)) return res.status(400).json({ error: 'url must include scheme' });
  if (code.length > 64 || url.length > 256) return res.status(400).json({ error: 'too long' });

  // Anti-clobber: if a *different* URL claimed the same code in the last minute, reject.
  const existing = pairings.get(code);
  if (existing && existing.url !== url && (Date.now() - existing.timestamp) < 60_000) {
    return res.status(409).json({ error: 'code already claimed' });
  }
  pairings.set(code, { url, timestamp: Date.now() });
  console.log(`[pair] registered ${code} -> ${url}`);
  res.json({ ok: true });
});

app.get('/api/pair/lookup/:code', (req, res) => {
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

app.get('/health', (req, res) => res.json({
  ok: true,
  ffmpeg: true, // assumed (Docker installs it); the build would have failed otherwise
  pairings: pairings.size,
}));

app.post('/api/transcribe', (req, res, next) => {
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
    return res.status(401).json({ error: 'Missing OpenAI API key. Add yours in the app to continue.' });
  }
  if (!apiKey.startsWith('sk-') || apiKey.length < 20) {
    if (req.file) try { fs.unlinkSync(req.file.path); } catch {}
    return res.status(401).json({ error: "That key doesn't look right. OpenAI keys start with sk-" });
  }
  if (!req.file) return res.status(400).json({ error: 'No file uploaded.' });

  const jobId = crypto.randomBytes(12).toString('hex');
  const job = {
    id: jobId,
    status: 'processing',
    stage: 'analyzing',
    message: 'Analyzing audio',
    progress: 0,
    transcript: '',
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

app.get('/api/jobs/:id', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: 'Job not found.' });
  res.json({
    id: job.id,
    status: job.status,
    stage: job.stage,
    message: job.message,
    progress: job.progress,
    transcript: job.transcript,
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

    const ext = path.extname(job.originalName).toLowerCase();
    const isCompressed = COMPRESSED_EXTS.includes(ext);
    const isSmall      = job.fileSize <= DIRECT_TO_WHISPER_MAX_BYTES;
    const isShort      = job.duration <= MAX_CHUNK;

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
  return new Promise((resolve, reject) => {
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
      if (code !== 0) return reject(new Error(`ffprobe failed: ${err.slice(0, 500)}`));
      const dur = parseFloat(out.trim());
      if (!isFinite(dur) || dur <= 0) return reject(new Error('Could not determine audio duration.'));
      resolve(dur);
    });
    proc.on('error', e => reject(new Error(`ffprobe not available: ${e.message}`)));
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
      else reject(new Error(`ffmpeg compress failed (${code}):\n${stderrTail}`));
    });
    proc.on('error', e => reject(new Error(`ffmpeg not available: ${e.message}`)));
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
      else reject(new Error(`ffmpeg silence detect failed (${code}):\n${stderrTail}`));
    });
    proc.on('error', e => reject(new Error(`ffmpeg not available: ${e.message}`)));
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
      if (code !== 0) return reject(new Error(`ffmpeg split failed (${code}):\n${stderrTail}`));
      const chunks = [];
      let i = 0;
      while (true) {
        const p = path.join(workDir, `chunk_${String(i).padStart(3, '0')}${ext}`);
        if (fs.existsSync(p)) { chunks.push(p); i++; } else break;
      }
      if (chunks.length === 0) return reject(new Error('No chunks produced.'));
      resolve(chunks);
    });
    proc.on('error', e => reject(new Error(`ffmpeg not available: ${e.message}`)));
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
  if (!isFinite(s)) return '';
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
