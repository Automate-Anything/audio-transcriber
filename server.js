// Audio transcription server.
// Accepts any audio/video file of any size, runs native ffmpeg locally to
// compress + smart-split at silence boundaries, then sends each chunk to
// OpenAI Whisper. Single external API.
//
// The user's OpenAI API key arrives per request via the `Authorization` header.
// It's never persisted on the server — held in the in-memory job state only for
// as long as the job takes, then wiped.

const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');
const FormData = require('form-data');

// ============================================================================
// Config
// ============================================================================

const PORT = process.env.PORT || 3000;
const MAX_FILE_SIZE = parseInt(process.env.MAX_FILE_SIZE || (5 * 1024 * 1024 * 1024), 10);

// Smart chunking (seconds)
const TARGET_CHUNK = 18 * 60;
const MIN_CHUNK   = 14 * 60;
const MAX_CHUNK   = 22 * 60;
const SILENCE_DB       = '-30';
const SILENCE_MIN_SEC  = '0.4';

// Audio output for transcription
const OUT_SAMPLE_RATE = 16000;
const OUT_BITRATE     = '24k';

// ============================================================================
// App + upload setup
// ============================================================================

const app = express();
const jobs = new Map();

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

app.use(express.json());
app.use(express.static('public'));

// ============================================================================
// Routes
// ============================================================================

app.get('/health', (req, res) => res.json({ ok: true }));

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
  // Read user's OpenAI API key from Authorization header (browser sends it per request).
  const auth = req.headers.authorization || '';
  const apiKey = auth.startsWith('Bearer ') ? auth.slice(7).trim() : '';

  if (!apiKey) {
    if (req.file) try { fs.unlinkSync(req.file.path); } catch {}
    return res.status(401).json({ error: 'Missing OpenAI API key. Add yours in the app to continue.' });
  }
  if (!apiKey.startsWith('sk-') || apiKey.length < 20) {
    if (req.file) try { fs.unlinkSync(req.file.path); } catch {}
    return res.status(401).json({ error: 'That key doesn\'t look right. OpenAI keys start with sk-' });
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
    createdAt: Date.now(),
    apiKey, // held in memory while this job runs; cleared when cleanupJob runs
  };
  jobs.set(jobId, job);

  // Fire and forget — client polls /api/jobs/:id
  processJob(job).catch(err => {
    console.error(`[${jobId}] error:`, err);
    job.status = 'error';
    job.stage = 'error';
    job.error = err.message || String(err);
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
  });
});

// ============================================================================
// Cleanup
// ============================================================================

setInterval(() => {
  const cutoff = Date.now() - 60 * 60 * 1000; // 1 hour
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
  // Wipe the key from memory once the job is done with it.
  job.apiKey = null;
}

// ============================================================================
// Pipeline
// ============================================================================

async function processJob(job) {
  job.workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tx-'));

  try {
    // 1. Probe duration
    job.stage = 'analyzing';
    job.message = 'Reading audio metadata';
    job.duration = await ffprobeDuration(job.inputPath);

    // 2. Compress to mono 16kHz 24kbps MP3 + scan for silences (one pass)
    job.stage = 'compressing';
    job.message = `Compressing ${formatTime(job.duration)} of audio`;
    const compressedPath = path.join(job.workDir, 'compressed.mp3');
    await compressAndDetectSilences(job, compressedPath);

    // 3. Decide on chunks
    let chunkPaths;
    if (job.duration <= MAX_CHUNK) {
      chunkPaths = [compressedPath];
      job.chunkInfo = { current: 0, total: 1, usedSilence: 0 };
    } else {
      job.stage = 'splitting';
      job.message = 'Splitting at silence boundaries';
      const splitPoints = pickSplitPoints(job.silences, job.duration);
      const silenceAligned = splitPoints.filter(
        t => job.silences.some(s => Math.abs(s.mid - t) < 0.01)
      ).length;
      chunkPaths = await splitAudio(compressedPath, splitPoints, job.workDir);
      job.chunkInfo = { current: 0, total: chunkPaths.length, usedSilence: silenceAligned };
    }

    // 4. Transcribe sequentially, appending text as we go
    job.stage = 'transcribing';
    let combined = '';
    for (let i = 0; i < chunkPaths.length; i++) {
      job.chunkInfo = { ...job.chunkInfo, current: i + 1 };
      job.progress = i / chunkPaths.length;
      job.message = `Transcribing chunk ${i + 1} of ${chunkPaths.length}`;

      const text = await transcribeFile(chunkPaths[i], job.apiKey);
      combined = combined ? `${combined} ${text}` : text;
      job.transcript = combined;
    }

    job.status = 'done';
    job.stage = 'done';
    job.progress = 1;
    job.message = 'Complete';
  } finally {
    // Cleanup temp files after a short delay (allows final poll to read transcript)
    setTimeout(() => cleanupJob(job), 60 * 1000);
  }
}

// ============================================================================
// FFmpeg helpers
// ============================================================================

function ffprobeDuration(filePath) {
  return new Promise((resolve, reject) => {
    const proc = spawn('ffprobe', [
      '-v', 'error',
      '-show_entries', 'format=duration',
      '-of', 'default=noprint_wrappers=1:nokey=1',
      filePath,
    ]);
    let out = '';
    let err = '';
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
    let pending = null;
    let stderrTail = '';

    proc.stderr.on('data', (data) => {
      const chunk = data.toString();
      stderrTail = (stderrTail + chunk).slice(-2000);

      for (const line of chunk.split('\n')) {
        const sM = line.match(/silence_start:\s*(-?[\d.]+)/);
        const eM = line.match(/silence_end:\s*([\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)/);
        if (sM) pending = { start: Math.max(0, parseFloat(sM[1])) };
        if (eM) {
          const end = parseFloat(eM[1]);
          const dur = parseFloat(eM[2]);
          if (pending) {
            pending.end = end; pending.duration = dur; pending.mid = (pending.start + end) / 2;
            job.silences.push(pending); pending = null;
          } else {
            const start = Math.max(0, end - dur);
            job.silences.push({ start, end, duration: dur, mid: (start + end) / 2 });
          }
        }
      }
    });

    proc.stdout.on('data', (data) => {
      const str = data.toString();
      const m = str.match(/out_time_ms=(\d+)/);
      if (m && job.duration) {
        // out_time_ms is actually in microseconds despite the name
        const elapsed = parseInt(m[1], 10) / 1_000_000;
        // Compression takes the bulk of pre-transcribe time; map to 0-0.5
        job.progress = Math.max(0, Math.min(0.5, (elapsed / job.duration) * 0.5));
      }
    });

    proc.on('close', code => {
      if (code === 0) {
        job.progress = 0.5;
        resolve();
      } else {
        reject(new Error(`ffmpeg compress failed (${code}):\n${stderrTail}`));
      }
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

function splitAudio(inputPath, splitPoints, workDir) {
  return new Promise((resolve, reject) => {
    const pattern = path.join(workDir, 'chunk_%03d.mp3');
    const args = [
      '-i', inputPath,
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
        const p = path.join(workDir, `chunk_${String(i).padStart(3, '0')}.mp3`);
        if (fs.existsSync(p)) { chunks.push(p); i++; }
        else break;
      }
      if (chunks.length === 0) return reject(new Error('No chunks produced.'));
      resolve(chunks);
    });
    proc.on('error', e => reject(new Error(`ffmpeg not available: ${e.message}`)));
  });
}

async function transcribeFile(filePath, apiKey) {
  const form = new FormData();
  form.append('file', fs.createReadStream(filePath), {
    filename: path.basename(filePath),
    contentType: 'audio/mpeg',
  });
  form.append('model', 'whisper-1');
  form.append('response_format', 'text');

  const response = await fetch('https://api.openai.com/v1/audio/transcriptions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      ...form.getHeaders(),
    },
    body: form,
  });

  if (!response.ok) {
    let detail = `OpenAI returned ${response.status}.`;
    try {
      const j = await response.json();
      if (j?.error?.message) detail = j.error.message;
    } catch {}
    throw new Error(detail);
  }

  return (await response.text()).trim();
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
// Start
// ============================================================================

app.listen(PORT, () => {
  console.log(`Transcribe server listening on :${PORT}`);
  console.log('Each user supplies their own OpenAI API key from the browser — no server-side key needed.');
});
