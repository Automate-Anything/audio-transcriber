// ============================================================================
// MCP server — exposes the transcription pipeline as Model Context Protocol
// tools so Claude / Cursor / any MCP client can drive the app.
//
// Design:
// - Stateless Streamable HTTP: a fresh McpServer + transport per request. The
//   jobId is the state carrier (held by the caller), so no MCP session store
//   is needed — which also means it survives our own restarts cleanly.
// - BYOK passthrough: provider keys arrive as HTTP headers from the connector
//   config (X-OpenAI-Key / X-AssemblyAI-Key), NEVER as tool arguments (so the
//   model never has to handle secrets, and they don't show up in transcripts).
// - Tools call the in-process service functions directly (no self-HTTP).
// ============================================================================
const { McpServer } = require('@modelcontextprotocol/sdk/server/mcp.js');
const { StreamableHTTPServerTransport } = require('@modelcontextprotocol/sdk/server/streamableHttp.js');
const { z } = require('zod');

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// Build a per-request MCP server whose tools close over this caller's keys.
function buildServer(keys, deps, ctx) {
  const { createTranscriptionJob, fetchAudioToTemp, consumeUpload, createUploadSlot, jobs, serializeJob, runPostProcess } = deps;
  const baseUrl = (ctx && ctx.baseUrl) || '';
  const server = new McpServer({ name: 'transcribe', version: '1.0.0' });

  const asText = (obj) => ({ content: [{ type: 'text', text: typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2) }] });
  const asError = (msg) => ({ content: [{ type: 'text', text: 'Error: ' + msg }], isError: true });

  // ---- create_upload -------------------------------------------------------
  server.registerTool('create_upload', {
    title: 'Create an upload slot for a local file',
    description:
      'Use this when the audio/video is a LOCAL file with no public URL. Returns an uploadUrl ' +
      'and uploadId. Upload the file bytes with an HTTP PUT to uploadUrl (e.g. in a code sandbox: ' +
      '`curl -T file.mp3 <uploadUrl>`), then call start_transcription with that uploadId instead ' +
      'of audioUrl. If the file is already at a public URL, skip this and pass audioUrl directly.',
    inputSchema: {
      fileName: z.string().optional().describe('Original file name (used only for extension hints).'),
    },
  }, async (args) => {
    if (!baseUrl) return asError('Server base URL unavailable; cannot create an upload slot.');
    const uploadId = createUploadSlot(args.fileName || 'audio');
    return asText({
      uploadId,
      uploadUrl: `${baseUrl}/api/uploads/${uploadId}`,
      method: 'PUT',
      next: 'PUT the raw file bytes to uploadUrl, then call start_transcription with this uploadId.',
    });
  });

  // ---- start_transcription ------------------------------------------------
  server.registerTool('start_transcription', {
    title: 'Start a transcription',
    description:
      'Transcribe an audio or video file. Provide either audioUrl (a public https URL) OR an ' +
      'uploadId from create_upload (for local files). Returns a jobId immediately; call ' +
      'get_transcription with that jobId to retrieve the result (transcription runs asynchronously ' +
      'and can take from seconds to several minutes depending on length). Use mode "multi_speaker" ' +
      'for conversations/meetings where you want speaker labels (diarization); use "single_speaker" ' +
      'for one voice (dictation, a single narrator).',
    inputSchema: {
      audioUrl: z.string().url().optional().describe('Public https URL to the audio or video file.'),
      uploadId: z.string().optional().describe('An uploadId from create_upload (use instead of audioUrl for local files).'),
      mode: z.enum(['single_speaker', 'multi_speaker']).default('single_speaker')
        .describe('single_speaker = OpenAI Whisper; multi_speaker = AssemblyAI with speaker labels.'),
      language: z.string().optional().describe('ISO code like "en" or "he". Omit to auto-detect.'),
      keyterms: z.string().optional().describe('Comma-separated names/jargon to bias recognition toward.'),
      multipleLanguages: z.boolean().optional().describe('Set true if the audio mixes languages (code-switching). Best with multi_speaker.'),
      languages: z.string().optional().describe('Comma-separated language names present, when multipleLanguages is true (hint).'),
    },
  }, async (args) => {
    const multi = args.mode === 'multi_speaker';
    const apiKey = multi ? keys.assembly : keys.openai;
    if (!apiKey) {
      return asError(
        (multi
          ? 'No AssemblyAI key configured. Add X-AssemblyAI-Key to this connector\u2019s headers'
          : 'No OpenAI key configured. Add X-OpenAI-Key to this connector\u2019s headers') +
        ' to use ' + args.mode + '.'
      );
    }
    if (!args.audioUrl && !args.uploadId) {
      return asError('Provide either audioUrl (a public URL) or uploadId (from create_upload).');
    }
    let fileInfo;
    try {
      fileInfo = args.uploadId ? consumeUpload(args.uploadId) : await fetchAudioToTemp(args.audioUrl);
    } catch (e) {
      return asError('Could not obtain the audio: ' + (e.message || e));
    }
    try {
      const jobId = createTranscriptionJob(fileInfo, apiKey, {
        provider: multi ? 'assemblyai' : 'openai',
        language: (args.language || '').toLowerCase(),
        keyterms: args.keyterms || '',
        codeSwitching: !!args.multipleLanguages,
        csLanguages: args.languages || '',
      });
      return asText({
        jobId,
        status: 'processing',
        next: 'Call get_transcription with this jobId to retrieve the transcript.',
      });
    } catch (e) {
      return asError(e.message || 'Could not start transcription.');
    }
  });

  // ---- get_transcription ---------------------------------------------------
  server.registerTool('get_transcription', {
    title: 'Get a transcription result',
    description:
      'Fetch the status and (when ready) the transcript for a jobId from start_transcription. ' +
      'This waits briefly for completion; if it returns status "processing", call it again to keep polling.',
    inputSchema: {
      jobId: z.string().describe('The jobId returned by start_transcription.'),
    },
  }, async (args) => {
    const deadline = Date.now() + 24000; // bounded wait to cut round-trips
    while (true) {
      const job = jobs.get(args.jobId);
      if (!job) return asError('No job with that ID (it may have expired or the server restarted).');
      if (job.status === 'done' || job.status === 'error') {
        const s = serializeJob(job);
        if (s.status === 'error') return asError(s.error || 'Transcription failed.');
        // Prefer speaker-labeled text when available.
        let body = s.transcript || '';
        if (Array.isArray(s.utterances) && s.utterances.length) {
          body = s.utterances.map(u => `${u.speaker}: ${u.text}`).join('\n');
        }
        return asText({
          status: 'done',
          provider: s.provider,
          durationSeconds: s.duration,
          transcript: body,
          utterances: s.utterances || undefined,
        });
      }
      if (Date.now() > deadline) {
        const s = serializeJob(job);
        return asText({
          status: 'processing',
          progress: Math.round((s.progress || 0) * 100) / 100,
          message: s.message,
          next: 'Still working — call get_transcription again with the same jobId.',
        });
      }
      await sleep(3000);
    }
  });

  // ---- post_process --------------------------------------------------------
  server.registerTool('post_process', {
    title: 'Summarize / extract action items / translate a transcript',
    description:
      'Run AI post-processing on a transcript: a summary, a consistent list of action items, ' +
      'and/or a translation. Requires an OpenAI key (X-OpenAI-Key header).',
    inputSchema: {
      transcript: z.string().describe('The transcript text to process.'),
      tasks: z.array(z.enum(['summary', 'actions', 'translate'])).min(1)
        .describe('Which outputs to produce.'),
      targetLanguage: z.string().optional().describe('Target language when "translate" is requested (e.g. "English", "Hebrew").'),
      model: z.enum(['gpt-4o-mini', 'gpt-4o', 'gpt-4.1']).optional().describe('Model to use (default gpt-4o-mini).'),
    },
  }, async (args) => {
    if (!keys.openai) return asError('No OpenAI key configured. Add X-OpenAI-Key to this connector\u2019s headers.');
    try {
      const result = await runPostProcess(keys.openai, {
        transcript: args.transcript,
        tasks: args.tasks,
        targetLanguage: args.targetLanguage,
        model: args.model,
      });
      return asText(result);
    } catch (e) {
      return asError(e.message || 'Post-processing failed.');
    }
  });

  return server;
}

// Mount POST /mcp on the given Express app. `express` is passed in so we can
// apply a JSON body parser scoped to this route only.
function mountMcp(app, express, deps) {
  app.post('/mcp', express.json({ limit: '4mb' }), async (req, res) => {
    if (deps.isShuttingDown && deps.isShuttingDown()) {
      return res.status(503).json({ jsonrpc: '2.0', error: { code: -32000, message: 'Server is updating — try again shortly.' }, id: null });
    }
    const keys = {
      openai: (req.headers['x-openai-key'] || '').toString().trim(),
      assembly: (req.headers['x-assemblyai-key'] || '').toString().trim(),
    };
    const baseUrl = deps.publicBaseUrl ? deps.publicBaseUrl(req) : '';
    const server = buildServer(keys, deps, { baseUrl });
    const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
    res.on('close', () => { try { transport.close(); } catch {} try { server.close(); } catch {} });
    try {
      await server.connect(transport);
      await transport.handleRequest(req, res, req.body);
    } catch (e) {
      console.error('[mcp] request error:', e);
      if (!res.headersSent) {
        res.status(500).json({ jsonrpc: '2.0', error: { code: -32603, message: 'Internal error' }, id: null });
      }
    }
  });
}

module.exports = { mountMcp };
