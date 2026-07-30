// api/chat.js
// Vercel Serverless API — Multi-Provider Free-Tier Gateway with a hard time budget.
//
// WHY THIS VERSION IS DIFFERENT (fixing the recurring "permanent" errors):
//
// The previous version tried up to 19 model attempts across 5 providers with
// NO per-attempt timeout and NO global time budget. If an early provider (e.g.
// Gemini, which has had free-tier quota issues on this account) hung or was
// slow, the whole chain could exceed Vercel's function limit and get killed
// with no useful error ever reaching the browser — which is the actual
// "permanent error" you were chasing. Two structural fixes:
//
// 1. GLOBAL_DEADLINE_MS: the handler tracks total elapsed time and stops
//    trying further providers once it's close to Vercel's timeout, returning
//    a clean JSON error instead of letting Vercel kill the function outright.
// 2. PER_ATTEMPT_TIMEOUT_MS: every single provider call is wrapped in an
//    AbortController so one slow provider can't consume the whole budget.
//
// Providers are also reordered by real-world speed/reliability for THIS
// account: Groq first (fastest — 300-800 tokens/sec on dedicated inference
// hardware, generous free tier, no card), then Cerebras (also fast and
// generous), then OpenRouter, then SambaNova, then Gemini last (this
// account hit a "0 free-tier quota" wall on Gemini earlier, so it's kept
// only as a last-resort fallback, not tried first where it just wastes time).
//
// IMPORTANT: vercel.json must include:
//   "functions": { "api/chat.js": { "maxDuration": 30 } }
// Without this, Vercel's Hobby plan defaults to a 10-second timeout, which
// this multi-provider chain can exceed even with the fixes above.

const DEFAULT_SYSTEM = `You are TalentTrack AI, an elite enterprise Talent Acquisition, Sourcing & Immigration Intelligence assistant.
Every question is asked in a recruitment/hiring context, even if phrased ambiguously — for example, "boolean search" or "boolean string" ALWAYS means a candidate-sourcing search query, NEVER a programming language boolean data type, unless the user explicitly asks about writing code.
Always structure your responses cleanly, professionally, and comprehensively.
Never output raw markdown artifacts like '#---' or '***' on their own. Use clean headers, bold sub-topics, bullet points, and numbered lists.
Do NOT truncate answers midway — complete every generated output in full detail.
Never reply with just a bare classification label such as "User Safety: safe" or similar — always give a complete, direct, helpful answer.`;

// Ordered fastest/most-reliable-first. Trimmed to the models most likely to
// actually respond quickly rather than trying every variant a provider offers.
const GROQ_MODELS = ['llama-3.1-8b-instant', 'llama-3.3-70b-versatile'];
const CEREBRAS_MODELS = ['llama3.1-8b'];
const OPENROUTER_MODELS = [
  'meta-llama/llama-3.2-3b-instruct:free',
  'mistralai/mistral-7b-instruct:free',
  'meta-llama/llama-3.1-8b-instruct:free',
];
const SAMBANOVA_MODELS = ['Meta-Llama-3.1-8B-Instruct'];
const GEMINI_MODELS = ['gemini-2.0-flash', 'gemini-1.5-flash'];

const GLOBAL_DEADLINE_MS = 25000; // stay safely under Vercel's 30s maxDuration
const PER_ATTEMPT_TIMEOUT_MS = 7000;

function isSafetyLabelOnly(text) {
  return /^(user safety|safety)\s*:\s*(safe|unsafe)\.?$/i.test((text || '').trim());
}

async function safeFetchJson(endpoint, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(endpoint, { ...options, signal: controller.signal });
    const text = await res.text().catch(() => '');
    let data;
    try { data = JSON.parse(text); }
    catch (e) { data = { error: { message: text.slice(0, 300) || `HTTP ${res.status} non-JSON response` } }; }
    return { ok: res.ok, status: res.status, data };
  } catch (err) {
    const msg = err.name === 'AbortError' ? `Timed out after ${timeoutMs}ms` : err.message;
    return { ok: false, status: 0, data: { error: { message: msg } } };
  } finally {
    clearTimeout(timer);
  }
}

async function callGemini(modelName, apiKey, system, prompt, timeoutMs) {
  const combinedSystem = system ? `${system}\n\n${DEFAULT_SYSTEM}` : DEFAULT_SYSTEM;
  return await safeFetchJson(
    `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${apiKey}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        system_instruction: { parts: [{ text: combinedSystem }] },
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.3, maxOutputTokens: 2048 },
      }),
    },
    timeoutMs
  );
}

async function callOpenAIStyle(endpoint, modelName, apiKey, system, prompt, timeoutMs, extraHeaders = {}) {
  const combinedSystem = system ? `${system}\n\n${DEFAULT_SYSTEM}` : DEFAULT_SYSTEM;
  return await safeFetchJson(
    endpoint,
    {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json', ...extraHeaders },
      body: JSON.stringify({
        model: modelName,
        messages: [
          { role: 'system', content: combinedSystem },
          { role: 'user', content: prompt },
        ],
        temperature: 0.3,
        max_tokens: 2048,
      }),
    },
    timeoutMs
  );
}

module.exports = async function handler(req, res) {
  const startTime = Date.now();
  const timeLeft = () => GLOBAL_DEADLINE_MS - (Date.now() - startTime);

  try {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') { res.status(200).end(); return; }

    // Trim every key defensively — a trailing newline or space from copy-paste
    // (very easy to do by accident from a browser or terminal) makes a key
    // invalid in a way that looks identical to "wrong key" but is much
    // harder to notice just by looking at it.
    const groqKey = (process.env.GROQ_API_KEY || '').trim() || undefined;
    const cerebrasKey = (process.env.CEREBRAS_API_KEY || '').trim() || undefined;
    const openrouterKey = (process.env.OPENROUTER_API_KEY || '').trim() || undefined;
    const sambanovaKey = (process.env.SAMBANOVA_API_KEY || '').trim() || undefined;
    const geminiKey = (process.env.GEMINI_API_KEY || '').trim() || undefined;

    const activeKeys = {
      GROQ_API_KEY: !!groqKey,
      CEREBRAS_API_KEY: !!cerebrasKey,
      OPENROUTER_API_KEY: !!openrouterKey,
      SAMBANOVA_API_KEY: !!sambanovaKey,
      GEMINI_API_KEY: !!geminiKey,
    };

    // GET /api/chat — visit this URL directly in your browser any time to
    // instantly check which keys Vercel currently sees, with zero AI calls
    // made and zero quota spent. This is the fastest way to confirm whether
    // a redeploy actually picked up new environment variables.
    if (req.method === 'GET') {
      const anyKey = Object.values(activeKeys).some(Boolean);
      res.status(200).json({
        status: 'diagnostic',
        message: anyKey
          ? 'At least one API key is detected by this deployment. If AI requests still fail, the key itself may be invalid/expired — see per-provider errors in a real POST request.'
          : 'NO API keys are detected by this deployment. Either none are set in Vercel, or they were added/changed after the last deploy and you have not redeployed since. Add a key in Vercel > Settings > Environment Variables, then go to Deployments > (latest) > "..." > Redeploy.',
        activeKeys,
      });
      return;
    }

    if (req.method !== 'POST') { res.status(405).json({ error: 'Method not allowed. Use POST for AI requests, or GET for a diagnostic check.' }); return; }

    let body = req.body || {};
    if (typeof body === 'string') { try { body = JSON.parse(body); } catch (e) { body = {}; } }

    const { system, prompt } = body;
    if (!prompt) { res.status(400).json({ error: 'Missing "prompt" in request body.' }); return; }

    if (!groqKey && !cerebrasKey && !openrouterKey && !sambanovaKey && !geminiKey) {
      res.status(400).json({
        error: 'No API keys detected in Vercel Environment Variables. Set at least GROQ_API_KEY (recommended — fastest free tier) in Vercel, then redeploy. Visit this same URL with GET (just paste it in your browser) any time to check key status without spending quota.',
        activeKeys,
      });
      return;
    }

    let errors = [];

    // Each tier is a function so we can loop over them and bail early once
    // the global time budget runs low, instead of blindly working through
    // every remaining provider and risking a hard Vercel timeout.
    const tiers = [
      {
        name: 'groq',
        active: !!groqKey,
        run: async (timeoutMs) => {
          for (const model of GROQ_MODELS) {
            if (timeLeft() < 1500) return null;
            const { ok, data } = await callOpenAIStyle('https://api.groq.com/openai/v1/chat/completions', model, groqKey, system, prompt, Math.min(timeoutMs, Math.max(timeLeft() - 500, 1000)));
            if (ok) {
              const text = data?.choices?.[0]?.message?.content?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `groq/${model}` };
            }
            errors.push(`Groq (${model}): ${data?.error?.message || 'Failed'}`);
          }
          return null;
        },
      },
      {
        name: 'cerebras',
        active: !!cerebrasKey,
        run: async (timeoutMs) => {
          for (const model of CEREBRAS_MODELS) {
            if (timeLeft() < 1500) return null;
            const { ok, data } = await callOpenAIStyle('https://api.cerebras.ai/v1/chat/completions', model, cerebrasKey, system, prompt, Math.min(timeoutMs, Math.max(timeLeft() - 500, 1000)));
            if (ok) {
              const text = data?.choices?.[0]?.message?.content?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `cerebras/${model}` };
            }
            errors.push(`Cerebras (${model}): ${data?.error?.message || 'Failed'}`);
          }
          return null;
        },
      },
      {
        name: 'openrouter',
        active: !!openrouterKey,
        run: async (timeoutMs) => {
          for (const model of OPENROUTER_MODELS) {
            if (timeLeft() < 1500) return null;
            const { ok, data } = await callOpenAIStyle('https://openrouter.ai/api/v1/chat/completions', model, openrouterKey, system, prompt, Math.min(timeoutMs, Math.max(timeLeft() - 500, 1000)), { 'HTTP-Referer': 'https://vercel.com', 'X-Title': 'TalentTrack Smart ATS' });
            if (ok) {
              const text = data?.choices?.[0]?.message?.content?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `openrouter/${model}` };
            }
            errors.push(`OpenRouter (${model}): ${data?.error?.message || 'Failed'}`);
          }
          return null;
        },
      },
      {
        name: 'sambanova',
        active: !!sambanovaKey,
        run: async (timeoutMs) => {
          for (const model of SAMBANOVA_MODELS) {
            if (timeLeft() < 1500) return null;
            const { ok, data } = await callOpenAIStyle('https://api.sambanova.ai/v1/chat/completions', model, sambanovaKey, system, prompt, Math.min(timeoutMs, Math.max(timeLeft() - 500, 1000)));
            if (ok) {
              const text = data?.choices?.[0]?.message?.content?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `sambanova/${model}` };
            }
            errors.push(`SambaNova (${model}): ${data?.error?.message || 'Failed'}`);
          }
          return null;
        },
      },
      {
        name: 'gemini',
        active: !!geminiKey,
        run: async (timeoutMs) => {
          for (const model of GEMINI_MODELS) {
            if (timeLeft() < 1500) return null;
            const { ok, data } = await callGemini(model, geminiKey, system, prompt, Math.min(timeoutMs, Math.max(timeLeft() - 500, 1000)));
            if (ok) {
              const text = data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `gemini/${model}` };
            }
            errors.push(`Gemini (${model}): ${data?.error?.message || 'Status ' + data?.status || 'failed'}`);
          }
          return null;
        },
      },
    ];

    for (const tier of tiers) {
      if (!tier.active) { errors.push(`${tier.name}: no API key configured.`); continue; }
      if (timeLeft() < 1500) { errors.push(`${tier.name}: skipped — out of time budget.`); continue; }
      const result = await tier.run(PER_ATTEMPT_TIMEOUT_MS);
      if (result) {
        res.status(200).json({ ...result, activeKeys, elapsedMs: Date.now() - startTime });
        return;
      }
    }

    res.status(502).json({
      error: 'All configured AI providers failed, timed out, or ran out of time budget. ' +
        'Open this exact URL in your browser (GET request, no quota used) to see which keys are actually detected right now: ' +
        '/api/chat — if it shows no keys detected, you added them in Vercel but have not redeployed since. ' +
        'If keys ARE detected but this still fails, check the "details" list below for the specific reason each provider rejected the request (commonly: invalid/expired key, or free-tier daily limit reached).',
      activeKeys,
      elapsedMs: Date.now() - startTime,
      details: errors,
    });

  } catch (globalErr) {
    res.status(502).json({ error: 'Serverless execution error occurred.', details: globalErr.message });
  }
};
