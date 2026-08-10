// api/chat.js
// Vercel Serverless API — Ultra-Resilient Engine with Fast 3.5s Failover & Emergency Keyless Backup

const DEFAULT_SYSTEM = `You are TalentTrack AI, an elite enterprise Talent Acquisition, Sourcing, Legal & Immigration Intelligence assistant.
Every question is asked in a professional business context, even if phrased ambiguously — for example, "boolean search" or "boolean string" ALWAYS means a candidate-sourcing search query, NEVER a programming language boolean data type, unless the user explicitly asks about writing code.
Always structure your responses cleanly, professionally, and comprehensively.
Never output raw markdown artifacts like '#---' or '***' on their own. Use clean headers, bold sub-topics, and bullet points using "-" as the only bullet character — never "*", "+", "•", or other symbols.
Do NOT truncate answers midway — complete every generated output in full detail.
Never reply with just a bare classification label such as "User Safety: safe" or similar — always give a complete, direct, helpful answer.

CRITICAL — ZERO INFORMATION LOSS RULE: whenever you are asked to reformat, restructure, standardize, or improve the layout of existing content (such as a resume/CV, a document, or any user-supplied text), you must preserve EVERY piece of information from the original — every bullet point, sentence, and detail. Only change formatting and structure, never the substance or the count of items, unless the user explicitly asks you to shorten or summarize. Producing fewer bullets/items than the original contained is a critical failure.

JOB DESCRIPTION CONSISTENCY RULE: whenever you generate a Job Description, in any module or context, always use exactly this structure — Job Title / Location / About the Role / Key Responsibilities / Key Skills & Qualifications / Preferred Qualifications — with "-" as the only bullet character. Keep this structure consistent every single time, and always incorporate every specific requirement, mandatory skill, or detail the user provides — never generalize it away or ignore it.`;

// Active, currently-valid model slugs. Gemini in particular has retired
// model names multiple times in 2026 (gemini-1.5-flash and
// gemini-1.5-flash-8b were both confirmed retired via a real 404 from
// Google's API) — keeping only currently-live model names here, checked
// against the actual error this app received, not assumed.
const GROQ_MODELS = ['llama-3.1-8b-instant', 'llama-3.3-70b-versatile'];
const CEREBRAS_MODELS = ['llama3.1-8b', 'llama3.3-70b'];
const GEMINI_MODELS = ['gemini-2.0-flash', 'gemini-2.5-flash'];
const MISTRAL_MODELS = ['mistral-small-latest', 'open-mistral-7b'];
const CLOUDFLARE_MODELS = ['@cf/meta/llama-3.1-8b-instruct'];
const SAMBANOVA_MODELS = ['Meta-Llama-3.1-8B-Instruct', 'Meta-Llama-3.3-70B-Instruct'];
const NVIDIA_MODELS = ['meta/llama-3.1-8b-instruct', 'mistralai/mixtral-8x7b-instruct-v0.1'];
const OPENROUTER_MODELS = [
  'meta-llama/llama-3.3-70b-instruct:free',
  'meta-llama/llama-3.1-8b-instruct:free',
  'qwen/qwen-2.5-72b-instruct:free',
  'mistralai/mistral-7b-instruct:free',
];

const GLOBAL_DEADLINE_MS = 27500; // just under Vercel's 30s maxDuration, more headroom for large-output tasks
const BASE_ATTEMPT_TIMEOUT_MS = 3500;
const MAX_ATTEMPT_TIMEOUT_MS = 19000;

// A quick chat reply and a full 6000-token CV reformat are not the same kind
// of request — generating more tokens genuinely takes longer, and a flat
// 3.5s timeout guarantees every attempt at a large task fails before the
// model can even finish, regardless of whether the provider/key was fine.
// This scales the per-attempt allowance with how much output was requested,
// while still capping it so a single slow attempt can't eat the whole
// global time budget.
function computeAttemptTimeout(maxTokens) {
  const scaled = BASE_ATTEMPT_TIMEOUT_MS + maxTokens * 2.2;
  return Math.min(Math.max(scaled, BASE_ATTEMPT_TIMEOUT_MS), MAX_ATTEMPT_TIMEOUT_MS);
}
const DEFAULT_MAX_TOKENS = 2048;
const MAX_TOKENS_CEILING = 8000;

// In-Memory Key Cooldown Manager (5-minute blacklist for rate-limited keys)
const keyCooldowns = new Map();

function isKeyCoolingDown(key) {
  const expiry = keyCooldowns.get(key);
  if (!expiry) return false;
  if (Date.now() > expiry) {
    keyCooldowns.delete(key);
    return false;
  }
  return true;
}

function setKeyCooldown(key, cooldownMs = 300000) {
  keyCooldowns.set(key, Date.now() + cooldownMs);
}

function clampMaxTokens(requested) {
  const n = parseInt(requested, 10);
  if (!n || n < 256) return DEFAULT_MAX_TOKENS;
  return Math.min(n, MAX_TOKENS_CEILING);
}

function isSafetyLabelOnly(text) {
  return /^(user safety|safety)\s*:\s*(safe|unsafe)\.?$/i.test((text || '').trim());
}

function getKeys(envVal) {
  if (!envVal) return [];
  return envVal.split(',').map(k => k.trim()).filter(Boolean);
}

// With pools of 10-100+ keys per provider, trying every key sequentially in
// one request can burn the entire time budget on a single provider and never
// reach the others. Instead, each request randomly samples a small number of
// non-cooling-down keys per provider. Across many requests over time, this
// naturally spreads load evenly across the whole key pool (true rotation)
// without needing to track rotation state between requests.
//
// For LARGE requests (e.g. full CV reformatting), trying multiple keys of
// the SAME provider is a worse bet than trying MORE DIFFERENT providers once
// each — a provider with a small per-request token ceiling will reject a
// large request the same way regardless of which key sends it (confirmed:
// Groq's llama-3.1-8b-instant rejected identical-sized requests from 3
// different keys with the exact same "TPM limit 6000" error). So large
// requests sample fewer keys per provider but reach more providers instead.
const LARGE_TASK_TOKEN_THRESHOLD = 3000;
function maxKeysPerProvider(maxTokens) {
  return maxTokens > LARGE_TASK_TOKEN_THRESHOLD ? 1 : 3;
}

function sampleKeysToTry(keys, count) {
  const available = keys.filter(k => !isKeyCoolingDown(k));
  const pool = available.length > 0 ? available : keys; // if all cooling down, try anyway rather than skip the provider entirely
  const shuffled = [...pool];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled.slice(0, count);
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

async function callGemini(modelName, apiKey, system, prompt, timeoutMs, maxTokens) {
  const combinedSystem = system ? `${system}\n\n${DEFAULT_SYSTEM}` : DEFAULT_SYSTEM;
  return await safeFetchJson(
    `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${apiKey}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        system_instruction: { parts: [{ text: combinedSystem }] },
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.3, maxOutputTokens: maxTokens },
      }),
    },
    timeoutMs
  );
}

async function callOpenAIStyle(endpoint, modelName, apiKey, system, prompt, timeoutMs, maxTokens, extraHeaders = {}) {
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
        max_tokens: maxTokens,
      }),
    },
    timeoutMs
  );
}

// Emergency Keyless Fallback Engine (Pollinations AI Safety Net)
async function callEmergencyKeylessFallback(system, prompt, timeoutMs) {
  const combinedSystem = system ? `${system}\n\n${DEFAULT_SYSTEM}` : DEFAULT_SYSTEM;
  try {
    const { ok, data } = await safeFetchJson(
      'https://text.pollinations.ai/openai',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'openai',
          messages: [
            { role: 'system', content: combinedSystem },
            { role: 'user', content: prompt }
          ]
        })
      },
      timeoutMs
    );
    if (ok) {
      const text = data?.choices?.[0]?.message?.content?.trim();
      if (text && !isSafetyLabelOnly(text)) {
        return { text, modelUsed: 'emergency-keyless/openai' };
      }
    }
  } catch (e) {}
  return null;
}

export default async function handler(req, res) {
  const startTime = Date.now();
  const timeLeft = () => GLOBAL_DEADLINE_MS - (Date.now() - startTime);

  try {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') { res.status(200).end(); return; }

    const groqKeys = getKeys(process.env.GROQ_API_KEYS || process.env.GROQ_API_KEY);
    const cerebrasKeys = getKeys(process.env.CEREBRAS_API_KEYS || process.env.CEREBRAS_API_KEY);
    const geminiKeys = getKeys(process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY);
    const mistralKeys = getKeys(process.env.MISTRAL_API_KEYS || process.env.MISTRAL_API_KEY);
    const cloudflareAccountId = (process.env.CLOUDFLARE_ACCOUNT_ID || '').trim() || undefined;
    const cloudflareApiToken = (process.env.CLOUDFLARE_API_TOKEN || '').trim() || undefined;
    const openrouterKeys = getKeys(process.env.OPENROUTER_API_KEYS || process.env.OPENROUTER_API_KEY);
    const sambanovaKeys = getKeys(process.env.SAMBANOVA_API_KEYS || process.env.SAMBANOVA_API_KEY);
    const nvidiaKeys = getKeys(process.env.NVIDIA_API_KEYS || process.env.NVIDIA_API_KEY);

    const activeKeys = {
      GROQ: groqKeys.length,
      CEREBRAS: cerebrasKeys.length,
      GEMINI: geminiKeys.length,
      MISTRAL: mistralKeys.length,
      CLOUDFLARE: (cloudflareAccountId && cloudflareApiToken) ? 1 : 0,
      SAMBANOVA: sambanovaKeys.length,
      OPENROUTER: openrouterKeys.length,
      NVIDIA: nvidiaKeys.length,
    };

    if (req.method === 'GET') {
      const totalKeys = Object.values(activeKeys).reduce((a, b) => a + b, 0);
      res.status(200).json({
        status: 'diagnostic',
        message: totalKeys > 0
          ? `${totalKeys} total API key(s) detected across all providers. TalentTrack AI Gateway is active. Each request randomly samples a few keys per provider (fewer for large tasks like CV formatting, to reach more providers instead), so load spreads across your full pool over many requests.`
          : 'NO API keys detected. Set GROQ_API_KEYS or GEMINI_API_KEYS (comma-separated) in Vercel Environment Variables and redeploy.',
        keyPoolSizes: activeKeys,
      });
      return;
    }

    if (req.method !== 'POST') {
      res.status(405).json({ error: 'Method not allowed. Use POST.' });
      return;
    }

    let body = req.body || {};
    if (typeof body === 'string') { try { body = JSON.parse(body); } catch (e) { body = {}; } }

    const { system, prompt, maxTokens: requestedMaxTokens } = body;
    if (!prompt) { res.status(400).json({ error: 'Missing "prompt" in request body.' }); return; }
    const maxTokens = clampMaxTokens(requestedMaxTokens);
    let errors = [];
    if (maxTokens > LARGE_TASK_TOKEN_THRESHOLD && groqKeys.length > 0) {
      errors.push(`Groq: skipped for this request — its free-tier per-request token ceiling is too small for a ${maxTokens}-token task (confirmed via prior real error).`);
    }

    const providers = [
      {
        name: 'groq',
        // Groq's free-tier TPM ceiling on llama-3.1-8b-instant is 6000 tokens
        // per request (confirmed via a real error: "Requested 10073" was
        // rejected with "Limit 6000"). No key rotation fixes a per-request
        // size cap smaller than the request — so for large tasks, skip Groq
        // entirely and go straight to providers with bigger allowances.
        keys: maxTokens > LARGE_TASK_TOKEN_THRESHOLD ? [] : groqKeys,
        run: async (key) => {
          for (const model of GROQ_MODELS) {
            if (timeLeft() < 1200) return null;
            const { ok, status, data } = await callOpenAIStyle('https://api.groq.com/openai/v1/chat/completions', model, key, system, prompt, Math.min(computeAttemptTimeout(maxTokens), Math.max(timeLeft() - 500, 1000)), maxTokens);
            if (ok) {
              const text = data?.choices?.[0]?.message?.content?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `groq/${model}` };
            }
            errors.push(`Groq (${model}): ${data?.error?.message || 'Failed status ' + status}`);
            if (status === 429 || status === 403 || status === 401) { setKeyCooldown(key); return null; }
          }
          return null;
        }
      },
      {
        name: 'gemini',
        keys: geminiKeys,
        run: async (key) => {
          for (const model of GEMINI_MODELS) {
            if (timeLeft() < 1200) return null;
            const { ok, status, data } = await callGemini(model, key, system, prompt, Math.min(computeAttemptTimeout(maxTokens), Math.max(timeLeft() - 500, 1000)), maxTokens);
            if (ok) {
              const text = data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `gemini/${model}` };
            }
            errors.push(`Gemini (${model}): ${data?.error?.message || 'Failed status ' + status}`);
            if (status === 429 || status === 403 || status === 401) { setKeyCooldown(key); return null; }
          }
          return null;
        }
      },
      {
        name: 'cerebras',
        keys: cerebrasKeys,
        run: async (key) => {
          for (const model of CEREBRAS_MODELS) {
            if (timeLeft() < 1200) return null;
            const { ok, status, data } = await callOpenAIStyle('https://api.cerebras.ai/v1/chat/completions', model, key, system, prompt, Math.min(computeAttemptTimeout(maxTokens), Math.max(timeLeft() - 500, 1000)), maxTokens);
            if (ok) {
              const text = data?.choices?.[0]?.message?.content?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `cerebras/${model}` };
            }
            errors.push(`Cerebras (${model}): ${data?.error?.message || 'Failed status ' + status}`);
            if (status === 429 || status === 403 || status === 401) { setKeyCooldown(key); return null; }
          }
          return null;
        }
      },
      {
        name: 'mistral',
        keys: mistralKeys,
        run: async (key) => {
          for (const model of MISTRAL_MODELS) {
            if (timeLeft() < 1200) return null;
            const { ok, status, data } = await callOpenAIStyle('https://api.mistral.ai/v1/chat/completions', model, key, system, prompt, Math.min(computeAttemptTimeout(maxTokens), Math.max(timeLeft() - 500, 1000)), maxTokens);
            if (ok) {
              const text = data?.choices?.[0]?.message?.content?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `mistral/${model}` };
            }
            errors.push(`Mistral (${model}): ${data?.error?.message || 'Failed status ' + status}`);
            if (status === 429 || status === 403 || status === 401) { setKeyCooldown(key); return null; }
          }
          return null;
        }
      },
      {
        name: 'cloudflare',
        keys: (cloudflareAccountId && cloudflareApiToken) ? [cloudflareApiToken] : [],
        run: async (key) => {
          for (const model of CLOUDFLARE_MODELS) {
            if (timeLeft() < 1200) return null;
            const endpoint = `https://api.cloudflare.com/client/v4/accounts/${cloudflareAccountId}/ai/v1/chat/completions`;
            const { ok, status, data } = await callOpenAIStyle(endpoint, model, key, system, prompt, Math.min(computeAttemptTimeout(maxTokens), Math.max(timeLeft() - 500, 1000)), maxTokens);
            if (ok) {
              const text = data?.choices?.[0]?.message?.content?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `cloudflare/${model}` };
            }
            errors.push(`Cloudflare (${model}): ${data?.error?.message || 'Failed status ' + status}`);
            if (status === 429 || status === 403 || status === 401) { setKeyCooldown(key); return null; }
          }
          return null;
        }
      },
      {
        name: 'sambanova',
        keys: sambanovaKeys,
        run: async (key) => {
          for (const model of SAMBANOVA_MODELS) {
            if (timeLeft() < 1200) return null;
            const { ok, status, data } = await callOpenAIStyle('https://api.sambanova.ai/v1/chat/completions', model, key, system, prompt, Math.min(computeAttemptTimeout(maxTokens), Math.max(timeLeft() - 500, 1000)), maxTokens);
            if (ok) {
              const text = data?.choices?.[0]?.message?.content?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `sambanova/${model}` };
            }
            errors.push(`SambaNova (${model}): ${data?.error?.message || 'Failed status ' + status}`);
            if (status === 429 || status === 403 || status === 401) { setKeyCooldown(key); return null; }
          }
          return null;
        }
      },
      {
        name: 'openrouter',
        keys: openrouterKeys,
        run: async (key) => {
          for (const model of OPENROUTER_MODELS) {
            if (timeLeft() < 1200) return null;
            const { ok, status, data } = await callOpenAIStyle('https://openrouter.ai/api/v1/chat/completions', model, key, system, prompt, Math.min(computeAttemptTimeout(maxTokens), Math.max(timeLeft() - 500, 1000)), maxTokens, { 'HTTP-Referer': 'https://vercel.com', 'X-Title': 'TalentTrack Smart ATS' });
            if (ok) {
              const text = data?.choices?.[0]?.message?.content?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `openrouter/${model}` };
            }
            errors.push(`OpenRouter (${model}): ${data?.error?.message || 'Failed status ' + status}`);
            if (status === 429 || status === 403 || status === 401) { setKeyCooldown(key); return null; }
          }
          return null;
        }
      },
      {
        name: 'nvidia',
        keys: nvidiaKeys,
        run: async (key) => {
          for (const model of NVIDIA_MODELS) {
            if (timeLeft() < 1200) return null;
            const { ok, status, data } = await callOpenAIStyle('https://integrate.api.nvidia.com/v1/chat/completions', model, key, system, prompt, Math.min(computeAttemptTimeout(maxTokens), Math.max(timeLeft() - 500, 1000)), maxTokens);
            if (ok) {
              const text = data?.choices?.[0]?.message?.content?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `nvidia/${model}` };
            }
            errors.push(`NVIDIA (${model}): ${data?.error?.message || 'Failed status ' + status}`);
            if (status === 429 || status === 403 || status === 401) { setKeyCooldown(key); return null; }
          }
          return null;
        }
      }
    ];

    // For each provider, randomly sample a small, bounded number of keys
    // rather than linearly scanning the whole pool. With 100+ keys on some
    // providers, scanning them all in one request could burn the entire
    // time budget on a single provider and never reach the others. Random
    // sampling means every key gets tried roughly equally often across many
    // requests over time — that IS the rotation, without needing to track
    // state between requests. Large tasks sample fewer keys per provider so
    // more DIFFERENT providers get reached instead (see maxKeysPerProvider).
    const keysPerProvider = maxKeysPerProvider(maxTokens);
    for (const provider of providers) {
      if (provider.keys.length === 0) continue;
      if (timeLeft() < 1200) break;
      const keysToTry = sampleKeysToTry(provider.keys, keysPerProvider);
      for (const key of keysToTry) {
        if (timeLeft() < 1200) break;
        const result = await provider.run(key);
        if (result) {
          res.status(200).json({ ...result, activeKeys, elapsedMs: Date.now() - startTime, keyPoolSize: provider.keys.length });
          return;
        }
      }
    }

    // EMERGENCY SAFETY NET: Try keyless fallback if all user keys are exhausted/rate-limited
    if (timeLeft() > 2000) {
      const emergencyRes = await callEmergencyKeylessFallback(system, prompt, Math.min(5000, timeLeft() - 500));
      if (emergencyRes) {
        res.status(200).json({ ...emergencyRes, activeKeys, elapsedMs: Date.now() - startTime, emergencyFallbackUsed: true });
        return;
      }
    }

    res.status(502).json({
      error: 'All configured AI providers failed, timed out, or reached rate limits.',
      activeKeys,
      elapsedMs: Date.now() - startTime,
      details: errors,
    });

  } catch (globalErr) {
    res.status(502).json({ error: 'Serverless execution error occurred.', details: globalErr.message });
  }
}
