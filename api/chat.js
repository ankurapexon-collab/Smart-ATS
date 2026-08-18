// api/chat.js
// Vercel Serverless API — Ultra-Resilient Engine with Auto-Failover & Multi-Provider Rotation

const DEFAULT_SYSTEM = `You are TalentTrack AI, an elite enterprise Talent Acquisition, Sourcing, Legal & Immigration Intelligence assistant.
Every question is asked in a professional business context.
Always structure your responses cleanly, professionally, and comprehensively.
Never output raw markdown artifacts on their own. Use clean headers, bold sub-topics, and bullet points using "-" as the only bullet character.
Do NOT truncate answers midway — complete every generated output in full detail.

CRITICAL — ZERO INFORMATION LOSS RULE: Whenever you are asked to reformat, restructure, or standardize existing content (such as a CV/resume), preserve EVERY piece of information from the original — every bullet point, sentence, date, degree, and technical detail. Only change formatting and structure, never omit content or reduce item counts.

EMPLOYMENT HISTORY FORMAT RULE:
For every position in Employment History, output:
Organisation: [Organisation Name], [Location], [Country] | [Duration/Tenure]
Role: [Designation]
Responsibilities
- [Bullet 1]
- [Bullet 2]
Never put a bullet point before Organisation: or Role:. "Organisation:" and "Role:" must be separate lines.

JOB DESCRIPTION CONSISTENCY RULE: Whenever you generate a Job Description, always use exactly this structure — Job Title / Location / About the Role / Key Responsibilities / Key Skills & Qualifications / Preferred Qualifications — with "-" as the bullet character.`;

// Active model slugs across all supported providers
const GROQ_MODELS = ['openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'llama-3.3-70b-specdec', 'llama3-70b-8192'];
const GEMINI_MODELS = ['gemini-2.5-flash', 'gemini-flash-latest', 'gemini-2.5-flash-lite', 'gemini-flash-lite-latest', 'gemini-pro-latest'];
const CEREBRAS_MODELS = ['gpt-oss-120b', 'llama-3.3-70b', 'llama3.1-8b'];
const SAMBANOVA_MODELS = ['Meta-Llama-3.3-70B-Instruct', 'DeepSeek-R1-Distill-Llama-70B'];
const OPENROUTER_MODELS = ['openrouter/free', 'openai/gpt-oss-20b:free', 'meta-llama/llama-3.3-70b-instruct'];
const MISTRAL_MODELS = ['mistral-small-latest', 'open-mistral-7b'];
const NVIDIA_MODELS = ['meta/llama-3.3-70b-instruct', 'z-ai/glm-4.7', 'nvidia/nemotron-4-340b-instruct', 'meta/llama-3.1-8b-instruct'];
const CLOUDFLARE_MODELS = ['@cf/meta/llama-3.1-8b-instruct'];

const GLOBAL_DEADLINE_MS = 28000;
const BASE_ATTEMPT_TIMEOUT_MS = 6500;
const MAX_ATTEMPT_TIMEOUT_MS = 19000;

function computeAttemptTimeout(maxTokens) {
  const scaled = BASE_ATTEMPT_TIMEOUT_MS + maxTokens * 2.2;
  return Math.min(Math.max(scaled, BASE_ATTEMPT_TIMEOUT_MS), MAX_ATTEMPT_TIMEOUT_MS);
}
const DEFAULT_MAX_TOKENS = 2048;
const MAX_TOKENS_CEILING = 4096;

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

function setKeyCooldown(key, cooldownMs = 120000) {
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

function sampleKeysToTry(keys, count) {
  const available = keys.filter(k => !isKeyCoolingDown(k));
  const pool = available.length > 0 ? available : keys;
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
          ? `${totalKeys} total API key(s) detected across all providers. TalentTrack AI Gateway is active.`
          : 'NO API keys detected. Set environment variables in Vercel.',
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

    const providers = [
      {
        name: 'groq',
        keys: groqKeys,
        run: async (key) => {
          for (const model of GROQ_MODELS) {
            if (timeLeft() < 1200) return null;
            const groqMaxTokens = Math.min(maxTokens, 4096);
            const { ok, status, data } = await callOpenAIStyle('https://api.groq.com/openai/v1/chat/completions', model, key, system, prompt, Math.min(computeAttemptTimeout(groqMaxTokens), Math.max(timeLeft() - 500, 1000)), groqMaxTokens);
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
      }
    ];

    const keysPerProvider = 5;
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

    if (timeLeft() > 2000) {
      const emergencyRes = await callEmergencyKeylessFallback(system, prompt, Math.min(6000, timeLeft() - 500));
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
