// api/chat.js
// Vercel Serverless API - Multi-Key & 8-Provider Ultra-Resilient Engine (Zero Quota Crash Guarantee)

const DEFAULT_SYSTEM = `You are TalentTrack AI, an elite enterprise Talent Acquisition, Sourcing, Legal & Immigration Intelligence assistant.
Every question is asked in a professional business context, even if phrased ambiguously — for example, "boolean search" or "boolean string" ALWAYS means a candidate-sourcing search query, NEVER a programming language boolean data type, unless the user explicitly asks about writing code.
Always structure your responses cleanly, professionally, and comprehensively.
Never output raw markdown artifacts like '#---' or '***' on their own. Use clean headers, bold sub-topics, and bullet points using "-" as the only bullet character — never "*", "+", "•", or other symbols.
Do NOT truncate answers midway — complete every generated output in full detail.
Never reply with just a bare classification label such as "User Safety: safe" or similar — always give a complete, direct, helpful answer.

CRITICAL — ZERO INFORMATION LOSS RULE: whenever you are asked to reformat, restructure, standardize, or improve the layout of existing content (such as a resume/CV, a document, or any user-supplied text), you must preserve EVERY piece of information from the original — every bullet point, sentence, and detail. Only change formatting and structure, never the substance or the count of items, unless the user explicitly asks you to shorten or summarize. Producing fewer bullets/items than the original contained is a critical failure.

JOB DESCRIPTION CONSISTENCY RULE: whenever you generate a Job Description, in any module or context, always use exactly this structure — Job Title / Location / About the Role / Key Responsibilities / Key Skills & Qualifications / Preferred Qualifications — with "-" as the only bullet character. Keep this structure consistent every single time, and always incorporate every specific requirement, mandatory skill, or detail the user provides — never generalize it away or ignore it.`;

// High-capacity free tier models
const GROQ_MODELS = ['llama-3.1-8b-instant', 'gemma2-9b-it', 'mixtral-8x7b-32768', 'llama-3.3-70b-versatile'];
const CEREBRAS_MODELS = ['llama3.1-8b', 'llama3.3-70b'];
const GEMINI_MODELS = ['gemini-1.5-flash', 'gemini-1.5-flash-8b', 'gemini-2.0-flash', 'gemini-1.5-pro'];
const MISTRAL_MODELS = ['mistral-small-latest', 'open-mistral-7b'];
const CLOUDFLARE_MODELS = ['@cf/meta/llama-3.1-8b-instruct'];
const SAMBANOVA_MODELS = ['Meta-Llama-3.1-8B-Instruct', 'Meta-Llama-3.3-70B-Instruct'];
const OPENROUTER_MODELS = [
  'google/gemini-2.0-flash-lite-preview-02-05:free',
  'google/gemini-2.0-flash-exp:free',
  'meta-llama/llama-3.3-70b-instruct:free',
  'meta-llama/llama-3.1-8b-instruct:free',
  'qwen/qwen-2.5-72b-instruct:free',
  'mistralai/mistral-7b-instruct:free',
  'openchat/openchat-7b:free'
];

const GLOBAL_DEADLINE_MS = 25000;
const PER_ATTEMPT_TIMEOUT_MS = 6000;
const DEFAULT_MAX_TOKENS = 2048;
const MAX_TOKENS_CEILING = 8000;

function clampMaxTokens(requested) {
  const n = parseInt(requested, 10);
  if (!n || n < 256) return DEFAULT_MAX_TOKENS;
  return Math.min(n, MAX_TOKENS_CEILING);
}

// Split comma-separated keys for instant multi-key rotation
function getKeys(envVal) {
  if (!envVal) return [];
  return envVal.split(',').map(k => k.trim()).filter(Boolean);
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

module.exports = async function handler(req, res) {
  const startTime = Date.now();
  const timeLeft = () => GLOBAL_DEADLINE_MS - (Date.now() - startTime);

  try {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') { res.status(200).end(); return; }

    const groqKeys = getKeys(process.env.GROQ_API_KEY);
    const cerebrasKeys = getKeys(process.env.CEREBRAS_API_KEY);
    const geminiKeys = getKeys(process.env.GEMINI_API_KEY);
    const mistralKeys = getKeys(process.env.MISTRAL_API_KEY);
    const cloudflareAccountId = (process.env.CLOUDFLARE_ACCOUNT_ID || '').trim();
    const cloudflareApiToken = (process.env.CLOUDFLARE_API_TOKEN || '').trim();
    const sambanovaKeys = getKeys(process.env.SAMBANOVA_API_KEY);
    const openrouterKeys = getKeys(process.env.OPENROUTER_API_KEY);

    const activeKeys = {
      GROQ_API_KEY: groqKeys.length,
      CEREBRAS_API_KEY: cerebrasKeys.length,
      GEMINI_API_KEY: geminiKeys.length,
      MISTRAL_API_KEY: mistralKeys.length,
      CLOUDFLARE: !!(cloudflareAccountId && cloudflareApiToken),
      SAMBANOVA_API_KEY: sambanovaKeys.length,
      OPENROUTER_API_KEY: openrouterKeys.length,
    };

    if (req.method === 'GET') {
      const anyKey = Object.values(activeKeys).some(Boolean);
      res.status(200).json({
        status: 'diagnostic',
        message: anyKey ? 'API keys detected and active.' : 'No API keys detected in Environment Variables.',
        activeKeys,
      });
      return;
    }

    if (req.method !== 'POST') { res.status(405).json({ error: 'Method not allowed. Use POST.' }); return; }

    let body = req.body || {};
    if (typeof body === 'string') { try { body = JSON.parse(body); } catch (e) { body = {}; } }

    const { system, prompt, maxTokens: requestedMaxTokens } = body;
    if (!prompt) { res.status(400).json({ error: 'Missing "prompt" in request body.' }); return; }
    const maxTokens = clampMaxTokens(requestedMaxTokens);

    let errors = [];

    // Provider Tiers with Automatic Key Rotation
    const tiers = [
      {
        name: 'groq',
        keys: groqKeys,
        run: async () => {
          for (const apiKey of groqKeys) {
            for (const model of GROQ_MODELS) {
              if (timeLeft() < 1200) return null;
              const { ok, data } = await callOpenAIStyle('https://api.groq.com/openai/v1/chat/completions', model, apiKey, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens);
              if (ok) {
                const text = data?.choices?.[0]?.message?.content?.trim();
                if (text) return { text, modelUsed: `groq/${model}` };
              }
              errors.push(`Groq (${model}): ${data?.error?.message || 'Failed'}`);
            }
          }
          return null;
        }
      },
      {
        name: 'cerebras',
        keys: cerebrasKeys,
        run: async () => {
          for (const apiKey of cerebrasKeys) {
            for (const model of CEREBRAS_MODELS) {
              if (timeLeft() < 1200) return null;
              const { ok, data } = await callOpenAIStyle('https://api.cerebras.ai/v1/chat/completions', model, apiKey, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens);
              if (ok) {
                const text = data?.choices?.[0]?.message?.content?.trim();
                if (text) return { text, modelUsed: `cerebras/${model}` };
              }
              errors.push(`Cerebras (${model}): ${data?.error?.message || 'Failed'}`);
            }
          }
          return null;
        }
      },
      {
        name: 'gemini',
        keys: geminiKeys,
        run: async () => {
          for (const apiKey of geminiKeys) {
            for (const model of GEMINI_MODELS) {
              if (timeLeft() < 1200) return null;
              const { ok, data } = await callGemini(model, apiKey, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens);
              if (ok) {
                const text = data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
                if (text) return { text, modelUsed: `gemini/${model}` };
              }
              errors.push(`Gemini (${model}): ${data?.error?.message || 'Failed'}`);
            }
          }
          return null;
        }
      },
      {
        name: 'mistral',
        keys: mistralKeys,
        run: async () => {
          for (const apiKey of mistralKeys) {
            for (const model of MISTRAL_MODELS) {
              if (timeLeft() < 1200) return null;
              const { ok, data } = await callOpenAIStyle('https://api.mistral.ai/v1/chat/completions', model, apiKey, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens);
              if (ok) {
                const text = data?.choices?.[0]?.message?.content?.trim();
                if (text) return { text, modelUsed: `mistral/${model}` };
              }
              errors.push(`Mistral (${model}): ${data?.error?.message || 'Failed'}`);
            }
          }
          return null;
        }
      },
      {
        name: 'cloudflare',
        keys: (cloudflareAccountId && cloudflareApiToken) ? ['cf'] : [],
        run: async () => {
          for (const model of CLOUDFLARE_MODELS) {
            if (timeLeft() < 1200) return null;
            const endpoint = `https://api.cloudflare.com/client/v4/accounts/${cloudflareAccountId}/ai/v1/chat/completions`;
            const { ok, data } = await callOpenAIStyle(endpoint, model, cloudflareApiToken, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens);
            if (ok) {
              const text = data?.choices?.[0]?.message?.content?.trim();
              if (text) return { text, modelUsed: `cloudflare/${model}` };
            }
            errors.push(`Cloudflare (${model}): ${data?.error?.message || 'Failed'}`);
          }
          return null;
        }
      },
      {
        name: 'sambanova',
        keys: sambanovaKeys,
        run: async () => {
          for (const apiKey of sambanovaKeys) {
            for (const model of SAMBANOVA_MODELS) {
              if (timeLeft() < 1200) return null;
              const { ok, data } = await callOpenAIStyle('https://api.sambanova.ai/v1/chat/completions', model, apiKey, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens);
              if (ok) {
                const text = data?.choices?.[0]?.message?.content?.trim();
                if (text) return { text, modelUsed: `sambanova/${model}` };
              }
              errors.push(`SambaNova (${model}): ${data?.error?.message || 'Failed'}`);
            }
          }
          return null;
        }
      },
      {
        name: 'openrouter',
        keys: openrouterKeys,
        run: async () => {
          for (const apiKey of openrouterKeys) {
            for (const model of OPENROUTER_MODELS) {
              if (timeLeft() < 1200) return null;
              const { ok, data } = await callOpenAIStyle('https://openrouter.ai/api/v1/chat/completions', model, apiKey, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens, { 'HTTP-Referer': 'https://vercel.com', 'X-Title': 'TalentTrack Smart ATS' });
              if (ok) {
                const text = data?.choices?.[0]?.message?.content?.trim();
                if (text) return { text, modelUsed: `openrouter/${model}` };
              }
              errors.push(`OpenRouter (${model}): ${data?.error?.message || 'Failed'}`);
            }
          }
          return null;
        }
      }
    ];

    for (const tier of tiers) {
      if (tier.keys.length === 0) { errors.push(`${tier.name}: no API key configured.`); continue; }
      if (timeLeft() < 1200) { errors.push(`${tier.name}: skipped — out of time budget.`); continue; }
      const result = await tier.run();
      if (result) {
        res.status(200).json({ ...result, activeKeys, elapsedMs: Date.now() - startTime });
        return;
      }
    }

    res.status(502).json({
      error: 'All AI providers failed or timed out. Please check activeKeys or try again in a moment.',
      activeKeys,
      details: errors
    });

  } catch (globalErr) {
    res.status(502).json({ error: 'Serverless execution error occurred.', details: globalErr.message });
  }
};
