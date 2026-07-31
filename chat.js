// api/chat.js
// Vercel Serverless API — Multi-Provider Free-Tier Gateway with a hard time budget.
//
// THIS ROUND'S CHANGES:
// 1. Added maxTokens support (request body can pass { maxTokens: N }), so
//    long tasks like full CV reformatting or contract generation can request
//    a bigger output budget than quick chat replies — this was the second
//    cause (alongside a bad prompt) of the resume formatter silently cutting
//    content short.
// 2. Strengthened DEFAULT_SYSTEM with two global rules that apply to every
//    request regardless of which module calls it: (a) never summarize or
//    omit content when asked to reformat/restructure — preserve everything,
//    and (b) when generating a Job Description in ANY context, always use
//    the same standardized structure and "-" bullets only, for consistency
//    between the dedicated JD Generator and ad-hoc requests via chat.
// 3. Added three more verified, genuinely free, no-credit-card providers:
//    Mistral AI, Cohere, and Cloudflare Workers AI — on top of the existing
//    Groq, Cerebras, OpenRouter, SambaNova, and Gemini. Each was checked for
//    current (2026) free-tier terms before being added, rather than assumed.
// 4. Kept the per-attempt timeout + global time budget + trimmed keys from
//    the previous round — those fixes remain necessary and are unchanged.
//
// IMPORTANT: vercel.json must include:
//   "functions": { "api/chat.js": { "maxDuration": 30 } }

const DEFAULT_SYSTEM = `You are TalentTrack AI, an elite enterprise Talent Acquisition, Sourcing, Legal & Immigration Intelligence assistant.
Every question is asked in a professional business context, even if phrased ambiguously — for example, "boolean search" or "boolean string" ALWAYS means a candidate-sourcing search query, NEVER a programming language boolean data type, unless the user explicitly asks about writing code.
Always structure your responses cleanly, professionally, and comprehensively.
Never output raw markdown artifacts like '#---' or '***' on their own. Use clean headers, bold sub-topics, and bullet points using "-" as the only bullet character — never "*", "+", "•", or other symbols.
Do NOT truncate answers midway — complete every generated output in full detail.
Never reply with just a bare classification label such as "User Safety: safe" or similar — always give a complete, direct, helpful answer.

CRITICAL — ZERO INFORMATION LOSS RULE: whenever you are asked to reformat, restructure, standardize, or improve the layout of existing content (such as a resume/CV, a document, or any user-supplied text), you must preserve EVERY piece of information from the original — every bullet point, sentence, and detail. Only change formatting and structure, never the substance or the count of items, unless the user explicitly asks you to shorten or summarize. Producing fewer bullets/items than the original contained is a critical failure.

JOB DESCRIPTION CONSISTENCY RULE: whenever you generate a Job Description, in any module or context, always use exactly this structure — Job Title / Location / About the Role / Key Responsibilities / Key Skills & Qualifications / Preferred Qualifications — with "-" as the only bullet character. Keep this structure consistent every single time, and always incorporate every specific requirement, mandatory skill, or detail the user provides — never generalize it away or ignore it.`;

// Ordered fastest/most-reliable-first based on verified 2026 free-tier terms.
const GROQ_MODELS = ['llama-3.1-8b-instant', 'llama-3.3-70b-versatile'];
const MISTRAL_MODELS = ['mistral-small-latest'];
const CEREBRAS_MODELS = ['llama3.1-8b'];
const CLOUDFLARE_MODELS = ['@cf/meta/llama-3.1-8b-instruct'];
const OPENROUTER_MODELS = [
  'meta-llama/llama-3.2-3b-instruct:free',
  'mistralai/mistral-7b-instruct:free',
  'meta-llama/llama-3.1-8b-instruct:free',
];
const COHERE_MODELS = ['command-r'];
const SAMBANOVA_MODELS = ['Meta-Llama-3.1-8B-Instruct'];
const GEMINI_MODELS = ['gemini-2.0-flash', 'gemini-1.5-flash'];

const GLOBAL_DEADLINE_MS = 25000; // stay safely under Vercel's 30s maxDuration
const PER_ATTEMPT_TIMEOUT_MS = 7000;
const DEFAULT_MAX_TOKENS = 2048;
const MAX_TOKENS_CEILING = 8000; // safety cap regardless of what the client requests

function clampMaxTokens(requested) {
  const n = parseInt(requested, 10);
  if (!n || n < 256) return DEFAULT_MAX_TOKENS;
  return Math.min(n, MAX_TOKENS_CEILING);
}

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

// Cohere uses its own v2 chat API shape (not OpenAI-compatible), so it needs
// its own call function and its own response-shape parsing at the call site.
async function callCohere(modelName, apiKey, system, prompt, timeoutMs, maxTokens) {
  const combinedSystem = system ? `${system}\n\n${DEFAULT_SYSTEM}` : DEFAULT_SYSTEM;
  return await safeFetchJson(
    'https://api.cohere.com/v2/chat',
    {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
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

    // Trim every key defensively — a trailing newline or space from copy-paste
    // makes a key invalid in a way that's hard to notice just by looking at it.
    const groqKey = (process.env.GROQ_API_KEY || '').trim() || undefined;
    const mistralKey = (process.env.MISTRAL_API_KEY || '').trim() || undefined;
    const cerebrasKey = (process.env.CEREBRAS_API_KEY || '').trim() || undefined;
    const cloudflareAccountId = (process.env.CLOUDFLARE_ACCOUNT_ID || '').trim() || undefined;
    const cloudflareApiToken = (process.env.CLOUDFLARE_API_TOKEN || '').trim() || undefined;
    const openrouterKey = (process.env.OPENROUTER_API_KEY || '').trim() || undefined;
    const cohereKey = (process.env.COHERE_API_KEY || '').trim() || undefined;
    const sambanovaKey = (process.env.SAMBANOVA_API_KEY || '').trim() || undefined;
    const geminiKey = (process.env.GEMINI_API_KEY || '').trim() || undefined;

    const activeKeys = {
      GROQ_API_KEY: !!groqKey,
      MISTRAL_API_KEY: !!mistralKey,
      CEREBRAS_API_KEY: !!cerebrasKey,
      CLOUDFLARE: !!(cloudflareAccountId && cloudflareApiToken),
      OPENROUTER_API_KEY: !!openrouterKey,
      COHERE_API_KEY: !!cohereKey,
      SAMBANOVA_API_KEY: !!sambanovaKey,
      GEMINI_API_KEY: !!geminiKey,
    };

    // GET /api/chat — visit this URL directly in your browser any time to
    // instantly check which keys Vercel currently sees, with zero AI calls
    // made and zero quota spent.
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

    const { system, prompt, maxTokens: requestedMaxTokens } = body;
    if (!prompt) { res.status(400).json({ error: 'Missing "prompt" in request body.' }); return; }
    const maxTokens = clampMaxTokens(requestedMaxTokens);

    if (!groqKey && !mistralKey && !cerebrasKey && !(cloudflareAccountId && cloudflareApiToken) && !openrouterKey && !cohereKey && !sambanovaKey && !geminiKey) {
      res.status(400).json({
        error: 'No API keys detected in Vercel Environment Variables. Set at least GROQ_API_KEY (recommended — fastest free tier) in Vercel, then redeploy. Visit this same URL with GET (just paste it in your browser) any time to check key status without spending quota.',
        activeKeys,
      });
      return;
    }

    let errors = [];

    const tiers = [
      {
        name: 'groq',
        active: !!groqKey,
        run: async () => {
          for (const model of GROQ_MODELS) {
            if (timeLeft() < 1500) return null;
            const { ok, data } = await callOpenAIStyle('https://api.groq.com/openai/v1/chat/completions', model, groqKey, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens);
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
        name: 'mistral',
        active: !!mistralKey,
        run: async () => {
          for (const model of MISTRAL_MODELS) {
            if (timeLeft() < 1500) return null;
            const { ok, data } = await callOpenAIStyle('https://api.mistral.ai/v1/chat/completions', model, mistralKey, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens);
            if (ok) {
              const text = data?.choices?.[0]?.message?.content?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `mistral/${model}` };
            }
            errors.push(`Mistral (${model}): ${data?.error?.message || 'Failed'}`);
          }
          return null;
        },
      },
      {
        name: 'cerebras',
        active: !!cerebrasKey,
        run: async () => {
          for (const model of CEREBRAS_MODELS) {
            if (timeLeft() < 1500) return null;
            const { ok, data } = await callOpenAIStyle('https://api.cerebras.ai/v1/chat/completions', model, cerebrasKey, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens);
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
        name: 'cloudflare',
        active: !!(cloudflareAccountId && cloudflareApiToken),
        run: async () => {
          for (const model of CLOUDFLARE_MODELS) {
            if (timeLeft() < 1500) return null;
            const endpoint = `https://api.cloudflare.com/client/v4/accounts/${cloudflareAccountId}/ai/v1/chat/completions`;
            const { ok, data } = await callOpenAIStyle(endpoint, model, cloudflareApiToken, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens);
            if (ok) {
              const text = data?.choices?.[0]?.message?.content?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `cloudflare/${model}` };
            }
            errors.push(`Cloudflare (${model}): ${data?.error?.message || 'Failed'}`);
          }
          return null;
        },
      },
      {
        name: 'openrouter',
        active: !!openrouterKey,
        run: async () => {
          for (const model of OPENROUTER_MODELS) {
            if (timeLeft() < 1500) return null;
            const { ok, data } = await callOpenAIStyle('https://openrouter.ai/api/v1/chat/completions', model, openrouterKey, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens, { 'HTTP-Referer': 'https://vercel.com', 'X-Title': 'TalentTrack Smart ATS' });
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
        name: 'cohere',
        active: !!cohereKey,
        run: async () => {
          for (const model of COHERE_MODELS) {
            if (timeLeft() < 1500) return null;
            const { ok, data } = await callCohere(model, cohereKey, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens);
            if (ok) {
              const text = data?.message?.content?.[0]?.text?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `cohere/${model}` };
            }
            errors.push(`Cohere (${model}): ${data?.error?.message || data?.message || 'Failed'}`);
          }
          return null;
        },
      },
      {
        name: 'sambanova',
        active: !!sambanovaKey,
        run: async () => {
          for (const model of SAMBANOVA_MODELS) {
            if (timeLeft() < 1500) return null;
            const { ok, data } = await callOpenAIStyle('https://api.sambanova.ai/v1/chat/completions', model, sambanovaKey, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens);
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
        run: async () => {
          for (const model of GEMINI_MODELS) {
            if (timeLeft() < 1500) return null;
            const { ok, data } = await callGemini(model, geminiKey, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens);
            if (ok) {
              const text = data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `gemini/${model}` };
            }
            errors.push(`Gemini (${model}): ${data?.error?.message || 'failed'}`);
          }
          return null;
        },
      },
    ];

    for (const tier of tiers) {
      if (!tier.active) { errors.push(`${tier.name}: no API key configured.`); continue; }
      if (timeLeft() < 1500) { errors.push(`${tier.name}: skipped — out of time budget.`); continue; }
      const result = await tier.run();
      if (result) {
        res.status(200).json({ ...result, activeKeys, elapsedMs: Date.now() - startTime, maxTokensUsed: maxTokens });
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
