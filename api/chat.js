// api/chat.js
// Vercel Serverless API - Node 24 Native 8-Provider Multi-Key Engine (Zero-Crash Guarantee)

const DEFAULT_SYSTEM = `You are TalentTrack AI, an elite enterprise Talent Acquisition, Sourcing, Legal & Immigration Intelligence assistant.
Every question is asked in a professional business context, even if phrased ambiguously — for example, "boolean search" or "boolean string" ALWAYS means a candidate-sourcing search query, NEVER a programming language boolean data type, unless the user explicitly asks about writing code.
Always structure your responses cleanly, professionally, and comprehensively.
Never output raw markdown artifacts like '#---' or '***' on their own. Use clean headers, bold sub-topics, and bullet points using "-" as the only bullet character — never "*", "+", "•", or other symbols.
Do NOT truncate answers midway — complete every generated output in full detail.
Never reply with just a bare classification label such as "User Safety: safe" or similar — always give a complete, direct, helpful answer.

CRITICAL — ZERO INFORMATION LOSS RULE: whenever you are asked to reformat, restructure, standardize, or improve the layout of existing content (such as a resume/CV, a document, or any user-supplied text), you must preserve EVERY piece of information from the original — every bullet point, sentence, and detail. Only change formatting and structure, never the substance or the count of items, unless the user explicitly asks you to shorten or summarize. Producing fewer bullets/items than the original contained is a critical failure.

JOB DESCRIPTION CONSISTENCY RULE: whenever you generate a Job Description, in any module or context, always use exactly this structure — Job Title / Location / About the Role / Key Responsibilities / Key Skills & Qualifications / Preferred Qualifications — with "-" as the only bullet character. Keep this structure consistent every single time, and always incorporate every specific requirement, mandatory skill, or detail the user provides — never generalize it away or ignore it.

BOOLEAN SEARCH PARSER & COMPILER SPECIFICATION:
When asked to generate, parse, or optimize Boolean search strings, you MUST act as a production-ready Lexer, AST Parser, and Query Compiler:
1. LEXER & TOKENIZER RULES:
   - LPAREN / RPAREN: '(' and ')' for explicit precedence grouping.
   - AND / OR / NOT: Standard uppercase operators.
   - QUOTED_TERM: Exact phrases wrapped in double quotes "...".
   - IMPLICIT AND: If two terms/groups are placed adjacent without an operator (e.g., "Java" "Developer"), automatically insert an implicit AND operator.
2. OPERATOR PRECEDENCE ORDER (Highest to Lowest):
   1. Parentheses ()
   2. Quoted Phrases / Exact Terms
   3. NOT operator
   4. AND operator (including implicit AND)
   5. OR operator
3. MULTI-ENGINE QUERY COMPILATION DELIVERABLES:
   Compile the resulting AST into 4 distinct, executable target query dialects:
   a) LinkedIn Recruiter / Sales Navigator String (No fake operators like intitle: or location:; clean titles, skill groups, uppercase AND/OR).
   b) Google X-Ray LinkedIn Profile Query (site:linkedin.com/in/ + title phrases + skill groups + noise filters -intitle:jobs -inurl:dir).
   c) GitHub Technical Sourcing Query (site:github.com + developer profile signatures ("joined on" OR "contributions") + tech stack).
   d) Elasticsearch / SQL Query Filter (Elasticsearch bool query / PostgreSQL tsquery syntax).
4. SYNTAX AUTO-REPAIR: Auto-close unbalanced parentheses/quotes. Sanitize special characters. Always wrap every Boolean search string in triple backticks with 'boolean' syntax tag (\`\`\`boolean ... \`\`\`).`;

// High-capacity free tier models prioritized
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

function isSafetyLabelOnly(text) {
  return /^(user safety|safety)\s*:\s*(safe|unsafe)\.?$/i.test((text || '').trim());
}

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
    const cloudflareAccountId = (process.env.CLOUDFLARE_ACCOUNT_ID || '').trim() || undefined;
    const cloudflareApiToken = (process.env.CLOUDFLARE_API_TOKEN || '').trim() || undefined;
    const openrouterKeys = getKeys(process.env.OPENROUTER_API_KEY);
    const sambanovaKeys = getKeys(process.env.SAMBANOVA_API_KEY);

    const activeKeys = {
      GROQ: groqKeys.length > 0,
      CEREBRAS: cerebrasKeys.length > 0,
      GEMINI: geminiKeys.length > 0,
      MISTRAL: mistralKeys.length > 0,
      CLOUDFLARE: !!(cloudflareAccountId && cloudflareApiToken),
      SAMBANOVA: sambanovaKeys.length > 0,
      OPENROUTER: openrouterKeys.length > 0,
    };

    if (req.method === 'GET') {
      const anyKey = Object.values(activeKeys).some(Boolean);
      res.status(200).json({
        status: 'diagnostic',
        message: anyKey
          ? 'API key(s) detected. System active.'
          : 'NO API keys detected. Set GROQ_API_KEY, GEMINI_API_KEY, or CEREBRAS_API_KEY in Vercel Environment Variables and redeploy.',
        activeKeys,
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

    if (!Object.values(activeKeys).some(Boolean)) {
      res.status(400).json({
        error: 'No API keys detected in Vercel Environment Variables. Set at least GROQ_API_KEY or GEMINI_API_KEY in Vercel, then redeploy.',
        activeKeys,
      });
      return;
    }

    let errors = [];

    const providers = [
      {
        name: 'groq',
        keys: groqKeys,
        run: async (key) => {
          for (const model of GROQ_MODELS) {
            if (timeLeft() < 1500) return null;
            const { ok, data } = await callOpenAIStyle('https://api.groq.com/openai/v1/chat/completions', model, key, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens);
            if (ok) {
              const text = data?.choices?.[0]?.message?.content?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `groq/${model}` };
            }
            errors.push(`Groq (${model}): ${data?.error?.message || 'Failed'}`);
          }
          return null;
        }
      },
      {
        name: 'cerebras',
        keys: cerebrasKeys,
        run: async (key) => {
          for (const model of CEREBRAS_MODELS) {
            if (timeLeft() < 1500) return null;
            const { ok, data } = await callOpenAIStyle('https://api.cerebras.ai/v1/chat/completions', model, key, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens);
            if (ok) {
              const text = data?.choices?.[0]?.message?.content?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `cerebras/${model}` };
            }
            errors.push(`Cerebras (${model}): ${data?.error?.message || 'Failed'}`);
          }
          return null;
        }
      },
      {
        name: 'gemini',
        keys: geminiKeys,
        run: async (key) => {
          for (const model of GEMINI_MODELS) {
            if (timeLeft() < 1500) return null;
            const { ok, data } = await callGemini(model, key, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens);
            if (ok) {
              const text = data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `gemini/${model}` };
            }
            errors.push(`Gemini (${model}): ${data?.error?.message || 'Failed'}`);
          }
          return null;
        }
      },
      {
        name: 'mistral',
        keys: mistralKeys,
        run: async (key) => {
          for (const model of MISTRAL_MODELS) {
            if (timeLeft() < 1500) return null;
            const { ok, data } = await callOpenAIStyle('https://api.mistral.ai/v1/chat/completions', model, key, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens);
            if (ok) {
              const text = data?.choices?.[0]?.message?.content?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `mistral/${model}` };
            }
            errors.push(`Mistral (${model}): ${data?.error?.message || 'Failed'}`);
          }
          return null;
        }
      },
      {
        name: 'cloudflare',
        keys: (cloudflareAccountId && cloudflareApiToken) ? [cloudflareApiToken] : [],
        run: async (key) => {
          for (const model of CLOUDFLARE_MODELS) {
            if (timeLeft() < 1500) return null;
            const endpoint = `https://api.cloudflare.com/client/v4/accounts/${cloudflareAccountId}/ai/v1/chat/completions`;
            const { ok, data } = await callOpenAIStyle(endpoint, model, key, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens);
            if (ok) {
              const text = data?.choices?.[0]?.message?.content?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `cloudflare/${model}` };
            }
            errors.push(`Cloudflare (${model}): ${data?.error?.message || 'Failed'}`);
          }
          return null;
        }
      },
      {
        name: 'sambanova',
        keys: sambanovaKeys,
        run: async (key) => {
          for (const model of SAMBANOVA_MODELS) {
            if (timeLeft() < 1500) return null;
            const { ok, data } = await callOpenAIStyle('https://api.sambanova.ai/v1/chat/completions', model, key, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens);
            if (ok) {
              const text = data?.choices?.[0]?.message?.content?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `sambanova/${model}` };
            }
            errors.push(`SambaNova (${model}): ${data?.error?.message || 'Failed'}`);
          }
          return null;
        }
      },
      {
        name: 'openrouter',
        keys: openrouterKeys,
        run: async (key) => {
          for (const model of OPENROUTER_MODELS) {
            if (timeLeft() < 1500) return null;
            const { ok, data } = await callOpenAIStyle('https://openrouter.ai/api/v1/chat/completions', model, key, system, prompt, Math.min(PER_ATTEMPT_TIMEOUT_MS, Math.max(timeLeft() - 500, 1000)), maxTokens, { 'HTTP-Referer': 'https://vercel.com', 'X-Title': 'TalentTrack Smart ATS' });
            if (ok) {
              const text = data?.choices?.[0]?.message?.content?.trim();
              if (text && !isSafetyLabelOnly(text)) return { text, modelUsed: `openrouter/${model}` };
            }
            errors.push(`OpenRouter (${model}): ${data?.error?.message || 'Failed'}`);
          }
          return null;
        }
      }
    ];

    for (const provider of providers) {
      if (provider.keys.length === 0) {
        errors.push(`${provider.name}: No API key configured.`);
        continue;
      }
      for (const key of provider.keys) {
        if (timeLeft() < 1500) { errors.push(`${provider.name}: skipped due to execution time limit.`); break; }
        const result = await provider.run(key);
        if (result) {
          res.status(200).json({ ...result, activeKeys, elapsedMs: Date.now() - startTime });
          return;
        }
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
};
