// api/chat.js
// Vercel Serverless API - Ultra-Resilient 5-Provider Gateway (Zero-500 Crash Guarantee)

const DEFAULT_SYSTEM = `You are TalentTrack AI, an elite enterprise Talent Acquisition, Sourcing & Immigration Intelligence assistant.
Always structure your responses cleanly, professionally, and comprehensively.
Never output raw markdown artifacts like '#---' or '***'. Use clean headers, bold sub-topics, bullet points, and numbered lists.
Do NOT truncate answers midway — complete every generated output in full detail.`;

// Tier 1: Gemini Direct (10 Lakhs Tokens/Min Free)
const GEMINI_MODELS = [
  'gemini-1.5-flash',
  'gemini-1.5-flash-8b',
  'gemini-2.0-flash',
  'gemini-1.5-pro'
];

// Tier 2: Cerebras Inference (1,000,000 Free Tokens/Day)
const CEREBRAS_MODELS = [
  'llama3.1-8b',
  'llama3.3-70b'
];

// Tier 3: Groq Direct (500,000 Free Tokens/Day)
const GROQ_MODELS = [
  'llama-3.1-8b-instant',
  'gemma2-9b-it',
  'mixtral-8x7b-32768',
  'llama-3.3-70b-versatile'
];

// Tier 4: SambaNova Systems
const SAMBANOVA_MODELS = [
  'Meta-Llama-3.3-70B-Instruct',
  'Meta-Llama-3.1-8B-Instruct'
];

// Tier 5: OpenRouter Active Free Models
const OPENROUTER_MODELS = [
  'google/gemini-2.0-flash-lite-preview-02-05:free',
  'google/gemini-2.0-flash-exp:free',
  'meta-llama/llama-3.3-70b-instruct:free',
  'meta-llama/llama-3.1-8b-instruct:free',
  'qwen/qwen-2.5-72b-instruct:free',
  'mistralai/mistral-7b-instruct:free',
  'openchat/openchat-7b:free'
];

// SAFE RESPONSE PARSER - Prevents HTTP 500 when upstream APIs return HTML errors (like 503 / 502)
async function safeFetchJson(endpoint, options) {
  try {
    const res = await fetch(endpoint, options);
    const text = await res.text().catch(() => '');
    let data;
    try {
      data = JSON.parse(text);
    } catch (e) {
      data = { error: { message: text.slice(0, 300) || `HTTP ${res.status} non-JSON response` } };
    }
    return { ok: res.ok, status: res.status, data };
  } catch (err) {
    return { ok: false, status: 500, data: { error: { message: err.message } } };
  }
}

async function callGemini(modelName, apiKey, system, prompt) {
  const combinedSystem = system ? `${system}\n\n${DEFAULT_SYSTEM}` : DEFAULT_SYSTEM;
  return await safeFetchJson(
    `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${apiKey}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        system_instruction: { parts: [{ text: combinedSystem }] },
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.3, maxOutputTokens: 4096 }
      })
    }
  );
}

async function callOpenAIStyle(endpoint, modelName, apiKey, system, prompt, extraHeaders = {}) {
  const combinedSystem = system ? `${system}\n\n${DEFAULT_SYSTEM}` : DEFAULT_SYSTEM;
  return await safeFetchJson(endpoint, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      ...extraHeaders
    },
    body: JSON.stringify({
      model: modelName,
      messages: [
        { role: 'system', content: combinedSystem },
        { role: 'user', content: prompt }
      ],
      temperature: 0.3,
      max_tokens: 4096
    })
  });
}

// NATIVE COMMONJS HANDLER WITH MASTER EXCEPTION PROTECTION
module.exports = async function handler(req, res) {
  try {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
      res.status(200).end();
      return;
    }

    if (req.method !== 'POST') {
      res.status(405).json({ error: 'Method not allowed. Use POST.' });
      return;
    }

    // Safely parse request body if stringified
    let body = req.body || {};
    if (typeof body === 'string') {
      try { body = JSON.parse(body); } catch (e) { body = {}; }
    }

    const { system, prompt } = body;
    if (!prompt) {
      res.status(400).json({ error: 'Missing "prompt" in request body.' });
      return;
    }

    const geminiKey = process.env.GEMINI_API_KEY;
    const cerebrasKey = process.env.CEREBRAS_API_KEY;
    const groqKey = process.env.GROQ_API_KEY;
    const sambanovaKey = process.env.SAMBANOVA_API_KEY;
    const openrouterKey = process.env.OPENROUTER_API_KEY;

    const activeKeys = {
      GEMINI_API_KEY: !!geminiKey,
      CEREBRAS_API_KEY: !!cerebrasKey,
      GROQ_API_KEY: !!groqKey,
      SAMBANOVA_API_KEY: !!sambanovaKey,
      OPENROUTER_API_KEY: !!openrouterKey
    };

    if (!geminiKey && !cerebrasKey && !groqKey && !sambanovaKey && !openrouterKey) {
      res.status(400).json({
        error: 'No API keys detected in Vercel Environment Variables. Please set GEMINI_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY in Vercel and redeploy.',
        activeKeys
      });
      return;
    }

    let errors = [];

    // 1. Gemini
    if (geminiKey) {
      for (const model of GEMINI_MODELS) {
        const { ok, data, status } = await callGemini(model, geminiKey, system, prompt);
        if (ok) {
          const text = data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
          if (text) {
            res.status(200).json({ text, modelUsed: `gemini/${model}`, activeKeys });
            return;
          }
        }
        errors.push(`Gemini (${model}): ${data?.error?.message || 'Status ' + status}`);
      }
    } else {
      errors.push('Gemini: GEMINI_API_KEY not configured in Vercel.');
    }

    // 2. Cerebras AI
    if (cerebrasKey) {
      for (const model of CEREBRAS_MODELS) {
        const { ok, data } = await callOpenAIStyle(
          'https://api.cerebras.ai/v1/chat/completions',
          model,
          cerebrasKey,
          system,
          prompt
        );
        if (ok) {
          const text = data?.choices?.[0]?.message?.content?.trim();
          if (text) {
            res.status(200).json({ text, modelUsed: `cerebras/${model}`, activeKeys });
            return;
          }
        }
        errors.push(`Cerebras (${model}): ${data?.error?.message || 'Failed'}`);
      }
    } else {
      errors.push('Cerebras: CEREBRAS_API_KEY not configured in Vercel.');
    }

    // 3. Groq
    if (groqKey) {
      for (const model of GROQ_MODELS) {
        const { ok, data } = await callOpenAIStyle(
          'https://api.groq.com/openai/v1/chat/completions',
          model,
          groqKey,
          system,
          prompt
        );
        if (ok) {
          const text = data?.choices?.[0]?.message?.content?.trim();
          if (text) {
            res.status(200).json({ text, modelUsed: `groq/${model}`, activeKeys });
            return;
          }
        }
        errors.push(`Groq (${model}): ${data?.error?.message || 'Failed'}`);
      }
    } else {
      errors.push('Groq: GROQ_API_KEY not configured in Vercel.');
    }

    // 4. SambaNova Systems
    if (sambanovaKey) {
      for (const model of SAMBANOVA_MODELS) {
        const { ok, data } = await callOpenAIStyle(
          'https://api.sambanova.ai/v1/chat/completions',
          model,
          sambanovaKey,
          system,
          prompt
        );
        if (ok) {
          const text = data?.choices?.[0]?.message?.content?.trim();
          if (text) {
            res.status(200).json({ text, modelUsed: `sambanova/${model}`, activeKeys });
            return;
          }
        }
        errors.push(`SambaNova (${model}): ${data?.error?.message || 'Failed'}`);
      }
    } else {
      errors.push('SambaNova: SAMBANOVA_API_KEY not configured in Vercel.');
    }

    // 5. OpenRouter
    if (openrouterKey) {
      for (const model of OPENROUTER_MODELS) {
        const { ok, data } = await callOpenAIStyle(
          'https://openrouter.ai/api/v1/chat/completions',
          model,
          openrouterKey,
          system,
          prompt,
          { 'HTTP-Referer': 'https://vercel.com', 'X-Title': 'TalentTrack Smart ATS' }
        );
        if (ok) {
          const text = data?.choices?.[0]?.message?.content?.trim();
          if (text) {
            res.status(200).json({ text, modelUsed: `openrouter/${model}`, activeKeys });
            return;
          }
        }
        errors.push(`OpenRouter (${model}): ${data?.error?.message || 'Failed'}`);
      }
    } else {
      errors.push('OpenRouter: OPENROUTER_API_KEY not configured in Vercel.');
    }

    res.status(502).json({
      error: 'All AI providers failed or exceeded quota. Please check activeKeys or try again in a moment.',
      activeKeys,
      details: errors
    });

  } catch (globalErr) {
    // Master Safety Net: Prevents raw uncaught 500 runtime crashes
    res.status(502).json({
      error: 'Serverless execution error occurred.',
      details: globalErr.message
    });
  }
};
