// api/chat.js
// Vercel Serverless API - 5-Provider Ultra-Resilient Engine (Gemini + Cerebras + Groq + SambaNova + OpenRouter)

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

// Tier 2: Cerebras Inference (1,000,000 Free Tokens/Day - Ultra-fast 2000+ tok/sec)
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

// Tier 4: SambaNova Systems (High-Capacity Free Tier)
const SAMBANOVA_MODELS = [
  'Meta-Llama-3.3-70B-Instruct',
  'Meta-Llama-3.1-8B-Instruct'
];

// Tier 5: OpenRouter Verified Active Free Models (NO paid deepseek-r1)
const OPENROUTER_MODELS = [
  'google/gemini-2.0-flash-lite-preview-02-05:free',
  'google/gemini-2.0-flash-exp:free',
  'meta-llama/llama-3.3-70b-instruct:free',
  'meta-llama/llama-3.1-8b-instruct:free',
  'qwen/qwen-2.5-72b-instruct:free',
  'mistralai/mistral-7b-instruct:free',
  'openchat/openchat-7b:free'
];

async function callGemini(modelName, apiKey, system, prompt) {
  const combinedSystem = system ? `${system}\n\n${DEFAULT_SYSTEM}` : DEFAULT_SYSTEM;
  const res = await fetch(
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
  const data = await res.json();
  return { ok: res.ok, status: res.status, data };
}

async function callOpenAIStyle(endpoint, modelName, apiKey, system, prompt, extraHeaders = {}) {
  const combinedSystem = system ? `${system}\n\n${DEFAULT_SYSTEM}` : DEFAULT_SYSTEM;
  const res = await fetch(endpoint, {
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
  const data = await res.json();
  return { ok: res.ok, status: res.status, data };
}

module.exports = async function handler(req, res) {
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

  const { system, prompt } = req.body || {};
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
    res.status(500).json({
      error: 'No API keys configured. Set GEMINI_API_KEY, CEREBRAS_API_KEY, GROQ_API_KEY, SAMBANOVA_API_KEY, or OPENROUTER_API_KEY in Vercel Environment Variables.',
      activeKeys
    });
    return;
  }

  let errors = [];

  // 1. Provider 1: Gemini Direct (10 Lakhs Tokens/Min Free)
  if (geminiKey) {
    for (const model of GEMINI_MODELS) {
      try {
        const { ok, data, status } = await callGemini(model, geminiKey, system, prompt);
        if (ok) {
          const text = data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
          if (text) {
            res.status(200).json({ text, modelUsed: `gemini/${model}`, activeKeys });
            return;
          }
        }
        errors.push(`Gemini (${model}): ${data?.error?.message || 'Status ' + status}`);
      } catch (err) {
        errors.push(`Gemini (${model}): ${err.message}`);
      }
    }
  } else {
    errors.push('Gemini: GEMINI_API_KEY not set in Vercel Environment Variables.');
  }

  // 2. Provider 2: Cerebras AI (1,000,000 Tokens/Day Free)
  if (cerebrasKey) {
    for (const model of CEREBRAS_MODELS) {
      try {
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
      } catch (err) {
        errors.push(`Cerebras (${model}): ${err.message}`);
      }
    }
  } else {
    errors.push('Cerebras: CEREBRAS_API_KEY not set in Vercel Environment Variables.');
  }

  // 3. Provider 3: Groq Direct (500,000 Tokens/Day Free)
  if (groqKey) {
    for (const model of GROQ_MODELS) {
      try {
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
      } catch (err) {
        errors.push(`Groq (${model}): ${err.message}`);
      }
    }
  } e
