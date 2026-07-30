// api/chat.js
// Vercel Serverless API - Ultra-Resilient Multi-Provider Engine (Gemini + Groq + OpenRouter)

const DEFAULT_SYSTEM = `You are TalentTrack AI, an elite enterprise Talent Acquisition, Sourcing & Immigration Intelligence assistant.
Always structure your responses cleanly, professionally, and comprehensively.
Never output raw markdown artifacts like '#---' or '***'. Use clean headers, bold sub-topics, bullet points, and numbered lists.
Do NOT truncate answers midway — complete every generated output in full detail.`;

// 100% Active Free Model Candidates
const GEMINI_MODELS = [
  'gemini-1.5-flash',
  'gemini-1.5-flash-8b',
  'gemini-2.0-flash',
  'gemini-1.5-pro'
];

const GROQ_MODELS = [
  'llama-3.1-8b-instant',
  'gemma2-9b-it',
  'mixtral-8x7b-32768',
  'llama-3.3-70b-versatile'
];

// Active OpenRouter Free Slugs
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

// NATIVE VERCEL COMMONJS HANDLER (Prevents ESM Warning)
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
  const groqKey = process.env.GROQ_API_KEY;
  const openrouterKey = process.env.OPENROUTER_API_KEY;

  if (!geminiKey && !groqKey && !openrouterKey) {
    res.status(500).json({
      error: 'No API keys configured. Set GEMINI_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY in Vercel Environment Variables.'
    });
    return;
  }

  let errors = [];

  // Provider 1: Google Gemini Direct
  if (geminiKey) {
    for (const model of GEMINI_MODELS) {
      try {
        const { ok, data, status } = await callGemini(model, geminiKey, system, prompt);
        if (ok) {
          const text = data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
          if (text) {
            res.status(200).json({ text, modelUsed: `gemini/${model}` });
            return;
          }
        }
        errors.push(`Gemini (${model}): ${data?.error?.message || 'Status ' + status}`);
      } catch (err) {
        errors.push(`Gemini (${model}): ${err.message}`);
      }
    }
  } else {
    errors.push('Gemini: GEMINI_API_KEY not configured in Vercel Environment Variables.');
  }

  // Provider 2: Groq Direct
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
            res.status(200).json({ text, modelUsed: `groq/${model}` });
            return;
          }
        }
        errors.push(`Groq (${model}): ${data?.error?.message || 'Failed'}`);
      } catch (err) {
        errors.push(`Groq (${model}): ${err.message}`);
      }
    }
  } else {
    errors.push('Groq: GROQ_API_KEY not configured in Vercel Environment Variables.');
  }

  // Provider 3: OpenRouter Free Models
  if (openrouterKey) {
    for (const model of OPENROUTER_MODELS) {
      try {
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
            res.status(200).json({ text, modelUsed: `openrouter/${model}` });
            return;
          }
        }
        errors.push(`OpenRouter (${model}): ${data?.error?.message || 'Failed'}`);
      } catch (err) {
        errors.push(`OpenRouter (${model}): ${err.message}`);
      }
    }
  } else {
    errors.push('OpenRouter: OPENROUTER_API_KEY not configured in Vercel Environment Variables.');
  }

  res.status(502).json({
    error: 'All AI providers failed or exceeded quota. Please verify environment variables or try again in a moment.',
    details: errors
  });
};
