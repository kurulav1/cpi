const API_ROUTES = Object.freeze({
  health: "/api/health",
  stream: "/api/chat/stream"
});

export async function fetchHealth() {
  const response = await fetch(API_ROUTES.health);
  if (!response.ok) {
    throw new Error(`Health check failed (${response.status})`);
  }
  return response.json();
}

export async function streamChat({ messages, settings, signal, onEvent }) {
  const response = await fetch(API_ROUTES.stream, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      messages,
      profileId: settings.profileId,
      template: settings.template,
      systemPrompt: settings.systemPrompt,
      temperature: settings.temperature,
      maxNewTokens: settings.maxNewTokens,
      performanceMode: Boolean(settings.performanceMode),
      quantMode: settings.quantMode || "none"
    }),
    signal
  });

  if (!response.ok) {
    throw new Error(`Inference request failed (${response.status}).`);
  }

  if (!response.body) {
    throw new Error(`The API returned no stream body (${response.status}).`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalEvent = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });

    while (true) {
      const newlineIndex = buffer.indexOf("\n");
      if (newlineIndex === -1) {
        break;
      }

      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);

      if (!line) {
        continue;
      }

      const event = JSON.parse(line);
      onEvent?.(event);

      if (event.type === "error" || event.error) {
        throw new Error(event.error || "Inference failed.");
      }

      if (event.type === "done" || event.type === "aborted") {
        finalEvent = event;
      }
    }
  }

  const tail = `${buffer}${decoder.decode()}`.trim();
  if (tail) {
    const event = JSON.parse(tail);
    onEvent?.(event);
    if (event.type === "error" || event.error) {
      throw new Error(event.error || "Inference failed.");
    }
    if (event.type === "done" || event.type === "aborted") {
      finalEvent = event;
    }
  }

  return finalEvent;
}
