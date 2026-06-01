const API_BASE_URL = process.env.EXPO_PUBLIC_DOGBRIDGE_API_URL ?? 'http://127.0.0.1:8000';

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed with ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function createDog(input: { name: string; breed?: string; age?: number; notes?: string }) {
  return request('/dogs', {
    method: 'POST',
    body: JSON.stringify(input),
  });
}

export function predictClip(clipId: number) {
  return request<{ predicted_label: string; confidence: number; message?: string }>(`/predict/${clipId}`, {
    method: 'POST',
  });
}

export function confirmPrediction(clipId: number, input: { confirmed_correct: boolean; corrected_label?: string; outcome_label?: string }) {
  return request(`/confirm/${clipId}`, {
    method: 'POST',
    body: JSON.stringify(input),
  });
}

