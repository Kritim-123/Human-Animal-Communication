const API_BASE_URL = process.env.EXPO_PUBLIC_DOGBRIDGE_API_URL ?? 'http://127.0.0.1:8000';

export type Dog = {
  id: number;
  name: string;
  breed?: string | null;
  age?: number | null;
  notes?: string | null;
  created_at: string;
};

export type Clip = {
  id: number;
  dog_id: number;
  file_path: string;
  duration_seconds?: number | null;
  location_context: string;
  situation_context: string;
  owner_label: string;
  outcome_label?: string | null;
  prediction_label?: string | null;
  prediction_confidence?: number | null;
  confirmed_correct?: boolean | null;
  notes?: string | null;
  created_at: string;
};

export type DogStats = {
  dog_id: number;
  clip_count: number;
  label_distribution: Record<string, number>;
  prediction_accuracy?: number | null;
  confirmed_count: number;
};

async function jsonRequest<T>(path: string, options: RequestInit = {}): Promise<T> {
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
  return jsonRequest<Dog>('/dogs', {
    method: 'POST',
    body: JSON.stringify(input),
  });
}

export function listDogs() {
  return jsonRequest<Dog[]>('/dogs');
}

export function getStats(dogId: number) {
  return jsonRequest<DogStats>(`/stats/${dogId}`);
}

export async function uploadClip(input: {
  dogId: number;
  audioUri: string;
  locationContext: string;
  situationContext: string;
  ownerLabel: string;
  outcomeLabel?: string;
  notes?: string;
}) {
  const formData = new FormData();
  formData.append('dog_id', String(input.dogId));
  formData.append('location_context', input.locationContext);
  formData.append('situation_context', input.situationContext);
  formData.append('owner_label', input.ownerLabel);
  if (input.outcomeLabel) {
    formData.append('outcome_label', input.outcomeLabel);
  }
  if (input.notes) {
    formData.append('notes', input.notes);
  }
  formData.append('file', {
    uri: input.audioUri,
    name: `dogbridge-${Date.now()}.m4a`,
    type: 'audio/m4a',
  } as unknown as Blob);

  const response = await fetch(`${API_BASE_URL}/clips`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Upload failed with ${response.status}`);
  }

  return response.json() as Promise<Clip>;
}

export function predictClip(clipId: number) {
  return jsonRequest<{ predicted_label: string; confidence: number; message?: string }>(`/predict/${clipId}`, {
    method: 'POST',
  });
}

export function confirmPrediction(clipId: number, input: { confirmed_correct: boolean; corrected_label?: string; outcome_label?: string }) {
  return jsonRequest(`/confirm/${clipId}`, {
    method: 'POST',
    body: JSON.stringify(input),
  });
}
