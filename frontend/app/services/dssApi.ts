const API_BASE = "http://localhost:5000/api/dss"; // Change to your backend URL

export async function getMlOutputs(train_id: string) {
  const res = await fetch(`${API_BASE}/ml_outputs/${train_id}`);
  if (!res.ok) throw new Error("Failed to fetch ML outputs");
  return await res.json();
}

export async function getLlmRecommendations(train_id: string) {
  const res = await fetch(`${API_BASE}/llm_recommendations/${train_id}`);
  if (!res.ok) throw new Error("Failed to fetch LLM recommendations");
  return await res.json();
}