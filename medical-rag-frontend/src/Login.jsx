import { useState } from "react";

const BASE = "http://127.0.0.1:8000";

export default function Login({ onAuth }) {
  const [mode, setMode] = useState("signup");
  
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    if (!username.trim() || !password) {
      setError("Enter a username and password.");
      return;
    }
    setLoading(true);
    setError("");

    try {
      const res = await fetch(`${BASE}/${mode === "login" ? "login" : "signup"}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: username.trim(), password }),
      });
      const data = await res.json();

      if (!res.ok) {
        setError(data.detail || "Something went wrong.");
        setLoading(false);
        return;
      }

      onAuth({ token: data.access_token, username: data.username });
    } catch {
      setError("Could not reach the server.");
    }
    setLoading(false);
  };

  return (
    <div style={s.page}>
      <div style={s.card}>
        <p style={s.heading}>Medical RAG Assistant</p>
        <p style={s.sub}>{mode === "login" ? "Log in to continue" : "Create an account"}</p>

        <input
          style={s.input}
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
        />
        <input
          style={s.input}
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
        />

        {error && <div style={s.error}>{error}</div>}

        <button style={s.btn(loading)} onClick={submit} disabled={loading}>
          {loading ? "Please wait…" : mode === "login" ? "Log in" : "Sign up"}
        </button>

        <div style={s.switchRow}>
          {mode === "login" ? (
            <>
              Don't have an account?{" "}
              <span style={s.link} onClick={() => { setMode("signup"); setError(""); }}>
                Sign up
              </span>
            </>
          ) : (
            <>
              Already have an account?{" "}
              <span style={s.link} onClick={() => { setMode("login"); setError(""); }}>
                Log in
              </span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

const s = {
  page: {
    minHeight: "100vh",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#111",
    fontFamily: "'Inter', 'Segoe UI', sans-serif",
  },
  card: {
    width: 340,
    background: "#1a1a1a",
    border: "1px solid #2a2a2a",
    borderRadius: 12,
    padding: "32px 28px",
  },
  heading: { fontSize: 20, fontWeight: 600, color: "#fff", margin: "0 0 4px" },
  sub: { fontSize: 14, color: "#aaa", margin: "0 0 20px" },
  input: {
    width: "100%",
    padding: "10px 14px",
    fontSize: 15,
    border: "1px solid #2a2a2a",
    borderRadius: 8,
    outline: "none",
    color: "#fff",
    background: "#0f0f0f",
    marginBottom: 12,
    boxSizing: "border-box",
  },
  btn: (disabled) => ({
    width: "100%",
    padding: "10px 0",
    fontSize: 15,
    fontWeight: 500,
    border: "1px solid #333",
    borderRadius: 8,
    background: disabled ? "#1a1a1a" : "#fff",
    color: disabled ? "#555" : "#111",
    cursor: disabled ? "not-allowed" : "pointer",
    marginTop: 4,
  }),
  error: {
    border: "1px solid #4a1a1a",
    background: "#2a1010",
    borderRadius: 8,
    padding: "8px 12px",
    color: "#f88",
    fontSize: 13,
    marginBottom: 12,
  },
  switchRow: { fontSize: 13, color: "#999", marginTop: 16, textAlign: "center" },
  link: { color: "#fff", textDecoration: "underline", cursor: "pointer" },
};
