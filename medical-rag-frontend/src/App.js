import { useState, useRef } from "react";
import ReactMarkdown from "react-markdown";
import PatientInfoCard from "./PatientInfoCard";
import Login from "./Login";
import MyReports from "./MyReports";

const BASE = "http://127.0.0.1:8000";

export default function App() {
  // ── AUTH ──────────────────────────────────────────────────────────────────
  const [token, setToken] = useState();
  const [username, setUsername] = useState();

  const handleAuth = ({ token, username }) => {
    localStorage.setItem("token", token);
    localStorage.setItem("username", username);
    setToken(token);
    setUsername(username);
  };

  const logout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("username");
    setToken("");
    setUsername("");
  };

  // Wrapper so every protected call carries the auth header automatically,
  // and a 401 (expired/invalid token) bumps the user back to the login screen.
  const authFetch = async (url, options = {}) => {
    const res = await fetch(url, {
      ...options,
      headers: {
        ...(options.headers || {}),
        Authorization: `Bearer ${token}`,
      },
    });
    if (res.status === 401) {
      logout();
    }
    return res;
  };

  const [tab, setTab] = useState("chat");

  // chat state
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [chunks, setChunks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [answerSource, setAnswerSource] = useState(null); // "local" | "web_search" | null

  // report state
  const [file, setFile] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportStatus, setReportStatus] = useState("");
  const [reportResult, setReportResult] = useState(null);
  const [reportAnalysis, setReportAnalysis] = useState("");
  const [reportStreaming, setReportStreaming] = useState(false);
  const [error, setError] = useState("");
  const fileRef = useRef(null);

  // patient info extraction state
  const [extracting, setExtracting] = useState(false);
  const [patientInfo, setPatientInfo] = useState(null);     // extracted, not yet confirmed
  const [reportText, setReportText] = useState(null);       // OCR'd text, cached client-side
  const [extractionMethod, setExtractionMethod] = useState(null);
  const [confirmedPatient, setConfirmedPatient] = useState(null);

  // If not logged in, show the login/signup screen and nothing else.
  if (!token) {
    return <Login onAuth={handleAuth} />;
  }

  // ── CHAT ──────────────────────────────────────────────────────────────────
  const ask = async () => {
    if (!query.trim() || loading) return;
    setLoading(true);
    setStreaming(true);
    setAnswer("");
    setChunks([]);
    setAnswerSource(null);
    try {
      const res = await authFetch(`${BASE}/chat-stream?query=${encodeURIComponent(query)}`, { method: "POST" });
      if (!res.body) throw new Error("No stream");
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        for (const line of dec.decode(value).split("\n")) {
          if (!line.startsWith("data: ")) continue;
          try {
            const d = JSON.parse(line.slice(6));
            if (d.type === "text") setAnswer((p) => p + d.content);
            if (d.type === "done") {
              setChunks(d.retrieval_details || []);
              setAnswerSource(d.source_type || "local");
            }
          } catch {}
        }
      }
    } catch (e) { console.error(e); }
    setLoading(false);
    setStreaming(false);
  };

  // ── REPORT: STEP 1 — extract patient info before running the real analysis ─
  const extractPatientInfo = async () => {
    if (!file || extracting) return;
    setExtracting(true);
    setError("");
    setPatientInfo(null);
    setConfirmedPatient(null);
    setReportResult(null);
    setReportAnalysis("");

    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await authFetch(`${BASE}/extract-patient-info`, { method: "POST", body: fd });
      const data = await res.json();

      if (!data.success) {
        setError(data.error || "Could not extract patient info.");
        setExtracting(false);
        return;
      }

      setReportText(data.report_text);
      setExtractionMethod(data.extraction_method);
      setPatientInfo(data.patient_info);
    } catch {
      setError("Could not reach the server.");
    }
    setExtracting(false);
  };

  // ── REPORT: STEP 2 — user confirmed/edited patient info, now run analysis ──
  const analyze = async (confirmedInfo) => {
    if (!reportText || reportLoading) return;
    setConfirmedPatient(confirmedInfo);
    setReportLoading(true);
    setReportStreaming(true);
    setReportResult(null);
    setReportAnalysis("");
    setReportStatus("");
    setError("");

    try {
      const res = await authFetch(`${BASE}/analyze-report-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          report_text: reportText,
          patient_info: confirmedInfo,
          extraction_method: extractionMethod,
          filename: file?.name || null,
        }),
      });
      if (!res.body) throw new Error("No stream");

      const reader = res.body.getReader();
      const dec = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        for (const line of dec.decode(value).split("\n")) {
          if (!line.startsWith("data: ")) continue;
          try {
            const d = JSON.parse(line.slice(6));

            if (d.type === "status") {
              setReportStatus(d.content);
            }
            if (d.type === "toa") {
              setReportResult(prev => ({
                ...prev,
                toa: {
                  thought: d.thought,
                  observation: d.observation,
                  search_terms: d.search_terms,
                }
              }));
              setReportStatus("Generating analysis...");
            }
            if (d.type === "text") {
              setReportAnalysis(prev => prev + d.content);
            }
            if (d.type === "done") {
              setReportResult(prev => ({
                ...prev,
                sources: d.sources,
                extraction_method: d.extraction_method,
                report_id: d.report_id,
              }));
              setReportStatus("");
            }
            if (d.type === "error") {
              setError(d.content);
            }
          } catch {}
        }
      }
    } catch {
      setError("Could not reach the server.");
    }

    setReportLoading(false);
    setReportStreaming(false);
  };

  // ── PARSERS ───────────────────────────────────────────────────────────────
  const parseThought = (text) => {
    if (!text) return [];
    return text.split("\n").filter(l => l.trim()).map(line => {
      const param = line.match(/Parameter:\s*([^|]+)/i);
      const val   = line.match(/Value:\s*([^|]+)/i);
      const unit  = line.match(/Unit:\s*([^|]+)/i);
      const ref   = line.match(/Reference:\s*(.+)/i);
      return {
        name:  param ? param[1].trim() : "",
        value: val   ? val[1].trim()   : "",
        unit:  unit  ? unit[1].trim()  : "",
        ref:   ref   ? ref[1].trim()   : "",
      };
    }).filter(p => p.name);
  };

  const parseObservation = (text) => {
    if (!text) return [];
    return text.split("\n").filter(l => l.trim()).map(line => {
      const param  = line.match(/Parameter:\s*([^|]+)/i);
      const status = line.match(/Status:\s*([^|]+)/i);
      const cond   = line.match(/Possible condition:\s*(.+)/i);
      return {
        name:   param  ? param[1].trim()  : "",
        status: status ? status[1].trim() : "",
        cond:   cond   ? cond[1].trim()   : "",
      };
    }).filter(p => p.name);
  };

  // ── STYLES ────────────────────────────────────────────────────────────────
  const s = {
    page: {
      maxWidth: 780, margin: "0 auto", padding: "48px 24px 80px",
      fontFamily: "'Inter', 'Segoe UI', sans-serif",
      color: "#fff", fontSize: 16, lineHeight: 1.6,
      background: "#111", minHeight: "100vh",
    },
    topRow: { display: "flex", justifyContent: "space-between", alignItems: "flex-start" },
    heading: { fontSize: 20, fontWeight: 600, margin: "0 0 4px", color: "#fff" },
    sub: { fontSize: 14, color: "#aaa", margin: "0 0 32px" },
    userRow: { display: "flex", alignItems: "center", gap: 10, marginTop: 2 },
    userTag: { fontSize: 13, color: "#999" },
    logoutBtn: {
      fontSize: 13, color: "#aaa", background: "none", border: "1px solid #2a2a2a",
      borderRadius: 6, padding: "4px 10px", cursor: "pointer",
    },
    tabs: { display: "flex", gap: 2, marginBottom: 32, borderBottom: "1px solid #2a2a2a" },
    tab: (active) => ({
      padding: "8px 16px", border: "none", background: "none",
      fontSize: 16, fontWeight: active ? 600 : 400, color: "#fff",
      cursor: "pointer",
      borderBottom: active ? "2px solid #fff" : "2px solid transparent",
      marginBottom: -1, transition: "all 0.15s",
    }),
    row: { display: "flex", gap: 8, marginBottom: 24 },
    input: {
      flex: 1, padding: "10px 14px", fontSize: 16,
      border: "1px solid #2a2a2a", borderRadius: 8,
      outline: "none", color: "#fff", background: "#1a1a1a",
    },
    btn: (disabled) => ({
      padding: "10px 20px", fontSize: 16, fontWeight: 500,
      border: "1px solid #333", borderRadius: 8,
      background: disabled ? "#1a1a1a" : "#fff",
      color: disabled ? "#555" : "#111",
      cursor: disabled ? "not-allowed" : "pointer",
      whiteSpace: "nowrap", transition: "background 0.15s",
    }),
    card: {
      border: "1px solid #2a2a2a", borderRadius: 10,
      padding: "16px 20px", marginBottom: 12, background: "#1a1a1a",
    },
    sectionLabel: {
      fontSize: 11, fontWeight: 600, letterSpacing: "0.08em",
      textTransform: "uppercase", color: "#888",
      marginBottom: 10, marginTop: 24,
    },
    uploadBox: {
      border: "1px dashed #333", borderRadius: 10,
      padding: "36px 24px", textAlign: "center",
      cursor: "pointer", marginBottom: 16, background: "#1a1a1a",
    },
    pill: {
      display: "inline-block", padding: "3px 10px", borderRadius: 4,
      fontSize: 13, background: "#1a1a1a", border: "1px solid #2a2a2a",
      color: "#fff", marginRight: 6, marginBottom: 6,
    },
    sourceBadge: {
      display: "inline-block", padding: "2px 8px", borderRadius: 4,
      fontSize: 11, fontWeight: 600, letterSpacing: "0.04em",
      background: "#1f3a5f", color: "#9ec5f0", marginBottom: 10,
    },
    chunkMeta: { fontSize: 12, color: "#888", marginBottom: 4 },
    chunkText: { fontSize: 14, color: "#ccc", lineHeight: 1.6, marginTop: 8 },
    error: {
      border: "1px solid #4a1a1a", background: "#2a1010",
      borderRadius: 8, padding: "12px 16px",
      color: "#f88", fontSize: 14, marginBottom: 16,
    },
    status: {
      fontSize: 13, color: "#888", marginBottom: 16,
      display: "flex", alignItems: "center", gap: 8,
    },
    th: {
      textAlign: "left", padding: "8px 12px", color: "#888",
      fontWeight: 600, fontSize: 12, letterSpacing: "0.05em",
      textTransform: "uppercase", borderBottom: "1px solid #2a2a2a",
    },
    td: {
      padding: "10px 12px", borderBottom: "1px solid #1f1f1f",
      fontSize: 14, verticalAlign: "top", color: "#fff",
    },
  };

  // ── RENDER ────────────────────────────────────────────────────────────────
  return (
    <div style={s.page}>
      <div style={s.topRow}>
        <div>
          <p style={s.heading}>Medical RAG Assistant</p>
          <p style={s.sub}>Ask medical questions or upload a report for analysis.</p>
        </div>
        <div style={s.userRow}>
          <span style={s.userTag}>{username}</span>
          <button style={s.logoutBtn} onClick={logout}>Log out</button>
        </div>
      </div>

      <div style={s.tabs}>
        <button style={s.tab(tab === "chat")} onClick={() => setTab("chat")}>Chat</button>
        <button style={s.tab(tab === "report")} onClick={() => setTab("report")}>Analyze report</button>
        <button style={s.tab(tab === "history")} onClick={() => setTab("history")}>My reports</button>
      </div>

      {/* ── CHAT TAB ── */}
      {tab === "chat" && (
        <>
          <div style={s.row}>
            <input
              style={s.input}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && ask()}
              placeholder="Ask a medical question…"
            />
            <button style={s.btn(loading)} onClick={ask} disabled={loading}>
              {loading ? "Thinking…" : "Ask"}
            </button>
          </div>

          {answer && (
            <>
              <p style={s.sectionLabel}>Answer</p>
              {answerSource === "web_search" && (
                <div style={s.sourceBadge}>ANSWERED VIA GOOGLE SEARCH</div>
              )}
              <div style={{ ...s.card, lineHeight: 1.8, fontSize: 16, marginBottom: 28 }}>
                <ReactMarkdown>{answer + (streaming ? "▌" : "")}</ReactMarkdown>
              </div>

              {chunks.length > 0 && (
                <>
                  <p style={s.sectionLabel}>Sources</p>
                  {chunks.map((c, i) => (
                    <div key={i} style={s.card}>
                      <div style={s.chunkMeta}>
                        {c.source_file?.split("\\").pop() ?? "Unknown"} · page {(c.page ?? 0) + 1} · score {c.rerank_score?.toFixed(3) ?? "—"}
                      </div>
                      <div style={s.chunkText}>{c.text}</div>
                    </div>
                  ))}
                </>
              )}
            </>
          )}
        </>
      )}

      {/* ── REPORT TAB ── */}
      {tab === "report" && (
        <>
          <div style={s.uploadBox} onClick={() => fileRef.current.click()}>
            <div style={{ fontSize: 14, color: "#ccc" }}>Click to upload your report</div>
            <div style={{ fontSize: 13, color: "#666", marginTop: 4 }}>PDF · JPG · PNG · TXT</div>
            <input
              ref={fileRef}
              type="file"
              accept=".pdf,.jpg,.jpeg,.png,.txt,.webp"
              onChange={(e) => {
                const f = e.target.files[0];
                if (f) {
                  setFile(f);
                  setPatientInfo(null);
                  setConfirmedPatient(null);
                  setReportText(null);
                  setReportResult(null);
                  setReportAnalysis("");
                  setError("");
                }
              }}
              style={{ display: "none" }}
            />
          </div>

          {file && (
            <div style={{ fontSize: 13, color: "#aaa", marginBottom: 16 }}>
              Selected: <strong style={{ color: "#fff" }}>{file.name}</strong>
            </div>
          )}

          {/* Step 1 button — only shown before extraction has happened */}
          {!patientInfo && (
            <button
              style={{ ...s.btn(!file || extracting), marginBottom: 28 }}
              onClick={extractPatientInfo}
              disabled={!file || extracting}
            >
              {extracting ? "Reading report…" : "Read report"}
            </button>
          )}

          {error && <div style={s.error}>{error}</div>}

          {/* Step 2 — review/edit extracted patient info, then confirm to analyze */}
          {patientInfo && !confirmedPatient && (
            <PatientInfoCard patientInfo={patientInfo} onConfirm={analyze} />
          )}

          {/* status line */}
          {reportStatus && (
            <div style={s.status}>
              <span style={{ animation: "spin 1s linear infinite", display: "inline-block" }}>⟳</span>
              {reportStatus}
            </div>
          )}

          {/* Tables appear as soon as TOA is done */}
          {reportResult?.toa && (
            <>
              <p style={s.sectionLabel}>Parameters found</p>
              <div style={s.card}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      {["Parameter", "Value", "Unit", "Reference range"].map(h => (
                        <th key={h} style={s.th}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {parseThought(reportResult.toa.thought).map((p, i) => (
                      <tr key={i}>
                        <td style={s.td}>{p.name}</td>
                        <td style={s.td}>{p.value}</td>
                        <td style={s.td}>{p.unit}</td>
                        <td style={s.td}>{p.ref}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <p style={s.sectionLabel}>Abnormal values</p>
              <div style={s.card}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      {["Parameter", "Status", "Possible condition"].map(h => (
                        <th key={h} style={s.th}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {parseObservation(reportResult.toa.observation).map((p, i) => (
                      <tr key={i}>
                        <td style={s.td}>{p.name}</td>
                        <td style={{
                          ...s.td,
                          color: p.status === "HIGH" ? "#f87171" : p.status === "LOW" ? "#60a5fa" : "#fff"
                        }}>{p.status}</td>
                        <td style={s.td}>{p.cond}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <p style={s.sectionLabel}>Searched for</p>
              <div style={{ marginBottom: 24 }}>
                {reportResult.toa.search_terms?.length > 0
                  ? reportResult.toa.search_terms.map((t, i) => <span key={i} style={s.pill}>{t}</span>)
                  : <span style={{ fontSize: 13, color: "#666" }}>No terms searched.</span>}
              </div>
            </>
          )}

          {/* Analysis streams in word by word */}
          {reportAnalysis && (
            <>
              <p style={s.sectionLabel}>Full analysis</p>
              <div style={{ ...s.card, fontSize: 15, lineHeight: 1.8, marginBottom: 24 }}>
                <ReactMarkdown>{reportAnalysis + (reportStreaming ? "▌" : "")}</ReactMarkdown>
              </div>
            </>
          )}

          {/* Sources appear at the end */}
          {reportResult?.sources?.length > 0 && (
            <>
              <p style={s.sectionLabel}>Book sources</p>
              <div style={s.card}>
                {reportResult.sources.map((src, i) => (
                  <div key={i} style={{
                    fontSize: 13, color: "#aaa", padding: "7px 0",
                    borderBottom: i < reportResult.sources.length - 1 ? "1px solid #222" : "none",
                  }}>
                    {src.term} — {src.file?.split("\\").pop() ?? "Unknown"}, p.{(src.page ?? 0) + 1}
                  </div>
                ))}
              </div>
            </>
          )}
        </>
      )}

      {/* ── MY REPORTS TAB ── */}
      {tab === "history" && <MyReports token={token} />}

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        * { box-sizing: border-box; }
        body { margin: 0; background: #111; }
      `}</style>
    </div>
  );
}