import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import PatientInfoCard from "./PatientInfoCard";
import Login from "./Login";
import MyReports from "./MyReports";
import ActivityLog from "./ActivityLog";
import Profile from "./Profile";
import PageIndexSourceTree from "./components/PageIndexSourceTree";
const BASE = "http://127.0.0.1:8000";

function getUserLocation() {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error("Geolocation not supported by this browser."));
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => resolve({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
      (err) => reject(err),
      { timeout: 8000 }
    );
  });
}



async function readSSE(response, onEvent) {
  if (!response.body) throw new Error("No stream");
  const reader = response.body.getReader();
  const dec = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (value) buffer += dec.decode(value, { stream: true });
    if (done) {
      buffer += dec.decode(); // flush any remaining decoder state
    }

    // SSE messages are separated by a blank line ("\n\n").
    let sepIndex;
    while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
      const rawMessage = buffer.slice(0, sepIndex);
      buffer = buffer.slice(sepIndex + 2);

      for (const line of rawMessage.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        try {
          onEvent(JSON.parse(line.slice(6)));
        } catch (e) {
          console.error("SSE parse error:", e, line);
        }
      }
    }

    if (done) break;
  }
}

// Converts the backend's page_index hierarchy object

function transformPageIndexHierarchy(hierarchy) {
  if (!hierarchy || !Array.isArray(hierarchy.nodes)) return [];

  const transformSection = (sec) => ({
    id: sec.node_id,
    title: sec.title,
    level: sec.level ?? 1,
    children: Array.isArray(sec.nodes) ? sec.nodes.map(transformSection) : [],
  });

  return hierarchy.nodes.map((pageNode) => ({
    page_id: pageNode.node_id,
    page_number: (pageNode.page ?? 0) + 1,
    filename: pageNode.file,
    score: pageNode.score,
    content: pageNode.content || "",
    sections: Array.isArray(pageNode.nodes) ? pageNode.nodes.map(transformSection) : [],
  }));
}

export default function App() {
  // AUTH
  const [token, setToken] = useState();
  const [username, setUsername] = useState();
  const [isLoading, setIsLoading] = useState(true); // Track if we're checking localStorage
  const [firstTime, setFirstTime] = useState(false); // Track if this is first login (should go to profile)

  // RESTORE TOKEN FROM LOCALSTORAGE ON MOUNT
  useEffect(() => {
    const savedToken = localStorage.getItem("token");
    const savedUsername = localStorage.getItem("username");

    if (savedToken && savedUsername) {
      setToken(savedToken);
      setUsername(savedUsername);
      console.log(" Token restored from localStorage for user:", savedUsername);
    }
    
    setIsLoading(false); // Done checking localStorage
  }, []); // Only run once on mount

  const handleAuth = ({ token, username, isFirstTime }) => {
    localStorage.setItem("token", token);
    localStorage.setItem("username", username);
    setToken(token);
    setUsername(username);
    setFirstTime(isFirstTime || false);
  };

  const logout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("username");
    setToken("");
    setUsername("");
    setFirstTime(false);
  };

  const authFetch = async (url, options = {}) => {
    const res = await fetch(url, {
      ...options,
      headers: {
        ...(options.headers || {}),
        Authorization: `Bearer ${token}`,
      },
    });
    if (res.status === 401) logout();
    return res;
  };

  // Set initial tab to profile if first time, else chat
  const [tab, setTab] = useState(firstTime ? "profile" : "chat");

  // chat state
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [pageIndexAnswer, setPageIndexAnswer] = useState("");
  const [chunks, setChunks] = useState([]);
  const [pageIndexPages, setPageIndexPages] = useState([]);
  const [webSources, setWebSources] = useState([]);
  const [pageIndexWebSources, setPageIndexWebSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [answerSource, setAnswerSource] = useState(null);

  // Stop/abort support for in-flight chat requests. A single controller
  // covers the chat-stream + page-index-stream + nearby-doctors calls that
  // ask() fires for one submitted query, so Stop cancels all of them.
  const abortControllerRef = useRef(null);

  // nearby doctors state
  const [doctorsData, setDoctorsData] = useState(null);
  const [doctorsLoading, setDoctorsLoading] = useState(false);
  const [locationError, setLocationError] = useState("");

  // report state
  const [file, setFile] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportStatus, setReportStatus] = useState("");
  const [reportResult, setReportResult] = useState(null);
  const [reportAnalysis, setReportAnalysis] = useState("");
  const [reportStreaming, setReportStreaming] = useState(false);
  const [error, setError] = useState("");
  const fileRef = useRef(null);

  // hierarchical page-index sources as array of SourcePage objects
  // Each object has: { page_id, page_number, filename, content, sections }
  const [pageIndexSources, setPageIndexSources] = useState([]);

  // report-tab Page Index (dual answer, mirrors chat tab)
  const [reportPageIndexAnswer, setReportPageIndexAnswer] = useState("");
  const [reportPageIndexSources, setReportPageIndexSources] = useState([]);
  const [reportPageIndexStreaming, setReportPageIndexStreaming] = useState(false);

  // patient info extraction state
  const [extracting, setExtracting] = useState(false);
  const [patientInfo, setPatientInfo] = useState(null);
  const [reportText, setReportText] = useState(null);
  const [extractionMethod, setExtractionMethod] = useState(null);
  const [confirmedPatient, setConfirmedPatient] = useState(null);

  //  SHOW LOADING WHILE CHECKING STORAGE
  if (isLoading) {
    return <div style={{ color: "#fff", background: "#111", minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center" }}>Loading…</div>;
  }

  if (!token) return <Login onAuth={handleAuth} />;

  // FETCH NEARBY DOCTORS
  const fetchNearbyDoctors = async (query, signal) => {
    setDoctorsData(null);
    setLocationError("");
    setDoctorsLoading(true);
    try {
      const { lat, lng } = await getUserLocation();
      const res = await authFetch(
        `${BASE}/nearby-doctors?lat=${lat}&lng=${lng}&query=${encodeURIComponent(query)}`,
        { signal }
      );
      const data = await res.json();
      if (data.success) {
        setDoctorsData(data);
      } else {
        setLocationError("Could not fetch nearby doctor info.");
      }
    } catch (err) {
      if (err.name === "AbortError") {
        // Stopped by the user — nothing to show.
      } else if (err.code === 1) {
        setLocationError("Location access denied. Please allow location permission to see nearby doctors.");
      } else {
        setLocationError("Could not get your location. Please check browser permissions.");
      }
    }
    setDoctorsLoading(false);
  };

  // FETCH PAGE INDEX FOR REPORT

  const fetchReportPageIndex = async (query) => {
    setReportPageIndexAnswer("");
    setReportPageIndexSources([]);
    setReportPageIndexStreaming(true);
    try {
      const res = await authFetch(`${BASE}/page-index-stream?query=${encodeURIComponent(query)}`, { method: "POST" });
      await readSSE(res, (d) => {
        if (d.type === "text") setReportPageIndexAnswer((p) => p + d.content);
        if (d.type === "done") {
          const sources = transformPageIndexHierarchy(d.page_index_sources);
          setReportPageIndexSources(sources);
        }
        if (d.type === "error") console.error("report page-index-stream error:", d.content);
      });
    } catch (e) {
      console.error(e);
    }
    setReportPageIndexStreaming(false);
  };

  // CHAT
  const ask = async () => {
    if (!query.trim() || loading) return;

    // Fresh controller for this submission — Stop aborts everything tied to it.
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setLoading(true);
    setStreaming(true);
    setAnswer("");
    setPageIndexAnswer("");
    setChunks([]);
    setPageIndexPages([]);
    setWebSources([]);
    setPageIndexWebSources([]);
    setPageIndexSources([]);
    setAnswerSource(null);
    setDoctorsData(null);
    setLocationError("");

    // Always fetch nearby doctors — this is a medical app, every query is relevant
    fetchNearbyDoctors(query, controller.signal);

    try {
      const res = await authFetch(`${BASE}/chat-stream?query=${encodeURIComponent(query)}`, {
        method: "POST",
        signal: controller.signal,
      });
      await readSSE(res, (d) => {
        if (d.type === "text") setAnswer((p) => p + d.content);
        if (d.type === "done") {
          setChunks(d.retrieval_details || []);
          setWebSources(d.web_sources || []);
          setAnswerSource(d.source_type || "local");
          setPageIndexAnswer(d.page_index_answer || "");
          setPageIndexPages(d.page_index_pages || []);
          setPageIndexWebSources(d.page_index_web_sources || []);
        }
      });
    } catch (e) {
      if (e.name !== "AbortError") console.error(e);
    }

    // ALSO call page-index-stream in parallel (hierarchical page-index answer/sources)
    // Skip if the user already hit Stop.
    if (!controller.signal.aborted) {
      try {
        const piRes = await authFetch(`${BASE}/page-index-stream?query=${encodeURIComponent(query)}`, {
          method: "POST",
          signal: controller.signal,
        });
        await readSSE(piRes, (d) => {
          if (d.type === "text") setPageIndexAnswer((p) => p + d.content);
          if (d.type === "done") {
            // Backend returns a hierarchy OBJECT ({title, node_id, nodes: [...]}),
            // not an array — transform it into the array shape the tree component needs.
            const sources = transformPageIndexHierarchy(d.page_index_sources);
            console.log("Page Index Sources received:", sources); // Debug log
            setPageIndexSources(sources);
          }
          if (d.type === "error") console.error("page-index-stream error:", d.content);
        });
      } catch (e) {
        if (e.name !== "AbortError") console.error(e);
      }
    }

    setLoading(false);
    setStreaming(false);
  };

  // STOP — cancels the in-flight chat-stream / page-index-stream / nearby-doctors
  // calls for the current query, so the user can immediately edit and resend.
  const stopGeneration = () => {
    abortControllerRef.current?.abort();
    setLoading(false);
    setStreaming(false);
    setDoctorsLoading(false);
  };

  // REPORT: STEP 1
  const extractPatientInfo = async () => {
    if (!file || extracting) return;
    setExtracting(true);
    setError("");
    setPatientInfo(null);
    setConfirmedPatient(null);
    setReportText(null);
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

  // REPORT: STEP 2
  const analyze = async (confirmedInfo) => {
    if (!reportText || reportLoading) return;
    setConfirmedPatient(confirmedInfo);
    setReportLoading(true);
    setReportStreaming(true);
    setReportResult(null);
    setReportAnalysis("");
    setReportStatus("");
    setError("");
    setDoctorsData(null);
    setLocationError("");
    setReportPageIndexAnswer("");
    setReportPageIndexSources([]);

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

      await readSSE(res, (d) => {
        if (d.type === "status") setReportStatus(d.content);

        if (d.type === "toa") {
          setReportResult(prev => ({
            ...prev,
            toa: { thought: d.thought, observation: d.observation, search_terms: d.search_terms }
          }));
          setReportStatus("Generating analysis...");
          // fetch doctors AND page-index answer based on the abnormal-finding search terms
          if (d.search_terms?.length > 0) {
            const combinedQuery = d.search_terms.join(", ");
            fetchNearbyDoctors(combinedQuery);
            fetchReportPageIndex(combinedQuery);
          }
        }

        if (d.type === "clear") setReportAnalysis("");
        if (d.type === "text") setReportAnalysis(prev => prev + d.content);
        if (d.type === "done") {
          setReportResult(prev => ({
            ...prev,
            sources: d.sources,
            extraction_method: d.extraction_method,
            report_id: d.report_id,
          }));
          setReportStatus("");
        }
        if (d.type === "error") setError(d.content);
      });
    } catch {
      setError("Could not reach the server.");
    }

    setReportLoading(false);
    setReportStreaming(false);
  };

  // PARSERS
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

  // STYLES
  const s = {
    page: {
      maxWidth: 1200, margin: "0 auto", padding: "48px 24px 80px",
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
    stopBtn: {
      padding: "10px 20px", fontSize: 16, fontWeight: 500,
      border: "1px solid #4a1a1a", borderRadius: 8,
      background: "#2a1010", color: "#f88",
      cursor: "pointer", whiteSpace: "nowrap", transition: "background 0.15s",
    },
    card: {
      border: "1px solid #2a2a2a", borderRadius: 10,
      padding: "16px 20px", marginBottom: 12, background: "#1a1a1a",
    },
    doctorsCard: {
      border: "1px solid #1a3a2a", borderRadius: 10,
      padding: "16px 20px", marginBottom: 12, background: "#0f1f17",
    },
    sectionLabel: {
      fontSize: 11, fontWeight: 600, letterSpacing: "0.08em",
      textTransform: "uppercase", color: "#888",
      marginBottom: 10, marginTop: 24,
    },
    doctorsLabel: {
      fontSize: 11, fontWeight: 600, letterSpacing: "0.08em",
      textTransform: "uppercase", color: "#4ade80",
      marginBottom: 10, marginTop: 24,
      display: "flex", alignItems: "center", gap: 6,
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
    specialistBadge: {
      display: "inline-block", padding: "2px 10px", borderRadius: 4,
      fontSize: 12, fontWeight: 600, letterSpacing: "0.04em",
      background: "#14532d", color: "#86efac", marginBottom: 10,
      marginRight: 8,
    },
    locationBadge: {
      display: "inline-block", padding: "2px 10px", borderRadius: 4,
      fontSize: 12, fontWeight: 500,
      background: "#1a2a1a", color: "#6ee7b7", marginBottom: 10,
    },
    sourceBadge: {
      display: "inline-block", padding: "2px 8px", borderRadius: 4,
      fontSize: 11, fontWeight: 600, letterSpacing: "0.04em",
      background: "#1f3a5f", color: "#9ec5f0", marginBottom: 10,
    },
    pageIndexBadge: {
      display: "inline-block", padding: "2px 8px", borderRadius: 4,
      fontSize: 11, fontWeight: 600, letterSpacing: "0.04em",
      background: "#3a2a1f", color: "#e0a070", marginBottom: 10,
    },
    chunkMeta: { fontSize: 12, color: "#888", marginBottom: 4 },
    chunkText: { fontSize: 14, color: "#ccc", lineHeight: 1.6, marginTop: 8 },
    webSourceCard: {
      border: "1px solid #2a2a2a", borderRadius: 10,
      padding: "12px 16px", marginBottom: 10, background: "#1a1a1a",
      display: "block", textDecoration: "none",
    },
    webSourceTitle: { fontSize: 14, color: "#9ec5f0", fontWeight: 500 },
    webSourceUrl: { fontSize: 12, color: "#666", marginTop: 4, wordBreak: "break-all" },
    error: {
      border: "1px solid #4a1a1a", background: "#2a1010",
      borderRadius: 8, padding: "12px 16px",
      color: "#f88", fontSize: 14, marginBottom: 16,
    },
    locationError: {
      border: "1px solid #2a3a1a", background: "#1a2a10",
      borderRadius: 8, padding: "10px 14px",
      color: "#a3e635", fontSize: 13, marginBottom: 12,
    },
    status: {
      fontSize: 13, color: "#888", marginBottom: 16,
      display: "flex", alignItems: "center", gap: 8,
    },
    doctorsStatus: {
      fontSize: 13, color: "#4ade80", marginBottom: 16,
      display: "flex", alignItems: "center", gap: 8,
    },
    dualAnswersContainer: {
      display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 28,
    },
    answerCard: {
      border: "1px solid #2a2a2a", borderRadius: 10,
      padding: "16px 20px", background: "#1a1a1a",
      lineHeight: 1.8, fontSize: 16,
    },
    answerTitle: {
      fontSize: 12, fontWeight: 600, letterSpacing: "0.08em",
      textTransform: "uppercase", color: "#888",
      marginBottom: 12,
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

  // DOCTORS SECTION COMPONENT (shared between tabs)
  const DoctorsSection = () => (
    <>
      {doctorsLoading && (
        <div style={s.doctorsStatus}>
          <span style={{ animation: "spin 1s linear infinite", display: "inline-block" }}>⟳</span>
          Finding nearby doctors based on your location…
        </div>
      )}

      {locationError && <div style={s.locationError}>{locationError}</div>}

      {doctorsData && (
        <>
          <div style={s.doctorsLabel}>🏥 Nearby Doctors</div>
          <div style={{ marginBottom: 8 }}>
            <span style={s.specialistBadge}>
              {doctorsData.specialist.charAt(0).toUpperCase() + doctorsData.specialist.slice(1)}
            </span>
            <span style={s.locationBadge}>{doctorsData.location}</span>
          </div>
          <div style={{ ...s.doctorsCard, lineHeight: 1.8, fontSize: 15 }}>
            <ReactMarkdown>{doctorsData.info}</ReactMarkdown>
          </div>
        </>
      )}
    </>
  );

  // RENDER
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

      {/* ── TABS ── */}
      <div style={s.tabs}>
        <button style={s.tab(tab === "profile")}  onClick={() => setTab("profile")}>Profile</button>
        <button style={s.tab(tab === "chat")}     onClick={() => setTab("chat")}>Chat</button>
        <button style={s.tab(tab === "report")}   onClick={() => setTab("report")}>Analyze report</button>
        <button style={s.tab(tab === "history")}  onClick={() => setTab("history")}>My reports</button>
        <button style={s.tab(tab === "activity")} onClick={() => setTab("activity")}>Activity</button>
      </div>

      {/* ── PROFILE TAB ── */}
      {tab === "profile" && <Profile />}

      {/* ── CHAT TAB ── */}
      {tab === "chat" && (
        <>
          <div style={s.row}>
            <input
              style={s.input}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !loading && ask()}
              placeholder="Ask a medical question…"
            />
            {loading ? (
              <button style={s.stopBtn} onClick={stopGeneration}>
                Stop
              </button>
            ) : (
              <button style={s.btn(loading)} onClick={ask} disabled={loading}>
                Ask
              </button>
            )}
          </div>

          {(answer || pageIndexAnswer) && (
            <>
              <p style={s.sectionLabel}>Answers</p>

              {/* DUAL ANSWER CARDS */}
              <div style={s.dualAnswersContainer}>
                {/* RAG ANSWER CARD */}
                <div style={s.answerCard}>
                  <div style={s.answerTitle}>
                    RAG Answer
                    <div style={{ fontSize: 11, fontWeight: 400, color: "#888", marginTop: 4 }}>
                      {answerSource === "web_search" ? " Web Search Fallback" : "Best-ranked retrieval with reranking"}
                    </div>
                    {answerSource === "web_search" && (
                      <div style={{ ...s.sourceBadge, marginLeft: 0, marginTop: 6 }}>WEB SEARCH</div>
                    )}
                  </div>
                  <div style={{ fontSize: 14, lineHeight: 1.7, color: "#e0e0e0", marginTop: 12 }}>
                    <ReactMarkdown>{answer + (streaming ? "▌" : "")}</ReactMarkdown>
                  </div>
                </div>

                {/* PAGE INDEX ANSWER CARD */}
                {pageIndexAnswer && (
                  <div style={s.answerCard}>
                    <div style={s.answerTitle}>
                      Page Index Answer
                      <div style={{ fontSize: 11, fontWeight: 400, color: "#888", marginTop: 4 }}>
                        {pageIndexAnswer && pageIndexAnswer.includes("No information found")
                          ? " Web Search Fallback"
                          : "Direct page-level retrieval"}
                      </div>
                    </div>
                    <div style={{ fontSize: 14, lineHeight: 1.7, color: "#e0e0e0", marginTop: 12 }}>
                      <ReactMarkdown>{pageIndexAnswer}</ReactMarkdown>
                    </div>
                  </div>
                )}
              </div>

              {/* RAG SOURCES FROM KNOWLEDGE BASE */}
              {chunks.length > 0 && (
                <>
                  <p style={s.sectionLabel}>RAG Sources (from knowledge base)</p>
                  {chunks.map((c, i) => (
                    <div key={i} style={s.card}>
                      <div style={s.chunkMeta}>
                        <strong>{c.source_file?.split("\\").pop() ?? "Unknown"}</strong> · Page {(c.page ?? 0) + 1} · Relevance: {c.rerank_score?.toFixed(3) ?? "—"}
                      </div>
                      <div style={s.chunkText}>{c.text}</div>
                    </div>
                  ))}
                </>
              )}

              {/* PAGE INDEX SOURCES - HIERARCHICAL */}
              {pageIndexSources && pageIndexSources.length > 0 ? (
                <>
                  <p style={s.sectionLabel}>Page Index Sources (hierarchical)</p>
                  <PageIndexSourceTree sources={pageIndexSources} />
                </>
              ) : pageIndexPages.length > 0 ? (
                /* FALLBACK: FLAT PAGE INDEX SOURCES (if hierarchy not available) */
                <>
                  <p style={s.sectionLabel}>📖 Page Index Sources (document references)</p>
                  <div style={s.card}>
                    {pageIndexPages.map((pg, i) => (
                      <div key={i} style={{
                        fontSize: 13, color: "#aaa", padding: "10px 0",
                        borderBottom: i < pageIndexPages.length - 1 ? "1px solid #222" : "none",
                      }}>
                        <span style={{ ...s.pageIndexBadge, marginRight: 8 }}>Page {pg.page}</span>
                        <strong>{pg.file?.split("\\").pop() ?? "Unknown"}</strong>
                      </div>
                    ))}
                  </div>
                </>
              ) : null}

              {/* RAG WEB SOURCES (from fallback) */}
              {webSources.length > 0 && (
                <>
                  <p style={s.sectionLabel}>Web Sources (RAG fallback)</p>
                  {webSources.map((src, i) => {
                    return (
                      <a
                        key={i}
                        href={src.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={s.webSourceCard}
                      >
                        <div style={s.webSourceTitle}>{src.title}</div>
                        <div style={s.webSourceUrl}>{src.url}</div>
                      </a>
                    );
                  })}
                </>
              )}

              {/* PAGE INDEX WEB SOURCES (from fallback) */}
              {pageIndexWebSources.length > 0 && (
                <>
                  <p style={s.sectionLabel}>Web Sources (Page Index fallback)</p>
                  {pageIndexWebSources.map((src, i) => {
                    return (
                      <a
                        key={i}
                        href={src.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={s.webSourceCard}
                      >
                        <div style={s.webSourceTitle}>{src.title}</div>
                        <div style={s.webSourceUrl}>{src.url}</div>
                      </a>
                    );
                  })}
                </>
              )}

              {/* ── NEARBY DOCTORS IN CHAT TAB ── */}
              <p style={{ ...s.sectionLabel, marginTop: 32 }}>Doctor recommendations</p>
              <DoctorsSection />
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

          {patientInfo && !confirmedPatient && (
            <PatientInfoCard patientInfo={patientInfo} onConfirm={analyze} />
          )}

          {reportStatus && (
            <div style={s.status}>
              <span style={{ animation: "spin 1s linear infinite", display: "inline-block" }}>⟳</span>
              {reportStatus}
            </div>
          )}

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

          {(reportAnalysis || reportPageIndexAnswer) && (
            <>
              <p style={s.sectionLabel}>Full analysis</p>
              <div style={s.dualAnswersContainer}>
                {/* RAG ANALYSIS CARD */}
                <div style={s.answerCard}>
                  <div style={s.answerTitle}>
                    RAG Answer
                    <div style={{ fontSize: 11, fontWeight: 400, color: "#888", marginTop: 4 }}>
                      Synthesized from report + retrieved book passages
                    </div>
                  </div>
                  <div style={{ fontSize: 14, lineHeight: 1.7, color: "#e0e0e0", marginTop: 12 }}>
                    <ReactMarkdown>{reportAnalysis + (reportStreaming ? "▌" : "")}</ReactMarkdown>
                  </div>
                </div>

                {/* PAGE INDEX ANALYSIS CARD */}
                {(reportPageIndexAnswer || reportPageIndexStreaming) && (
                  <div style={s.answerCard}>
                    <div style={s.answerTitle}>
                      Page Index Answer
                      <div style={{ fontSize: 11, fontWeight: 400, color: "#888", marginTop: 4 }}>
                        {reportPageIndexAnswer && reportPageIndexAnswer.includes("No information found")
                          ? " Web Search Fallback"
                          : "Direct page-level retrieval"}
                      </div>
                    </div>
                    <div style={{ fontSize: 14, lineHeight: 1.7, color: "#e0e0e0", marginTop: 12 }}>
                      <ReactMarkdown>{reportPageIndexAnswer + (reportPageIndexStreaming ? "▌" : "")}</ReactMarkdown>
                    </div>
                  </div>
                )}
              </div>
            </>
          )}

          {reportResult?.sources?.length > 0 && (
            <>
              <p style={s.sectionLabel}>Book sources (RAG)</p>
              <div style={s.card}>
                {reportResult.sources.map((src, i) => {
                  const rowStyle = {
                    fontSize: 13, color: "#aaa", padding: "7px 0",
                    borderBottom: i < reportResult.sources.length - 1 ? "1px solid #222" : "none",
                  };

                  if (src.url) {
                    return (
                      <div key={i} style={rowStyle}>
                        {src.term} — <a href={src.url} target="_blank" rel="noopener noreferrer" style={{ color: "#9ec5f0" }}>Google Search source</a>
                      </div>
                    );
                  }

                  return (
                    <div key={i} style={rowStyle}>
                      {src.term} — {src.file?.split("\\").pop() ?? "Unknown"}, p.{(src.page ?? 0) + 1}
                    </div>
                  );
                })}
              </div>
            </>
          )}

          {reportPageIndexSources.length > 0 && (
            <>
              <p style={s.sectionLabel}>Page Index Sources (hierarchical)</p>
              <PageIndexSourceTree sources={reportPageIndexSources} />
            </>
          )}

          {/* ── NEARBY DOCTORS IN REPORT TAB ── */}
          <p style={{ ...s.sectionLabel, marginTop: 32 }}>Doctor recommendations</p>
          <DoctorsSection />
        </>
      )}

      {/* ── MY REPORTS TAB ── */}
      {tab === "history" && <MyReports token={token} />}

      {/* ── ACTIVITY TAB ── */}
      {tab === "activity" && <ActivityLog token={token} />}

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        * { box-sizing: border-box; }
        body { margin: 0; background: #111; }
        @media (max-width: 900px) {
          [style*="gridTemplateColumns: 1fr 1fr"] {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </div>
  );
}