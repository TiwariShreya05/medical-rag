import { useState, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";

const BASE = "http://127.0.0.1:8000";

const CHAT_ACTIONS = new Set([
  "query",
  "page_index_query",
  "nearby_doctors",
  "off_topic_rejected",
]);

// Pair up entries (chat "query", "page_index_query", "nearby_doctors", and
// "off_topic_rejected") that came from the same question submission, so
// they can be shown together like the original Chat page did. Entries are
// paired when they have the exact same query text and were logged within
// PAIR_WINDOW_MIN minutes of each other. This is generic across any number
// of action types, so adding a new action type later won't require
// touching this function again.
const PAIR_WINDOW_MIN = 20;

function groupIntoSessions(rawActivities) {
  const sorted = [...rawActivities].sort(
    (a, b) => new Date(a.timestamp) - new Date(b.timestamp)
  );
  const used = new Set();
  const sessions = [];

  for (let i = 0; i < sorted.length; i++) {
    const current = sorted[i];
    if (used.has(current.id)) continue;
    used.add(current.id);

    const group = [current];

    for (let j = i + 1; j < sorted.length; j++) {
      const candidate = sorted[j];
      if (used.has(candidate.id)) continue;
      if (candidate.query !== current.query) continue;
      if (group.some((g) => g.action === candidate.action)) continue; // no duplicate action types in one session
      const diffMin =
        Math.abs(new Date(candidate.timestamp) - new Date(current.timestamp)) /
        60000;
      if (diffMin <= PAIR_WINDOW_MIN) {
        group.push(candidate);
        used.add(candidate.id);
      }
    }

    const chatEntry = group.find((e) => e.action === "query") || null;
    const pageIndexEntry = group.find((e) => e.action === "page_index_query") || null;
    const doctorsEntry = group.find((e) => e.action === "nearby_doctors") || null;
    const blockedEntry = group.find((e) => e.action === "off_topic_rejected") || null;

    const latestTimestamp = group
      .map((e) => e.timestamp)
      .sort()
      .pop();

    sessions.push({
      id: current.id,
      query: current.query,
      timestamp: latestTimestamp || current.timestamp,
      chat: chatEntry,
      pageIndex: pageIndexEntry,
      doctors: doctorsEntry,
      blocked: blockedEntry,
    });
  }

  return sessions.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
}

export default function ActivityLog({ token }) {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedSession, setSelectedSession] = useState(null);
  const [error, setError] = useState("");

  const fetchActivityLog = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${BASE}/activity-log`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error("Failed to fetch activity log");
      const data = await res.json();
      const onlyChats = (data.activities || []).filter((a) =>
        CHAT_ACTIONS.has(a.action)
      );
      setSessions(groupIntoSessions(onlyChats));
    } catch (err) {
      setError("Could not load activity log. Please try again.");
      console.error(err);
    }
    setLoading(false);
  }, [token]);

  useEffect(() => {
    fetchActivityLog();
  }, [fetchActivityLog]);

  const formatTimestamp = (iso) => {
    const date = new Date(iso);
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      year: date.getFullYear() !== new Date().getFullYear() ? "numeric" : undefined,
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  };

  const getAnswer = (entry) => entry?.metadata?.answer || entry?.result || "";
  const getSources = (entry) => entry?.metadata?.sources || null;
  const getWebSources = (entry) => entry?.metadata?.web_sources || [];
  const getDoctorInfo = (entry) => entry?.metadata?.info || "";
  const getSpecialist = (entry) => entry?.metadata?.specialist || "";
  const getCity = (entry) => entry?.metadata?.city || "";

  const styles = {
    container: {
      padding: "24px",
      fontFamily: "'Inter', 'Segoe UI', sans-serif",
      color: "#fff",
      background: "#111",
      minHeight: "100vh",
    },
    header: { fontSize: 20, fontWeight: 600, marginBottom: 4, color: "#fff" },
    subheader: { fontSize: 13, color: "#888", marginBottom: 24 },
    listContainer: { maxHeight: "calc(100vh - 140px)", overflowY: "auto" },
    listItem: (isHovered) => ({
      padding: "14px 16px",
      marginBottom: 8,
      background: isHovered ? "#1a1a1a" : "transparent",
      borderRadius: "8px",
      cursor: "pointer",
      transition: "all 0.15s",
      border: `1px solid ${isHovered ? "#2a2a2a" : "#1a1a1a"}`,
    }),
    listItemTitle: {
      fontSize: 14,
      fontWeight: 500,
      color: "#fff",
      marginBottom: 6,
      display: "-webkit-box",
      WebkitLineClamp: 1,
      WebkitBoxOrient: "vertical",
      overflow: "hidden",
    },
    listItemTime: { fontSize: 12, color: "#666", whiteSpace: "nowrap", marginLeft: 12 },
    emptyState: { textAlign: "center", padding: "40px 20px", color: "#888" },
    errorState: {
      border: "1px solid #4a1a1a",
      background: "#2a1010",
      borderRadius: 8,
      padding: "12px 16px",
      color: "#f88",
      fontSize: 14,
      marginBottom: 16,
    },
    loadingState: { textAlign: "center", padding: "40px 20px", color: "#888" },

    fullPage: {
      position: "fixed",
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: "#111",
      zIndex: 1000,
      display: "flex",
      flexDirection: "column",
    },
    pageHeader: {
      display: "flex",
      alignItems: "center",
      gap: 14,
      padding: "18px 24px",
      borderBottom: "1px solid #222",
      position: "sticky",
      top: 0,
      background: "#111",
      zIndex: 1,
    },
    backButton: {
      background: "#1a1a1a",
      border: "1px solid #2a2a2a",
      color: "#ccc",
      fontSize: 14,
      cursor: "pointer",
      padding: "8px 14px",
      borderRadius: 8,
    },
    pageTimestamp: { fontSize: 12, color: "#888" },
    pageBody: {
      flex: 1,
      overflowY: "auto",
      padding: "32px 24px 60px",
      maxWidth: 1000,
      width: "100%",
      margin: "0 auto",
    },
    questionBlock: { marginBottom: 32 },
    turnLabel: {
      fontSize: 11,
      fontWeight: 600,
      letterSpacing: "0.06em",
      textTransform: "uppercase",
      color: "#666",
      marginBottom: 10,
    },
    questionText: {
      fontSize: 16,
      fontWeight: 500,
      color: "#fff",
      lineHeight: 1.6,
    },
    divider: { border: "none", borderTop: "1px solid #222", margin: "28px 0" },
    dualAnswersContainer: {
      display: "grid",
      gridTemplateColumns: "1fr 1fr",
      gap: 16,
      marginBottom: 28,
    },
    answerCard: {
      background: "#1a1a1a",
      border: "1px solid #2a2a2a",
      borderRadius: 10,
      padding: "20px",
    },
    answerCardHeader: { marginBottom: 12 },
    answerCardTitle: {
      fontSize: 12,
      fontWeight: 600,
      letterSpacing: "0.08em",
      textTransform: "uppercase",
      color: "#888",
    },
    answerCardSubtitle: { fontSize: 11, fontWeight: 400, color: "#888", marginTop: 4 },
    answerText: {
      fontSize: 14,
      color: "#e0e0e0",
      lineHeight: 1.8,
      marginTop: 12,
    },
    sourcesSection: { marginTop: 20 },
    sourcesTitle: {
      fontSize: 11,
      fontWeight: 600,
      letterSpacing: "0.08em",
      textTransform: "uppercase",
      color: "#888",
      marginBottom: 10,
    },
    sourceCard: {
      background: "#1f1f1f",
      border: "1px solid #2a2a2a",
      borderRadius: 8,
      padding: "12px 16px",
      marginBottom: 8,
      fontSize: 13,
      color: "#ccc",
    },
    sourceFile: { color: "#fff", fontWeight: 500 },
    noAnswer: {
      fontSize: 14,
      color: "#666",
      fontStyle: "italic",
      background: "#1a1a1a",
      border: "1px solid #2a2a2a",
      borderRadius: 8,
      padding: "14px 16px",
    },
    blockedNotice: {
      fontSize: 14,
      color: "#ff9e9e",
      background: "#1f1414",
      border: "1px solid #3a1f1f",
      borderRadius: 8,
      padding: "14px 16px",
      lineHeight: 1.6,
    },
    sectionLabel: {
      fontSize: 11,
      fontWeight: 600,
      letterSpacing: "0.08em",
      textTransform: "uppercase",
      color: "#888",
      marginBottom: 10,
      marginTop: 24,
    },
    webSourceCard: {
      border: "1px solid #2a2a2a",
      borderRadius: 10,
      padding: "12px 16px",
      marginBottom: 10,
      background: "#1a1a1a",
      display: "block",
      textDecoration: "none",
      cursor: "pointer",
      transition: "all 0.15s",
    },
    webSourceTitle: { fontSize: 13, color: "#9ec5f0", fontWeight: 500 },
    webSourceUrl: { fontSize: 12, color: "#666", marginTop: 4, wordBreak: "break-all" },
  };

  const flattenPageIndexSources = (node, depth = 0, acc = []) => {
    if (!node) return acc;
    if (node.file) {
      acc.push({
        title: node.title,
        file: node.file,
        page: node.page,
        score: node.score,
        depth,
      });
    }
    (node.nodes || []).forEach((child) => flattenPageIndexSources(child, depth + 1, acc));
    return acc;
  };

  const [hoveredId, setHoveredId] = useState(null);

  if (loading) {
    return (
      <div style={styles.container}>
        <div style={styles.header}>Activity Log</div>
        <div style={styles.loadingState}>Loading your activity…</div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>Activity Log</div>
      <div style={styles.subheader}>Your recent questions and searches</div>

      {error && <div style={styles.errorState}>{error}</div>}

      {sessions.length === 0 ? (
        <div style={styles.emptyState}>
          <p style={{ fontSize: 14, marginBottom: 8 }}>No queries yet</p>
          <p style={{ fontSize: 13, color: "#666" }}>
            Your chat questions will appear here.
          </p>
        </div>
      ) : (
        <div style={styles.listContainer}>
          {sessions.map((session) => (
            <div
              key={session.id}
              style={styles.listItem(hoveredId === session.id)}
              onMouseEnter={() => setHoveredId(session.id)}
              onMouseLeave={() => setHoveredId(null)}
              onClick={() => setSelectedSession(session)}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={styles.listItemTitle}>{session.query || "—"}</div>
                </div>
                <div style={styles.listItemTime}>{formatTimestamp(session.timestamp)}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {selectedSession && (
        <div style={styles.fullPage}>
          <div style={styles.pageHeader}>
            <button style={styles.backButton} onClick={() => setSelectedSession(null)}>
              ← Back
            </button>
            <span style={styles.pageTimestamp}>
              {new Date(selectedSession.timestamp).toLocaleString()}
            </span>
          </div>

          <div style={styles.pageBody}>
            <div style={styles.questionBlock}>
              <div style={styles.turnLabel}>You asked</div>
              <div style={styles.questionText}>{selectedSession.query}</div>
            </div>

            {/* OFF-TOPIC / BLOCKED NOTICE */}
            {selectedSession.blocked &&
              !selectedSession.chat &&
              !selectedSession.pageIndex &&
              !selectedSession.doctors && (
                <div style={styles.blockedNotice}>
                  This question was outside the medical assistant's scope and was
                  not answered by the RAG pipeline.
                </div>
              )}

            {/* DUAL ANSWERS CONTAINER */}
            {(selectedSession.chat || selectedSession.pageIndex) && (
              <>
                <p style={styles.sectionLabel}>Answers</p>
                <div style={styles.dualAnswersContainer}>
                  {/* RAG ANSWER */}
                  {selectedSession.chat && (
                    <div style={styles.answerCard}>
                      <div style={styles.answerCardHeader}>
                        <div style={styles.answerCardTitle}>RAG Answer</div>
                        <div style={styles.answerCardSubtitle}>
                          Best-ranked retrieval with reranking
                        </div>
                      </div>
                      {getAnswer(selectedSession.chat) ? (
                        <div style={styles.answerText}>
                          <ReactMarkdown>{getAnswer(selectedSession.chat)}</ReactMarkdown>
                        </div>
                      ) : (
                        <div style={styles.noAnswer}>
                          No saved response for this entry.
                        </div>
                      )}
                    </div>
                  )}

                  {/* PAGE INDEX ANSWER */}
                  {selectedSession.pageIndex && (
                    <div style={styles.answerCard}>
                      <div style={styles.answerCardHeader}>
                        <div style={styles.answerCardTitle}>Page Index Answer</div>
                        <div style={styles.answerCardSubtitle}>
                          Direct page-level retrieval
                        </div>
                      </div>
                      {getAnswer(selectedSession.pageIndex) ? (
                        <div style={styles.answerText}>
                          <ReactMarkdown>{getAnswer(selectedSession.pageIndex)}</ReactMarkdown>
                        </div>
                      ) : (
                        <div style={styles.noAnswer}>
                          No saved response for this entry.
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </>
            )}

            {/* SUGGESTED DOCTORS */}
            {selectedSession.doctors && (
              <>
                <p style={styles.sectionLabel}>Suggested Doctors</p>
                <div style={styles.answerCard}>
                  <div style={styles.answerCardHeader}>
                    <div style={styles.answerCardTitle}>
                      {getSpecialist(selectedSession.doctors) || "Nearby Doctors"}
                    </div>
                    {getCity(selectedSession.doctors) && (
                      <div style={styles.answerCardSubtitle}>
                        Near {getCity(selectedSession.doctors)}
                      </div>
                    )}
                  </div>
                  {getDoctorInfo(selectedSession.doctors) ? (
                    <div style={styles.answerText}>
                      <ReactMarkdown>{getDoctorInfo(selectedSession.doctors)}</ReactMarkdown>
                    </div>
                  ) : (
                    <div style={styles.noAnswer}>
                      No saved doctor suggestion for this entry.
                    </div>
                  )}
                </div>
              </>
            )}

            {/* RAG SOURCES */}
            {selectedSession.chat && getSources(selectedSession.chat)?.length > 0 && (
              <>
                <p style={styles.sectionLabel}>RAG Sources (from knowledge base)</p>
                {getSources(selectedSession.chat).map((s, i) => (
                  <div key={i} style={styles.sourceCard}>
                    <span style={styles.sourceFile}>{s.file?.split("\\").pop() ?? "Unknown"}</span> · Page {(s.page ?? 0) + 1}
                    {s.score != null ? ` · Relevance: ${s.score.toFixed(3)}` : ""}
                  </div>
                ))}
              </>
            )}

            {/* PAGE INDEX SOURCES */}
            {selectedSession.pageIndex && getSources(selectedSession.pageIndex) && (
              <>
                <p style={styles.sectionLabel}>Page Index Sources (hierarchical)</p>
                {flattenPageIndexSources(getSources(selectedSession.pageIndex)).map((s, i) => (
                  <div
                    key={i}
                    style={{ ...styles.sourceCard, marginLeft: s.depth * 14 }}
                  >
                    {s.page != null ? (
                      <>
                        <span style={styles.sourceFile}>Page {s.page + 1}</span> · {s.file?.split("\\").pop() ?? "Unknown"}
                        {s.score != null ? ` · Relevance: ${s.score.toFixed(3)}` : ""}
                      </>
                    ) : (
                      s.title
                    )}
                  </div>
                ))}
              </>
            )}

            {/* RAG WEB SOURCES */}
            {selectedSession.chat && getWebSources(selectedSession.chat)?.length > 0 && (
              <>
                <p style={styles.sectionLabel}>Web Sources (RAG fallback)</p>
                {getWebSources(selectedSession.chat).map((s, i) => (
                  <a
                    key={i}
                    href={s.url || "#"}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={styles.webSourceCard}
                  >
                    <div style={styles.webSourceTitle}>{s.title || "Untitled"}</div>
                    <div style={styles.webSourceUrl}>{s.url || ""}</div>
                  </a>
                ))}
              </>
            )}
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @media (max-width: 900px) {
          [style*="gridTemplateColumns: 1fr 1fr"] {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </div>
  );
}