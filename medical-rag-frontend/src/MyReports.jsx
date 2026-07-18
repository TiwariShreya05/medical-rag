import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";

const BASE = "http://127.0.0.1:8000";

export default function MyReports({ token }) {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [downloadingId, setDownloadingId] = useState(null);
  const [deletingId, setDeletingId] = useState(null);
  const [previewReport, setPreviewReport] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [confirmDeleteId, setConfirmDeleteId] = useState(null);

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${BASE}/my-reports`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      
      if (!res.ok) throw new Error();
      const data = await res.json();
      setReports(data.reports || []);
    } catch {
      setError("Could not load your reports.");
    }
    setLoading(false);
  };

  useEffect(() => {
  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${BASE}/my-reports`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error();
      const data = await res.json();
      setReports(data.reports || []);
    } catch {
      setError("Could not load your reports.");
    }
    setLoading(false);
  };
  load();
}, [token]);

  const download = async (reportId) => {
    setDownloadingId(reportId);
    try {
      const res = await fetch(`${BASE}/my-reports/${reportId}/download`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error();
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `report_${reportId}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch {
      setError("Could not download that report.");
    }
    setDownloadingId(null);
  };

  const deleteReport = async (reportId) => {
    setDeletingId(reportId);
    try {
      const res = await fetch(`${BASE}/my-reports/${reportId}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
      
      if (!res.ok) throw new Error();
      setReports((prev) => prev.filter((r) => r.id !== reportId));
      setConfirmDeleteId(null);
      if (previewReport?.id === reportId) setPreviewReport(null);
    } catch {
      setError("Could not delete that report.");
    }
    setDeletingId(null);
  };

  const openPreview = async (reportId) => {
    setPreviewLoading(true);
    setPreviewReport(null);
    try {
      const res = await fetch(`${BASE}/my-reports/${reportId}/preview`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      
      if (!res.ok) throw new Error();
      const data = await res.json();
      setPreviewReport(data);
    } catch {
      setError("Could not load report preview.");
    }
    setPreviewLoading(false);
  };

  const closePreview = () => {
    setPreviewReport(null);
    setPreviewLoading(false);
  };

  if (loading) return <div style={s.muted}>Loading your reports…</div>;
  if (error) return <div style={s.error}>{error}</div>;
  if (reports.length === 0) return <div style={s.muted}>No reports analyzed yet.</div>;

  return (
    <div>
      {reports.map((r) => (
        <div
          key={r.id}
          style={s.card}
          onClick={() => openPreview(r.id)}
          onMouseEnter={e => e.currentTarget.style.borderColor = "#555"}
          onMouseLeave={e => e.currentTarget.style.borderColor = "#2a2a2a"}
        >
          <div style={s.row}>
            <div>
              <div style={s.title}>{r.patient_info?.name || r.filename || "Report"}</div>
              <div style={s.meta}>
                {r.patient_info?.age ? `Age ${r.patient_info.age}` : ""}
                {r.patient_info?.age && r.created_at ? " · " : ""}
                {r.created_at}
              </div>
              <div style={s.clickHint}>Click to preview</div>
            </div>

            <div style={s.btnGroup}>
              {/* Download */}
              <button
                style={s.btn(downloadingId === r.id)}
                onClick={(e) => { e.stopPropagation(); download(r.id); }}
                disabled={downloadingId === r.id}
              >
                {downloadingId === r.id ? "Preparing…" : "Download PDF"}
              </button>

              {/* Delete */}
              {confirmDeleteId === r.id ? (
                <div style={s.confirmRow}>
                  <span style={s.confirmText}>Delete?</span>
                  <button
                    style={s.deleteConfirmBtn}
                    onClick={(e) => { e.stopPropagation(); deleteReport(r.id); }}
                    disabled={deletingId === r.id}
                  >
                    {deletingId === r.id ? "…" : "Yes"}
                  </button>
                  <button
                    style={s.cancelBtn}
                    onClick={(e) => { e.stopPropagation(); setConfirmDeleteId(null); }}
                  >
                    No
                  </button>
                </div>
              ) : (
                <button
                  style={s.deleteBtn}
                  onClick={(e) => { e.stopPropagation(); setConfirmDeleteId(r.id); }}
                >
                  Delete
                </button>
              )}
            </div>
          </div>
        </div>
      ))}

      {/* ── PREVIEW MODAL ── */}
      {(previewLoading || previewReport) && (
        <div style={s.overlay} onClick={closePreview}>
          <div style={s.modal} onClick={(e) => e.stopPropagation()}>
            <div style={s.modalHeader}>
              <span style={s.modalTitle}>
                {previewReport
                  ? previewReport.patient_info?.name || previewReport.filename || "Report"
                  : "Loading preview…"}
              </span>
              <button style={s.closeBtn} onClick={closePreview}>✕</button>
            </div>

            {previewLoading && (
              <div style={{ padding: 24, color: "#888", fontSize: 14 }}>Loading…</div>
            )}

            {previewReport && (
              <div style={s.modalBody}>

                {/* Patient info */}
                {previewReport.patient_info && (
                  <div style={s.section}>
                    <div style={s.sectionLabel}>Patient info</div>
                    <div style={s.infoGrid}>
                      {Object.entries(previewReport.patient_info)
                        .filter(([, v]) => v)
                        .map(([k, v]) => (
                          <div key={k} style={s.infoRow}>
                            <span style={s.infoKey}>{k}</span>
                            <span style={s.infoVal}>{v}</span>
                          </div>
                        ))}
                    </div>
                  </div>
                )}

                {/* Parameters */}
                {previewReport.thought && (
                  <div style={s.section}>
                    <div style={s.sectionLabel}>Parameters found</div>
                    <div style={s.preText}>{previewReport.thought}</div>
                  </div>
                )}

                {/* Abnormal values */}
                {previewReport.observation && (
                  <div style={s.section}>
                    <div style={s.sectionLabel}>Abnormal values</div>
                    <div style={s.preText}>{previewReport.observation}</div>
                  </div>
                )}

                {/* Search terms */}
                {previewReport.search_terms && (() => {
                  try {
                    const terms = typeof previewReport.search_terms === "string"
                      ? JSON.parse(previewReport.search_terms)
                      : previewReport.search_terms;
                    return terms.length > 0 ? (
                      <div style={s.section}>
                        <div style={s.sectionLabel}>Searched for</div>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                          {terms.map((t, i) => (
                            <span key={i} style={s.pill}>{t}</span>
                          ))}
                        </div>
                      </div>
                    ) : null;
                  } catch { return null; }
                })()}

                {/* Full analysis */}
                {previewReport.analysis && (
                  <div style={s.section}>
                    <div style={s.sectionLabel}>Full analysis</div>
                    <div style={s.analysisBox}>
                      <ReactMarkdown>{previewReport.analysis}</ReactMarkdown>
                    </div>
                  </div>
                )}

                {/* Sources */}
                {previewReport.sources && (() => {
                  try {
                    const sources = typeof previewReport.sources === "string"
                      ? JSON.parse(previewReport.sources)
                      : previewReport.sources;
                    return sources.length > 0 ? (
                      <div style={s.section}>
                        <div style={s.sectionLabel}>Sources</div>
                        {sources.map((src, i) => (
                          <div key={i} style={s.sourceRow}>
                            {src.term} — {src.file?.split("\\").pop() ?? "Unknown"}, p.{(src.page ?? 0) + 1}
                          </div>
                        ))}
                      </div>
                    ) : null;
                  } catch { return null; }
                })()}

                {/* Download inside modal */}
                <button
                  style={{ ...s.btn(downloadingId === previewReport.id), marginTop: 8, width: "100%" }}
                  onClick={() => download(previewReport.id)}
                  disabled={downloadingId === previewReport.id}
                >
                  {downloadingId === previewReport.id ? "Preparing…" : "Download PDF"}
                </button>

              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

const s = {
  card: {
    border: "1px solid #2a2a2a",
    borderRadius: 10,
    padding: "14px 18px",
    marginBottom: 12,
    background: "#1a1a1a",
    cursor: "pointer",
    transition: "border-color 0.15s",
  },
  
  row: { display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 },
  title: { fontSize: 15, fontWeight: 600, color: "#fff" },
  meta: { fontSize: 12, color: "#888", marginTop: 2 },
  clickHint: { fontSize: 11, color: "#555", marginTop: 4 },
  btnGroup: { display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" },
  btn: (disabled) => ({
    padding: "7px 14px",
    fontSize: 13,
    fontWeight: 500,
    border: "1px solid #333",
    borderRadius: 7,
    background: disabled ? "#1a1a1a" : "#fff",
    color: disabled ? "#555" : "#111",
    cursor: disabled ? "not-allowed" : "pointer",
    whiteSpace: "nowrap",
  }),
  deleteBtn: {
    padding: "7px 14px",
    fontSize: 13,
    fontWeight: 500,
    border: "1px solid #4a1a1a",
    borderRadius: 7,
    background: "transparent",
    color: "#f87171",
    cursor: "pointer",
    whiteSpace: "nowrap",
  },
  confirmRow: { display: "flex", alignItems: "center", gap: 6 },
  confirmText: { fontSize: 13, color: "#f87171" },
  deleteConfirmBtn: {
    padding: "5px 10px", fontSize: 12, fontWeight: 600,
    border: "1px solid #4a1a1a", borderRadius: 6,
    background: "#4a1a1a", color: "#f87171", cursor: "pointer",
  },
  cancelBtn: {
    padding: "5px 10px", fontSize: 12,
    border: "1px solid #333", borderRadius: 6,
    background: "transparent", color: "#aaa", cursor: "pointer",
  },
  muted: { fontSize: 14, color: "#888", padding: "20px 0" },
  error: {
    border: "1px solid #4a1a1a", background: "#2a1010",
    borderRadius: 8, padding: "12px 16px", color: "#f88", fontSize: 14,
  },
  overlay: {
    position: "fixed", inset: 0,
    background: "rgba(0,0,0,0.75)",
    display: "flex", alignItems: "center", justifyContent: "center",
    zIndex: 1000, padding: 24,
  },
  modal: {
    background: "#1a1a1a", border: "1px solid #2a2a2a",
    borderRadius: 12, width: "100%", maxWidth: 640,
    maxHeight: "85vh", display: "flex", flexDirection: "column",
    overflow: "hidden",
  },
  modalHeader: {
    display: "flex", justifyContent: "space-between", alignItems: "center",
    padding: "16px 20px", borderBottom: "1px solid #2a2a2a",
    flexShrink: 0,
  },
  modalTitle: { fontSize: 16, fontWeight: 600, color: "#fff" },
  closeBtn: {
    background: "none", border: "none", color: "#888",
    fontSize: 18, cursor: "pointer", padding: "0 4px",
  },
  modalBody: { overflowY: "auto", padding: "20px", flex: 1 },
  section: { marginBottom: 24 },
  sectionLabel: {
    fontSize: 11, fontWeight: 600, letterSpacing: "0.08em",
    textTransform: "uppercase", color: "#888", marginBottom: 10,
  },
  infoGrid: { display: "flex", flexDirection: "column", gap: 6 },
  infoRow: { display: "flex", gap: 12, fontSize: 14 },
  infoKey: { color: "#888", width: 120, flexShrink: 0, textTransform: "capitalize" },
  infoVal: { color: "#fff" },
  preText: {
    fontSize: 13, color: "#ccc", lineHeight: 1.7,
    whiteSpace: "pre-wrap", fontFamily: "monospace",
    background: "#111", borderRadius: 8, padding: "12px 14px",
  },
  
  pill: {
    display: "inline-block", padding: "3px 10px", borderRadius: 4,
    fontSize: 12, background: "#111", border: "1px solid #2a2a2a", color: "#fff",
  },
  
  analysisBox: {
    fontSize: 14, color: "#ccc", lineHeight: 1.8,
    background: "#111", borderRadius: 8, padding: "14px 16px",
  },
  sourceRow: {
    fontSize: 13, color: "#aaa", padding: "6px 0",
    borderBottom: "1px solid #222",
  },
};
