import { useState } from "react";
import { Pencil, Check } from "lucide-react";

/**
 * Shows extracted patient info as editable fields. Call onConfirm(cleanedInfo)
 * when the user clicks "Confirm & Analyze".
 *
 * Usage:
 *   <PatientInfoCard patientInfo={extractedInfo} onConfirm={(info) => analyze(info)} />
 */
export default function PatientInfoCard({ patientInfo, onConfirm }) {
  const [editing, setEditing] = useState(true); // open in edit mode right after extraction
  const [form, setForm] = useState({
    name: patientInfo?.name || "",
    age: patientInfo?.age ?? "",
    sex: patientInfo?.sex || "",
    height_cm: patientInfo?.height_cm ?? "",
    weight_kg: patientInfo?.weight_kg ?? "",
    blood_group: patientInfo?.blood_group || "",
    allergies: (patientInfo?.allergies || []).join(", "),
  });

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleConfirm = () => {
    const cleaned = {
      name: form.name.trim() || null,
      age: form.age ? parseInt(form.age, 10) : null,
      sex: form.sex || null,
      height_cm: form.height_cm ? parseFloat(form.height_cm) : null,
      weight_kg: form.weight_kg ? parseFloat(form.weight_kg) : null,
      blood_group: form.blood_group.trim() || null,
      allergies: form.allergies
        .split(",")
        .map((a) => a.trim())
        .filter(Boolean),
    };
    setEditing(false);
    onConfirm(cleaned);
  };

  const fields = [
    { key: "name", label: "Name", type: "text" },
    { key: "age", label: "Age", type: "number" },
    { key: "sex", label: "Sex", type: "select", options: ["Male", "Female", "Other"] },
    { key: "height_cm", label: "Height (cm)", type: "number" },
    { key: "weight_kg", label: "Weight (kg)", type: "number" },
    { key: "blood_group", label: "Blood Group", type: "text" },
    { key: "allergies", label: "Allergies (comma separated)", type: "text" },
  ];

  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <span style={styles.title}>Patient Information</span>
        {!editing && (
          <button style={styles.iconBtn} onClick={() => setEditing(true)}>
            <Pencil size={16} />
          </button>
        )}
      </div>

      <div style={styles.grid}>
        {fields.map((f) => (
          <div key={f.key} style={styles.fieldRow}>
            <label style={styles.label}>{f.label}</label>
            {editing ? (
              f.type === "select" ? (
                <select
                  style={styles.input}
                  value={form[f.key]}
                  onChange={(e) => handleChange(f.key, e.target.value)}
                >
                  <option value="">Select</option>
                  {f.options.map((o) => (
                    <option key={o} value={o}>
                      {o}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  style={styles.input}
                  type={f.type}
                  value={form[f.key]}
                  onChange={(e) => handleChange(f.key, e.target.value)}
                  placeholder="Not detected"
                />
              )
            ) : (
              <span style={styles.value}>{form[f.key] || "—"}</span>
            )}
          </div>
        ))}
      </div>

      {editing && (
        <div style={styles.actions}>
          <button style={styles.confirmBtn} onClick={handleConfirm}>
            <Check size={16} /> Confirm & Analyze
          </button>
        </div>
      )}
    </div>
  );
}

const styles = {
  card: {
    background: "#0a0a0a",
    border: "1px solid #2a2a2a",
    borderRadius: "10px",
    padding: "20px",
    marginBottom: "20px",
    color: "#f5f5f5",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "14px",
  },
  title: {
    fontSize: "18px",
    fontWeight: 600,
  },
  iconBtn: {
    background: "transparent",
    border: "1px solid #333",
    borderRadius: "6px",
    color: "#ccc",
    padding: "6px",
    cursor: "pointer",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "14px",
  },
  fieldRow: {
    display: "flex",
    flexDirection: "column",
    gap: "4px",
  },
  label: {
    fontSize: "13px",
    color: "#999",
  },
  input: {
    background: "#161616",
    border: "1px solid #333",
    borderRadius: "6px",
    color: "#f5f5f5",
    padding: "8px 10px",
    fontSize: "15px",
  },
  value: {
    fontSize: "16px",
    color: "#f5f5f5",
  },
  actions: {
    marginTop: "16px",
    display: "flex",
    justifyContent: "flex-end",
  },
  confirmBtn: {
    background: "#1f6f3f",
    border: "none",
    borderRadius: "6px",
    color: "#fff",
    padding: "10px 16px",
    fontSize: "14px",
    fontWeight: 600,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    gap: "6px",
  },
};