export const S = {
  app: {
    minHeight: "100vh",
    background: "linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%)",
    color: "#e0e6f0",
    fontFamily: "'Sora', 'Segoe UI', sans-serif",
  },
  header: {
    background: "rgba(0,200,150,0.04)",
    borderBottom: "1px solid rgba(0,200,150,0.12)",
    padding: "16px 40px",
    display: "flex", alignItems: "center", justifyContent: "space-between",
    backdropFilter: "blur(10px)", position: "sticky", top: 0, zIndex: 10,
  },
  headerLeft: { display: "flex", alignItems: "center", gap: "12px" },
  headerTitle: {
    fontSize: "18px", fontWeight: "700", margin: 0,
    background: "linear-gradient(90deg, #00c896, #00a8ff)",
    WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
  },
  userBadge: {
    display: "flex", alignItems: "center", gap: "8px",
    background: "rgba(0,200,150,0.08)", border: "1px solid rgba(0,200,150,0.2)",
    borderRadius: "20px", padding: "6px 14px", fontSize: "13px", color: "#00c896",
  },
  logoutBtn: {
    background: "transparent", border: "1px solid rgba(255,100,100,0.25)",
    borderRadius: "8px", padding: "6px 14px", color: "rgba(255,120,120,0.7)",
    fontSize: "13px", cursor: "pointer", fontFamily: "inherit", transition: "all 0.2s",
  },
  main: { maxWidth: "1100px", margin: "0 auto", padding: "36px 24px" },
  card: {
    background: "rgba(255,255,255,0.03)", border: "1px solid rgba(0,200,150,0.15)",
    borderRadius: "16px", padding: "28px 32px", marginBottom: "28px",
  },
  label: {
    fontSize: "12px", color: "#00c896", fontWeight: "700",
    letterSpacing: "0.8px", textTransform: "uppercase", marginBottom: "14px", display: "block",
  },
  select: {
    background: "#0d1b2a",
    border: "1px solid rgba(0,200,150,0.3)",
    borderRadius: "10px", padding: "12px 16px",
    color: "#e0e6f0",
    fontSize: "14px", outline: "none", cursor: "pointer",
    WebkitAppearance: "none",
    MozAppearance: "none",
    appearance: "none",
  },
  btnPrimary: {
    background: "linear-gradient(135deg, #00c896, #00a8ff)", border: "none",
    borderRadius: "10px", padding: "12px 24px", color: "#fff", fontWeight: "600",
    fontSize: "14px", cursor: "pointer", display: "flex", alignItems: "center",
    gap: "8px", fontFamily: "inherit", whiteSpace: "nowrap",
  },
  statsRow: { display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: "16px", marginBottom: "24px" },
  statCard: {
    background: "rgba(255,255,255,0.03)", border: "1px solid rgba(0,200,150,0.12)",
    borderRadius: "14px", padding: "20px 24px",
  },
  statLbl: {
    fontSize: "11px", color: "#6b8aad", fontWeight: "700",
    letterSpacing: "0.6px", textTransform: "uppercase", marginBottom: "6px",
  },
  statVal: { fontSize: "20px", fontWeight: "700", color: "#e0e6f0" },
  tableCard: {
    background: "rgba(255,255,255,0.03)", border: "1px solid rgba(0,200,150,0.12)",
    borderRadius: "16px", overflow: "hidden",
  },
  tableHead: {
    padding: "18px 24px", borderBottom: "1px solid rgba(255,255,255,0.05)",
    fontSize: "15px", fontWeight: "700",
  },
  th: {
    padding: "12px 20px", textAlign: "left", fontSize: "11px", fontWeight: "700",
    color: "#6b8aad", letterSpacing: "0.7px", textTransform: "uppercase",
    background: "rgba(0,0,0,0.2)", borderBottom: "1px solid rgba(255,255,255,0.05)",
  },
  td: { padding: "14px 20px", fontSize: "14px", borderBottom: "1px solid rgba(255,255,255,0.04)", verticalAlign: "middle" },
  rank: {
    width: "32px", height: "32px", borderRadius: "8px",
    background: "rgba(0,200,150,0.1)", border: "1px solid rgba(0,200,150,0.2)",
    display: "flex", alignItems: "center", justifyContent: "center",
    fontSize: "13px", fontWeight: "700", color: "#00c896",
  },
  catBadge: {
    display: "inline-block", padding: "3px 10px", borderRadius: "6px", fontSize: "12px",
    background: "rgba(0,168,255,0.1)", color: "#00a8ff", border: "1px solid rgba(0,168,255,0.2)",
  },
  errorBox: {
    background: "rgba(255,80,80,0.08)", border: "1px solid rgba(255,80,80,0.25)",
    borderRadius: "12px", padding: "16px 20px", color: "#ff6b6b", fontSize: "14px", marginBottom: "20px",
  },
  modeBadge: (mode) => ({
    display: "inline-block", padding: "4px 12px", borderRadius: "20px",
    fontSize: "13px", fontWeight: "600",
    background: mode === "Hybrid" ? "rgba(0,200,150,0.15)" : "rgba(0,168,255,0.15)",
    color: mode === "Hybrid" ? "#00c896" : "#00a8ff",
    border: `1px solid ${mode === "Hybrid" ? "rgba(0,200,150,0.3)" : "rgba(0,168,255,0.3)"}`,
  }),
};