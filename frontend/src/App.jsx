import { useState, useEffect } from "react";
import Login from "./Login";

const API = "http://localhost:5000/api";

// ── ICONS ──────────────────────────────────────────────────────────
const IconSearch = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
  </svg>
);

// ── STYLES chung ───────────────────────────────────────────────────
const S = {
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
    display: "flex", alignItems: "center",
    justifyContent: "space-between",
    backdropFilter: "blur(10px)",
    position: "sticky", top: 0, zIndex: 10,
  },
  headerLeft: { display: "flex", alignItems: "center", gap: "12px" },
  headerTitle: {
    fontSize: "18px", fontWeight: "700", margin: 0,
    background: "linear-gradient(90deg, #00c896, #00a8ff)",
    WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
  },
  userBadge: {
    display: "flex", alignItems: "center", gap: "8px",
    background: "rgba(0,200,150,0.08)",
    border: "1px solid rgba(0,200,150,0.2)",
    borderRadius: "20px", padding: "6px 14px",
    fontSize: "13px", color: "#00c896",
  },
  logoutBtn: {
    background: "transparent",
    border: "1px solid rgba(255,100,100,0.25)",
    borderRadius: "8px", padding: "6px 14px",
    color: "rgba(255,120,120,0.7)", fontSize: "13px",
    cursor: "pointer", fontFamily: "inherit",
    transition: "all 0.2s",
  },
  main: { maxWidth: "1100px", margin: "0 auto", padding: "36px 24px" },
  card: {
    background: "rgba(255,255,255,0.03)",
    border: "1px solid rgba(0,200,150,0.15)",
    borderRadius: "16px", padding: "28px 32px",
    marginBottom: "28px",
  },
  label: {
    fontSize: "12px", color: "#00c896", fontWeight: "700",
    letterSpacing: "0.8px", textTransform: "uppercase",
    marginBottom: "14px", display: "block",
  },
  input: {
    flex: 1, minWidth: "200px",
    background: "rgba(255,255,255,0.05)",
    border: "1px solid rgba(0,200,150,0.3)",
    borderRadius: "10px", padding: "12px 16px",
    color: "#e0e6f0", fontSize: "15px", outline: "none",
    fontFamily: "inherit",
  },
  select: {
    background: "rgba(255,255,255,0.05)",
    border: "1px solid rgba(0,200,150,0.3)",
    borderRadius: "10px", padding: "12px 16px",
    color: "#e0e6f0", fontSize: "14px", outline: "none", cursor: "pointer",
  },
  btnPrimary: {
    background: "linear-gradient(135deg, #00c896, #00a8ff)",
    border: "none", borderRadius: "10px", padding: "12px 24px",
    color: "#fff", fontWeight: "600", fontSize: "14px",
    cursor: "pointer", display: "flex", alignItems: "center", gap: "8px",
    fontFamily: "inherit", whiteSpace: "nowrap",
  },
  statsRow: {
    display: "grid", gridTemplateColumns: "repeat(3,1fr)",
    gap: "16px", marginBottom: "24px",
  },
  statCard: {
    background: "rgba(255,255,255,0.03)",
    border: "1px solid rgba(0,200,150,0.12)",
    borderRadius: "14px", padding: "20px 24px",
  },
  statLbl: {
    fontSize: "11px", color: "#6b8aad", fontWeight: "700",
    letterSpacing: "0.6px", textTransform: "uppercase", marginBottom: "6px",
  },
  statVal: { fontSize: "20px", fontWeight: "700", color: "#e0e6f0" },
  tableCard: {
    background: "rgba(255,255,255,0.03)",
    border: "1px solid rgba(0,200,150,0.12)",
    borderRadius: "16px", overflow: "hidden",
  },
  tableHead: {
    padding: "18px 24px",
    borderBottom: "1px solid rgba(255,255,255,0.05)",
    fontSize: "15px", fontWeight: "700",
  },
  th: {
    padding: "12px 20px", textAlign: "left",
    fontSize: "11px", fontWeight: "700", color: "#6b8aad",
    letterSpacing: "0.7px", textTransform: "uppercase",
    background: "rgba(0,0,0,0.2)",
    borderBottom: "1px solid rgba(255,255,255,0.05)",
  },
  td: {
    padding: "14px 20px", fontSize: "14px",
    borderBottom: "1px solid rgba(255,255,255,0.04)",
    verticalAlign: "middle",
  },
  rank: {
    width: "32px", height: "32px", borderRadius: "8px",
    background: "rgba(0,200,150,0.1)",
    border: "1px solid rgba(0,200,150,0.2)",
    display: "flex", alignItems: "center", justifyContent: "center",
    fontSize: "13px", fontWeight: "700", color: "#00c896",
  },
  catBadge: {
    display: "inline-block", padding: "3px 10px",
    borderRadius: "6px", fontSize: "12px",
    background: "rgba(0,168,255,0.1)", color: "#00a8ff",
    border: "1px solid rgba(0,168,255,0.2)",
  },
  errorBox: {
    background: "rgba(255,80,80,0.08)",
    border: "1px solid rgba(255,80,80,0.25)",
    borderRadius: "12px", padding: "16px 20px",
    color: "#ff6b6b", fontSize: "14px", marginBottom: "20px",
  },
  modeBadge: (mode) => ({
    display: "inline-block", padding: "4px 12px",
    borderRadius: "20px", fontSize: "13px", fontWeight: "600",
    background: mode === "Hybrid" ? "rgba(0,200,150,0.15)" : "rgba(0,168,255,0.15)",
    color: mode === "Hybrid" ? "#00c896" : "#00a8ff",
    border: `1px solid ${mode === "Hybrid" ? "rgba(0,200,150,0.3)" : "rgba(0,168,255,0.3)"}`,
  }),
};

// ══════════════════════════════════════════════════════════════════════
// TRANG KHÁCH — Top sản phẩm phổ biến
// ══════════════════════════════════════════════════════════════════════
function GuestPage({ onLoginClick }) {
  const [popular, setPopular] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API}/popular`)
      .then(r => r.json())
      .then(d => { setPopular(d.products || []); setLoading(false); })
      .catch(() => { setError("Không kết nối được server!"); setLoading(false); });
  }, []);

  return (
    <div style={S.app}>
      {/* Header */}
      <div style={S.header}>
        <div style={S.headerLeft}>
          <span style={{ fontSize: "22px" }}>🛒</span>
          <h1 style={S.headerTitle}>Hệ Thống Gợi Ý Sản Phẩm</h1>
        </div>
        <button onClick={onLoginClick} style={S.btnPrimary}>
          ✨ Đăng nhập để xem gợi ý cá nhân
        </button>
      </div>

      <div style={S.main}>
        {/* Banner */}
        <div style={{
          ...S.card,
          background: "linear-gradient(135deg, rgba(0,200,150,0.08), rgba(0,168,255,0.06))",
          border: "1px solid rgba(0,200,150,0.2)",
          textAlign: "center", padding: "40px",
        }}>
          <div style={{ fontSize: "40px", marginBottom: "12px" }}>👋</div>
          <h2 style={{ fontSize: "20px", fontWeight: "700", marginBottom: "8px" }}>
            Chào mừng bạn!
          </h2>
          <p style={{ color: "#6b8aad", fontSize: "14px", marginBottom: "20px" }}>
            Đăng nhập để nhận gợi ý sản phẩm <strong style={{ color: "#00c896" }}>cá nhân hóa</strong> theo sở thích của bạn
          </p>
          <button onClick={onLoginClick} style={{ ...S.btnPrimary, margin: "0 auto" }}>
            ✨ Đăng nhập ngay
          </button>
        </div>

        {/* Top sản phẩm */}
        <div style={S.tableCard}>
          <div style={{ ...S.tableHead, display: "flex", alignItems: "center", gap: "8px" }}>
            🔥 Top Sản Phẩm Được Mua Nhiều Nhất
          </div>

          {loading && (
            <div style={{ textAlign: "center", padding: "40px", color: "#6b8aad" }}>
              ⏳ Đang tải...
            </div>
          )}

          {error && <div style={{ ...S.errorBox, margin: "16px" }}>❌ {error}</div>}

          {!loading && !error && (
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={S.th}>#</th>
                  <th style={S.th}>Tên sản phẩm</th>
                  <th style={S.th}>Danh mục</th>
                  <th style={S.th}>Lượt mua</th>
                  <th style={S.th}>Rating TB</th>
                </tr>
              </thead>
              <tbody>
                {popular.map((item, idx) => (
                  <tr key={idx} style={{
                    background: idx % 2 === 0 ? "transparent" : "rgba(255,255,255,0.01)",
                  }}>
                    <td style={S.td}>
                      <div style={S.rank}>{idx + 1}</div>
                    </td>
                    <td style={{ ...S.td, fontWeight: "500", color: "#c8d8e8" }}>
                      {item.product_name}
                    </td>
                    <td style={S.td}>
                      <span style={S.catBadge}>{item.category}</span>
                    </td>
                    <td style={{ ...S.td, color: "#00c896", fontWeight: "600" }}>
                      {item.buy_count} lượt
                    </td>
                    <td style={{ ...S.td, color: "#ffd700" }}>
                      ⭐ {item.avg_rating}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&display=swap');`}</style>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════
// TRANG USER — Gợi ý cá nhân
// ══════════════════════════════════════════════════════════════════════
function UserPage({ userId, onLogout }) {
  const [topN, setTopN] = useState(10);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  // Tự tìm gợi ý khi vào trang
  useEffect(() => { fetchRecommend(); }, []);

  const fetchRecommend = async () => {
    setLoading(true); setError(""); setResult(null);
    try {
      const res = await fetch(`${API}/recommend?user_id=${userId}&top_n=${topN}`);
      const data = await res.json();
      if (!res.ok) setError(data.error || "Lỗi không xác định");
      else setResult(data);
    } catch {
      setError("Không kết nối được server!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={S.app}>
      {/* Header */}
      <div style={S.header}>
        <div style={S.headerLeft}>
          <span style={{ fontSize: "22px" }}>🛒</span>
          <h1 style={S.headerTitle}>Hệ Thống Gợi Ý Sản Phẩm</h1>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div style={S.userBadge}>👤 {userId}</div>
          <button
            onClick={onLogout}
            style={S.logoutBtn}
            onMouseEnter={e => e.target.style.background = "rgba(255,100,100,0.08)"}
            onMouseLeave={e => e.target.style.background = "transparent"}
          >
            Đăng xuất
          </button>
        </div>
      </div>

      <div style={S.main}>
        {/* Control */}
        <div style={S.card}>
          <span style={S.label}>⚙️ Cài đặt gợi ý</span>
          <div style={{ display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap" }}>
            <select
              style={S.select} value={topN}
              onChange={e => setTopN(Number(e.target.value))}
            >
              {[5, 10, 15, 20].map(n => (
                <option key={n} value={n}>Top {n} sản phẩm</option>
              ))}
            </select>
            <button style={S.btnPrimary} onClick={fetchRecommend} disabled={loading}>
              <IconSearch />
              {loading ? "Đang tính..." : "Làm mới gợi ý"}
            </button>
          </div>
        </div>

        {error && <div style={S.errorBox}>❌ {error}</div>}

        {loading && (
          <div style={{ textAlign: "center", padding: "60px", color: "#6b8aad" }}>
            <div style={{ fontSize: "32px", marginBottom: "12px" }}>⏳</div>
            Đang tính toán gợi ý cá nhân...
          </div>
        )}

        {result && !loading && (
          <>
            {/* Stats */}
            <div style={S.statsRow}>
              <div style={S.statCard}>
                <div style={S.statLbl}>👤 User ID</div>
                <div style={S.statVal}>{result.user_id}</div>
              </div>
              <div style={S.statCard}>
                <div style={S.statLbl}>📊 Tổng tương tác</div>
                <div style={S.statVal}>{result.n_interactions}</div>
                <div style={{ marginTop: "8px", display: "flex", gap: "6px", flexWrap: "wrap" }}>
                  {Object.entries(result.source_counts || {}).map(([src, cnt]) => (
                    <span key={src} style={{
                      padding: "2px 8px", borderRadius: "20px", fontSize: "11px",
                      background: "rgba(255,255,255,0.05)", color: "#8aa8c8",
                      border: "1px solid rgba(255,255,255,0.08)",
                    }}>
                      {src}: {cnt}
                    </span>
                  ))}
                </div>
              </div>
              <div style={S.statCard}>
                <div style={S.statLbl}>⚙️ Mode</div>
                <div style={{ marginTop: "4px" }}>
                  <span style={S.modeBadge(result.mode)}>
                    {result.mode === "Hybrid" ? "🔀" : "❄️"} {result.mode}
                  </span>
                  <div style={{ fontSize: "12px", color: "#4a6a8a", marginTop: "8px" }}>
                    {result.mode === "Hybrid"
                      ? "CB × 0.4 + CF × 0.6"
                      : "Content-Based Only"}
                  </div>
                </div>
              </div>
            </div>

            {/* Table */}
            <div style={S.tableCard}>
              <div style={{ ...S.tableHead }}>
                🎯 Top {result.recommendations.length} Sản Phẩm Gợi Ý Cho Bạn
              </div>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr>
                    <th style={S.th}>#</th>
                    <th style={S.th}>Tên sản phẩm</th>
                    <th style={S.th}>Danh mục</th>
                    <th style={S.th}>Điểm phù hợp</th>
                  </tr>
                </thead>
                <tbody>
                  {result.recommendations.map((item, idx) => (
                    <tr key={idx} style={{
                      background: idx % 2 === 0 ? "transparent" : "rgba(255,255,255,0.01)",
                    }}>
                      <td style={S.td}><div style={S.rank}>{item.rank}</div></td>
                      <td style={{ ...S.td, fontWeight: "500", color: "#c8d8e8" }}>
                        {item.product_name}
                      </td>
                      <td style={S.td}>
                        <span style={S.catBadge}>{item.category}</span>
                      </td>
                      <td style={S.td}>
                        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                          <span style={{ color: "#00c896", fontWeight: "600", fontSize: "13px", minWidth: "50px" }}>
                            {item.score}
                          </span>
                          <div style={{ flex: 1, height: "6px", background: "rgba(255,255,255,0.06)", borderRadius: "3px" }}>
                            <div style={{
                              height: "100%", width: `${item.score * 100}%`,
                              background: "linear-gradient(90deg, #00c896, #00a8ff)",
                              borderRadius: "3px",
                            }} />
                          </div>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&display=swap');`}</style>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════
// ROOT APP — Quản lý flow đăng nhập
// ══════════════════════════════════════════════════════════════════════
export default function App() {
  // page: "login" | "guest" | "user"
  const [page, setPage] = useState("login");
  const [userId, setUserId] = useState("");

  const handleLogin = (id) => {
    setUserId(id);
    setPage("user");
  };

  const handleGuest = () => setPage("guest");
  const handleLogout = () => { setUserId(""); setPage("login"); };

  if (page === "login") return <Login onLogin={handleLogin} onGuest={handleGuest} />;
  if (page === "guest") return <GuestPage onLoginClick={() => setPage("login")} />;
  return <UserPage userId={userId} onLogout={handleLogout} />;
}