import { useState, useEffect } from "react";
import Login from "./Login";
import { S } from "./styles";

const API = "http://localhost:5000/api";

const IconSearch = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
  </svg>
);

const ReasonBadge = ({ reason }) => {
  const cfg = {
    cb: { label: "Tương đồng nội dung", bg: "rgba(255,180,0,0.1)", color: "#ffb800", border: "rgba(255,180,0,0.25)" },
    cf: { label: "Người tương tự thích", bg: "rgba(180,0,255,0.1)", color: "#c060ff", border: "rgba(180,0,255,0.25)" },
  };
  const c = cfg[reason] || cfg.cb;
  return (
    <span style={{
      display: "inline-block", padding: "3px 10px", borderRadius: "6px",
      fontSize: "11px", fontWeight: "600",
      background: c.bg, color: c.color, border: `1px solid ${c.border}`,
    }}>{c.label}</span>
  );
};

const MODEL_TABS = [
  { key: "hybrid",  label: "Hybrid",       desc: "CB × 0.4 + CF × 0.6" },
  { key: "content", label: "Content-Based", desc: "Dựa trên nội dung sản phẩm" },
  { key: "collab",  label: "Collaborative", desc: "Dựa trên hành vi người dùng" },
];

const MODE_INFO = {
  "Hybrid": {
    icon: "🔀",
    formula: "CB × 0.4 + CF × 0.6",
    desc: "Kết hợp Content-Based và Collaborative Filtering để tối ưu độ chính xác",
  },
  "Cold Start": {
    icon: "❄️",
    formula: "Content-Based Only",
    desc: "Người dùng mới, chưa đủ dữ liệu — dùng nội dung sản phẩm để gợi ý",
  },
  "Content-Based": {
    icon: "📄",
    formula: "Cosine Similarity (TF-IDF)",
    desc: "Gợi ý dựa trên đặc trưng nội dung: danh mục, mô tả, thuộc tính sản phẩm",
  },
  "Collaborative": {
    icon: "👥",
    formula: "Neural Collaborative Filtering",
    desc: "Gợi ý dựa trên hành vi mua/xem của những người dùng có sở thích tương tự",
  },
};

const ModelTabs = ({ active, onChange }) => (
  <div style={{ display: "flex", gap: "8px", marginBottom: "20px", flexWrap: "wrap" }}>
    {MODEL_TABS.map(t => (
      <button key={t.key} onClick={() => onChange(t.key)} style={{
        padding: "10px 18px", borderRadius: "10px", fontFamily: "inherit",
        fontSize: "13px", fontWeight: "600", cursor: "pointer", transition: "all 0.2s",
        background: active === t.key ? "linear-gradient(135deg,#00c896,#00a8ff)" : "rgba(255,255,255,0.04)",
        color: active === t.key ? "#fff" : "#8aa8c8",
        border: active === t.key ? "none" : "1px solid rgba(255,255,255,0.08)",
      }}>
        {t.label}
        <span style={{ display: "block", fontSize: "10px", fontWeight: "400", opacity: 0.75, marginTop: "2px" }}>
          {t.desc}
        </span>
      </button>
    ))}
  </div>
);

const MetricsPanel = () => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open || metrics) return;
    setLoading(true);
    fetch(`${API}/metrics`)
      .then(r => r.json())
      .then(d => { setMetrics(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, [open]);

  const lowerIsBetter = new Set(['rmse', 'mae']);
  const best = {};
  if (metrics) {
    const models = ['content_based', 'collaborative', 'hybrid'];
    const allKeys = new Set([
      ...Object.keys(metrics.content_based || {}),
      ...Object.keys(metrics.collaborative || {}),
      ...Object.keys(metrics.hybrid || {}),
    ]);
    allKeys.forEach(k => {
      const vals = models.map(m => metrics[m]?.[k]).filter(v => v !== undefined);
      best[k] = lowerIsBetter.has(k) ? Math.min(...vals) : Math.max(...vals);
    });
  }

  const modelCfg = [
    { key: 'content_based',  label: 'Content-Based', color: '#ffb800' },
    { key: 'collaborative',  label: 'Collaborative',  color: '#c060ff' },
    { key: 'hybrid',         label: 'Hybrid',         color: '#00c896' },
  ];
};

const ModeCard = ({ mode, metrics }) => {
  const [showMetrics, setShowMetrics] = useState(false);
  const info = MODE_INFO[mode] || MODE_INFO["Hybrid"];

  const modelKey =
    mode === "Hybrid" ? "hybrid"
    : mode === "Content-Based" || mode === "Cold Start" ? "content_based"
    : "collaborative";

  const modeMetrics = metrics?.[modelKey];

  const toggleBtnStyle = {
    background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: "6px", padding: "3px 8px", color: "#8aa8c8",
    fontSize: "11px", cursor: "pointer", fontFamily: "inherit", transition: "all 0.2s",
  };

  return (
    <div style={S.statCard}>
      <div style={S.statLbl}>Mode</div>

      {/* Badge tên mode */}
      <div style={{ marginTop: "4px", marginBottom: "8px" }}>
        <span style={S.modeBadge(mode)}>
          {info.icon} {mode}
        </span>
      </div>

      <div style={{
        fontSize: "11px", color: "#00c896", fontWeight: "600",
        fontFamily: "monospace", marginBottom: "4px",
      }}>
        {info.formula}
      </div>

      <div style={{ fontSize: "11px", color: "#4a6a8a", marginBottom: "10px", lineHeight: "1.5" }}>
        {info.desc}
      </div>

      {modeMetrics && (
        <>
          <button
            onClick={() => setShowMetrics(v => !v)}
            style={toggleBtnStyle}
            onMouseEnter={e => e.currentTarget.style.background = "rgba(0,200,150,0.1)"}
            onMouseLeave={e => e.currentTarget.style.background = "rgba(255,255,255,0.05)"}
          >
            {showMetrics ? "Ẩn chỉ số ▲" : "Xem chỉ số ▼"}
          </button>

          {showMetrics && (
            <div style={{
              display: "flex", flexWrap: "wrap", gap: "6px",
              borderTop: "1px solid rgba(255,255,255,0.05)",
              paddingTop: "10px", marginTop: "10px",
            }}>
              {Object.entries(modeMetrics).map(([k, v]) => (
                <div key={k} style={{
                  background: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.07)",
                  borderRadius: "6px", padding: "4px 8px", textAlign: "center",
                }}>
                  <div style={{ fontSize: "10px", color: "#6b8aad", textTransform: "uppercase", letterSpacing: "0.5px" }}>
                    {k.toUpperCase()}
                  </div>
                  <div style={{ fontSize: "13px", fontWeight: "700", color: "#e0e6f0" }}>
                    {typeof v === "number" ? v.toFixed(4) : v}
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
};

const InteractionCard = ({ nInteractions, sourceCounts, interactions }) => {
  const [showDetail, setShowDetail] = useState(false);

  const toggleBtnStyle = {
    background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: "6px", padding: "3px 8px", color: "#8aa8c8",
    fontSize: "11px", cursor: "pointer", fontFamily: "inherit", transition: "all 0.2s",
  };

  const eventColor = (type) => {
    const t = (type || "").toLowerCase();
    if (t === "purchase" || t === "buy") return { bg: "rgba(0,200,150,0.1)", color: "#00c896", border: "rgba(0,200,150,0.2)" };
    if (t === "cart")                    return { bg: "rgba(255,180,0,0.1)",  color: "#ffb800", border: "rgba(255,180,0,0.2)" };
    return                                      { bg: "rgba(0,168,255,0.1)",  color: "#00a8ff", border: "rgba(0,168,255,0.2)" };
  };

  return (
    <div style={{ ...S.statCard, display: "flex", flexDirection: "column" }}>
      <div style={S.statLbl}>Tổng tương tác</div>

      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginBottom: "12px" }}>
        <div style={{ ...S.statVal, fontSize: "36px", lineHeight: 1 }}>{nInteractions}</div>
        {interactions && interactions.length > 0 && (
          <button
            onClick={() => setShowDetail(v => !v)}
            style={{ ...toggleBtnStyle, marginTop: "8px" }}
            onMouseEnter={e => e.currentTarget.style.background = "rgba(0,200,150,0.1)"}
            onMouseLeave={e => e.currentTarget.style.background = "rgba(255,255,255,0.05)"}
          >
            {showDetail ? "Ẩn ▲" : "Chi tiết ▼"}
          </button>
        )}
      </div>

      <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
        {Object.entries(sourceCounts || {}).map(([src, cnt]) => (
          <span key={src} style={{
            padding: "2px 8px", borderRadius: "20px", fontSize: "11px",
            background: "rgba(255,255,255,0.05)", color: "#8aa8c8",
            border: "1px solid rgba(255,255,255,0.08)",
          }}>
            {src}: {cnt}
          </span>
        ))}
      </div>

      {showDetail && (
        <div style={{
          marginTop: "10px",
          maxHeight: "160px", overflowY: "auto",
          display: "flex", flexDirection: "column", gap: "4px",
          borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: "8px",
          scrollbarWidth: "thin", scrollbarColor: "rgba(0,200,150,0.2) transparent",
        }}>
          {interactions.map((item, i) => {
            const ec = eventColor(item.event_type || item.source);
            return (
              <div key={i} style={{
                display: "flex", justifyContent: "space-between", alignItems: "center",
                padding: "5px 8px", borderRadius: "6px",
                background: "rgba(255,255,255,0.025)",
              }}>
                <span style={{
                  color: "#c8d8e8", fontSize: "11px", flex: 1,
                  marginRight: "8px", overflow: "hidden",
                  textOverflow: "ellipsis", whiteSpace: "nowrap",
                }}>
                  {item.product_name || item.name || `ID: ${item.product_id}`}
                </span>
                <span style={{
                  padding: "1px 7px", borderRadius: "4px",
                  fontSize: "10px", fontWeight: "600", whiteSpace: "nowrap",
                  background: ec.bg, color: ec.color,
                  border: `1px solid ${ec.border}`,
                }}>
                  {item.event_type || item.source || "view"}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

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
      <div style={S.header}>
        <div style={S.headerLeft}>
          <span style={{ fontSize: "22px" }}>🛒</span>
          <h1 style={S.headerTitle}>Ecommerce Recommendation</h1>
        </div>
        <button onClick={onLoginClick} style={S.btnPrimary}>Đăng nhập</button>
      </div>

      <div style={S.main}>
        <div style={S.tableCard}>
          <div style={{ ...S.tableHead, display: "flex", alignItems: "center", gap: "8px" }}>
            Top Sản Phẩm Được Mua Nhiều Nhất
          </div>
          {loading && <div style={{ textAlign: "center", padding: "40px", color: "#6b8aad" }}>⏳ Đang tải...</div>}
          {error && <div style={{ ...S.errorBox, margin: "16px" }}>{error}</div>}
          {!loading && !error && (
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={S.th}>#</th>
                  <th style={S.th}>Tên sản phẩm</th>
                  <th style={S.th}>Danh mục</th>
                  <th style={S.th}>Lượt mua</th>
                  <th style={S.th}>Rating</th>
                </tr>
              </thead>
              <tbody>
                {popular.map((item, idx) => (
                  <tr key={idx} style={{ background: idx % 2 === 0 ? "transparent" : "rgba(255,255,255,0.01)" }}>
                    <td style={S.td}><div style={S.rank}>{idx + 1}</div></td>
                    <td style={{ ...S.td, fontWeight: "500", color: "#c8d8e8" }}>{item.product_name}</td>
                    <td style={S.td}><span style={S.catBadge}>{item.category}</span></td>
                    <td style={{ ...S.td, color: "#00c896", fontWeight: "600" }}>{item.buy_count} lượt</td>
                    <td style={{ ...S.td, color: "#ffd700" }}>⭐ {item.avg_rating}</td>
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

function UserPage({ userId, onLogout }) {
  const [topN, setTopN]           = useState(10);
  const [modelType, setModelType] = useState("hybrid");
  const [loading, setLoading]     = useState(false);
  const [result, setResult]       = useState(null);
  const [error, setError]         = useState("");
  const [metrics, setMetrics]     = useState(null);

  useEffect(() => {
    fetch(`${API}/metrics`)
      .then(r => r.json())
      .then(d => setMetrics(d))
      .catch(() => {});
  }, []);

  useEffect(() => { fetchRecommend(); }, []);

  useEffect(() => {
    if (result) fetchRecommend();
  }, [modelType]);

  const fetchRecommend = async () => {
    setLoading(true); setError(""); setResult(null);
    try {
      const res  = await fetch(`${API}/recommend?user_id=${userId}&top_n=${topN}&model=${modelType}`);
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
      <div style={S.header}>
        <div style={S.headerLeft}>
          <span style={{ fontSize: "22px" }}>🛒</span>
          <h1 style={S.headerTitle}>Hệ Thống Gợi Ý Sản Phẩm</h1>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div style={S.userBadge}>{userId}</div>
          <button onClick={onLogout} style={S.logoutBtn}
            onMouseEnter={e => e.target.style.background = "rgba(255,100,100,0.08)"}
            onMouseLeave={e => e.target.style.background = "transparent"}
          >Đăng xuất</button>
        </div>
      </div>

      <div style={S.main}>

        <MetricsPanel />

        <div style={S.card}>
          <span style={S.label}>Cài đặt gợi ý</span>

          <ModelTabs active={modelType} onChange={m => setModelType(m)} />

          <div style={{ display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap" }}>
            <div style={{ position: "relative" }}>
              <select
                style={S.select}
                value={topN}
                onChange={e => setTopN(Number(e.target.value))}
              >
                {[5, 10, 15, 20].map(n => (
                  <option key={n} value={n} style={{ background: "#0d1b2a", color: "#e0e6f0" }}>
                    Top {n} sản phẩm
                  </option>
                ))}
              </select>
            </div>
            <button style={S.btnPrimary} onClick={fetchRecommend} disabled={loading}>
              <IconSearch />
              {loading ? "Đang tính..." : "Làm mới gợi ý"}
            </button>
          </div>
        </div>

        {error && <div style={S.errorBox}>{error}</div>}

        {loading && (
          <div style={{ textAlign: "center", padding: "60px", color: "#6b8aad" }}>
            <div style={{ fontSize: "32px", marginBottom: "12px" }}>⏳</div>
            Đang tính toán gợi ý cá nhân...
          </div>
        )}

        {result && !loading && (
          <>
            <div style={S.statsRow}>

              <div style={S.statCard}>
                <div style={S.statLbl}>User ID</div>
                <div style={S.statVal}>{result.user_id}</div>
              </div>

              <InteractionCard
                nInteractions={result.n_interactions}
                sourceCounts={result.source_counts}
                interactions={result.interactions || []}
              />

              <ModeCard mode={result.mode} metrics={metrics} />

            </div>

            <div style={S.tableCard}>
              <div style={S.tableHead}>
                Top {result.recommendations.length} Sản Phẩm Gợi Ý Cho Bạn
              </div>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr>
                    <th style={S.th}>#</th>
                    <th style={S.th}>Tên sản phẩm</th>
                    <th style={S.th}>Danh mục</th>
                    <th style={S.th}>Lý do gợi ý</th>
                    <th style={S.th}>Điểm phù hợp</th>
                  </tr>
                </thead>
                <tbody>
                  {result.recommendations.map((item, idx) => {
                    const scorePct = item.score > 0
                      ? parseFloat((item.score * 100).toFixed(2))
                      : 0;

                    return (
                      <tr key={idx} style={{ background: idx % 2 === 0 ? "transparent" : "rgba(255,255,255,0.01)" }}>
                        <td style={S.td}><div style={S.rank}>{item.rank}</div></td>
                        <td style={{ ...S.td, fontWeight: "500", color: "#c8d8e8" }}>{item.product_name}</td>
                        <td style={S.td}><span style={S.catBadge}>{item.category}</span></td>
                        <td style={S.td}><ReasonBadge reason={item.reason} /></td>
                        <td style={S.td}>
                          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                            <span style={{ color: "#00c896", fontWeight: "600", fontSize: "13px", minWidth: "56px" }}>
                              {item.score > 0 ? `${scorePct}%` : "—"}
                            </span>
                            {item.score > 0 && (
                              <div style={{ flex: 1, height: "6px", background: "rgba(255,255,255,0.06)", borderRadius: "3px" }}>
                                <div style={{
                                  height: "100%", width: `${Math.min(item.score * 100, 100)}%`,
                                  background: "linear-gradient(90deg, #00c896, #00a8ff)", borderRadius: "3px",
                                }} />
                              </div>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&display=swap');
        select option { background: #0d1b2a !important; color: #e0e6f0 !important; }
      `}</style>
    </div>
  );
}

export default function App() {
  const [page, setPage]     = useState("login");
  const [userId, setUserId] = useState("");

  const handleLogin  = (id) => { setUserId(id); setPage("user"); };
  const handleGuest  = ()   => setPage("guest");
  const handleLogout = ()   => { setUserId(""); setPage("login"); };

  if (page === "login") return <Login onLogin={handleLogin} onGuest={handleGuest} />;
  if (page === "guest") return <GuestPage onLoginClick={() => setPage("login")} />;
  return <UserPage userId={userId} onLogout={handleLogout} />;
}