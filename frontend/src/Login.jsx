import { useState, useEffect, useRef } from "react";

const API = "http://localhost:5000/api";

function StarCanvas() {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let animId;
    const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
    resize();
    window.addEventListener("resize", resize);
    const stars = Array.from({ length: 180 }, () => ({
      x: Math.random() * canvas.width, y: Math.random() * canvas.height,
      r: Math.random() * 1.5 + 0.3, twinkle: Math.random() * Math.PI * 2,
      speed: Math.random() * 0.008 + 0.002,
    }));
    const nebulas = [
      { x: 0.2, y: 0.3, r: 280, color: "rgba(72,0,180,0.12)" },
      { x: 0.75, y: 0.6, r: 320, color: "rgba(0,80,200,0.10)" },
      { x: 0.5, y: 0.85, r: 200, color: "rgba(0,180,150,0.08)" },
      { x: 0.1, y: 0.8, r: 180, color: "rgba(120,0,200,0.09)" },
    ];
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const bg = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
      bg.addColorStop(0, "#04061a"); bg.addColorStop(0.4, "#080d2a");
      bg.addColorStop(0.7, "#060e1f"); bg.addColorStop(1, "#030814");
      ctx.fillStyle = bg; ctx.fillRect(0, 0, canvas.width, canvas.height);
      nebulas.forEach(n => {
        const grad = ctx.createRadialGradient(n.x*canvas.width, n.y*canvas.height, 0, n.x*canvas.width, n.y*canvas.height, n.r);
        grad.addColorStop(0, n.color); grad.addColorStop(1, "transparent");
        ctx.fillStyle = grad; ctx.beginPath();
        ctx.arc(n.x*canvas.width, n.y*canvas.height, n.r, 0, Math.PI*2); ctx.fill();
      });
      stars.forEach(s => {
        s.twinkle += s.speed;
        const alpha = 0.4 + 0.6 * Math.abs(Math.sin(s.twinkle));
        ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, Math.PI*2);
        ctx.fillStyle = `rgba(200,220,255,${alpha})`; ctx.fill();
      });
      animId = requestAnimationFrame(draw);
    };
    draw();
    return () => { cancelAnimationFrame(animId); window.removeEventListener("resize", resize); };
  }, []);
  return <canvas ref={canvasRef} style={{ position:"fixed", inset:0, zIndex:0, display:"block" }} />;
}

export default function Login({ onLogin, onGuest }) {
  const [userId, setUserId] = useState("");
  const [focused, setFocused] = useState(false);
  const [shake, setShake] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loginError, setLoginError] = useState("");

  const triggerShake = () => { setShake(true); setTimeout(() => setShake(false), 500); };

  const handleLogin = async () => {
    if (!userId.trim()) {
      triggerShake();
      setLoginError("Vui lòng nhập User ID!");
      return;
    }
    setLoading(true);
    setLoginError("");
    try {
      const res = await fetch(`${API}/recommend?user_id=${userId.trim()}&top_n=1`);
      const data = await res.json();
      if (!res.ok) {
        setLoginError(data.error || "Không tìm thấy User ID này!");
        triggerShake();
      } else {
        onLogin(userId.trim());
      }
    } catch {
      setLoginError("Không kết nối được server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      position:"relative", minHeight:"100vh",
      display:"flex", alignItems:"center", justifyContent:"center",
      fontFamily:"'Sora','Segoe UI',sans-serif", overflow:"hidden",
    }}>
      <StarCanvas />
      <div style={{
        position:"relative", zIndex:1,
        width:"100%", maxWidth:"420px", margin:"24px",
        animation:"fadeUp 0.7s cubic-bezier(.22,1,.36,1) both",
      }}>
        <div style={{
          position:"absolute", inset:"-40px",
          background:"radial-gradient(ellipse, rgba(0,180,255,0.07) 0%, transparent 70%)",
          pointerEvents:"none",
        }} />

        <div style={{
          background:"rgba(8,14,35,0.75)", backdropFilter:"blur(24px)",
          border:"1px solid rgba(100,150,255,0.15)", borderRadius:"24px",
          padding:"48px 40px 40px",
          boxShadow:"0 0 60px rgba(0,100,255,0.08), 0 40px 80px rgba(0,0,0,0.5)",
        }}>
          <div style={{ textAlign:"center", marginBottom:"32px" }}>
            <div style={{
              width:"60px", height:"60px",
              background:"linear-gradient(135deg, #00c8ff, #7b2fff)",
              borderRadius:"18px", display:"flex", alignItems:"center",
              justifyContent:"center", fontSize:"28px",
              margin:"0 auto 16px", boxShadow:"0 8px 32px rgba(0,150,255,0.3)",
            }}>🛒</div>
            <h1 style={{
              margin:0, fontSize:"22px", fontWeight:"700",
              background:"linear-gradient(90deg, #a0c4ff, #ffffff, #c8a0ff)",
              WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent",
            }}>Hệ Thống Gợi Ý</h1>
            <p style={{ margin:"6px 0 0", fontSize:"13px", color:"rgba(160,180,220,0.6)" }}>
              Khám phá sản phẩm dành riêng cho bạn
            </p>
          </div>

          <div style={{ marginBottom:"12px", animation: shake ? "shake 0.4s ease" : "none" }}>
            <label style={{
              display:"block", fontSize:"12px", fontWeight:"600",
              color:"rgba(140,170,220,0.8)", letterSpacing:"0.8px",
              textTransform:"uppercase", marginBottom:"8px",
            }}>User ID</label>
            <input
              value={userId}
              onChange={e => { setUserId(e.target.value); setLoginError(""); }}
              onKeyDown={e => e.key === "Enter" && handleLogin()}
              onFocus={() => setFocused(true)}
              onBlur={() => setFocused(false)}
              placeholder="Nhập User ID (vd: U003023)"
              style={{
                width:"100%", boxSizing:"border-box",
                background:"rgba(255,255,255,0.04)",
                border:`1px solid ${loginError ? "rgba(255,80,80,0.5)" : focused ? "rgba(0,180,255,0.5)" : "rgba(100,130,200,0.2)"}`,
                borderRadius:"12px", padding:"14px 16px",
                color:"#e8f0ff", fontSize:"15px", outline:"none",
                fontFamily:"inherit",
                boxShadow: focused && !loginError ? "0 0 0 3px rgba(0,180,255,0.08)" : "none",
                transition:"border 0.2s, box-shadow 0.2s",
              }}
            />
          </div>

          {loginError && (
            <div style={{
              marginBottom:"12px", padding:"10px 14px",
              background:"rgba(255,80,80,0.08)",
              border:"1px solid rgba(255,80,80,0.25)",
              borderRadius:"8px", fontSize:"13px", color:"#ff6b6b",
            }}>{loginError}</div>
          )}

          <button onClick={handleLogin} disabled={loading} style={{
            width:"100%", padding:"14px",
            background: loading ? "rgba(0,150,255,0.3)" : "linear-gradient(135deg, #0090ff, #7b2fff)",
            border:"none", borderRadius:"12px",
            color:"#fff", fontSize:"15px", fontWeight:"600",
            cursor: loading ? "not-allowed" : "pointer",
            boxShadow:"0 4px 24px rgba(0,100,255,0.25)",
            marginBottom:"12px", fontFamily:"inherit", transition:"opacity 0.2s",
          }}>
            {loading ? "Đang kiểm tra..." : "Đăng nhập"}
          </button>

          <div style={{
            display:"flex", alignItems:"center", gap:"12px",
            margin:"16px 0", color:"rgba(120,140,180,0.4)", fontSize:"12px",
          }}>
            <div style={{ flex:1, height:"1px", background:"rgba(100,130,200,0.15)" }} />
            hoặc
            <div style={{ flex:1, height:"1px", background:"rgba(100,130,200,0.15)" }} />
          </div>

          <button onClick={onGuest} style={{
            width:"100%", padding:"13px", background:"transparent",
            border:"1px solid rgba(100,130,200,0.25)", borderRadius:"12px",
            color:"rgba(160,190,230,0.8)", fontSize:"14px",
            cursor:"pointer", fontFamily:"inherit", transition:"background 0.2s",
          }}
            onMouseEnter={e => e.target.style.background = "rgba(100,130,200,0.08)"}
            onMouseLeave={e => e.target.style.background = "transparent"}
          >
            Tiếp tục với tư cách khách
          </button>
        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&display=swap');
        @keyframes fadeUp { from{opacity:0;transform:translateY(30px)} to{opacity:1;transform:translateY(0)} }
        @keyframes shake {
          0%,100%{transform:translateX(0)} 20%{transform:translateX(-8px)}
          40%{transform:translateX(8px)} 60%{transform:translateX(-5px)} 80%{transform:translateX(5px)}
        }
        *{margin:0;padding:0;box-sizing:border-box;}
      `}</style>
    </div>
  );
}