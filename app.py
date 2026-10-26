"""
Multimodal Construction Intelligence Platform
Skill Wallet Capstone Project 1

Streamlit + Google Gemini. No database. Deploys on Streamlit Community Cloud.
Run locally:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import re
import urllib.parse
from datetime import datetime

import requests
import streamlit as st
from PIL import Image

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:  # app still runs in demo mode
    GENAI_AVAILABLE = False

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
APP_TITLE = "Multimodal Construction Intelligence Platform"
MODEL_NAME = "gemini-2.5-flash"
MAX_IMAGE_SIDE = 1600
DISCLAIMER = (
    "AI-assisted output for screening and planning only. It does not replace inspection "
    "by a licensed structural engineer or certified safety officer."
)

st.set_page_config(page_title=APP_TITLE, page_icon="🏗️", layout="wide")

CSS = """
<style>
:root { --brand:#f59e0b; --ink:#0f172a; --muted:#64748b; --card:#ffffff; --line:#e2e8f0; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }
.hero { background: linear-gradient(120deg,#0f172a 0%,#1e3a5f 60%,#f59e0b 160%);
        border-radius: 16px; padding: 1.4rem 1.8rem; color: #fff; margin-bottom: 1rem; }
.hero h1 { margin:0; font-size:1.7rem; color:#fff; }
.hero p { margin:.3rem 0 0; opacity:.85; font-size:.95rem; }
.mcard { background:var(--card); border:1px solid var(--line); border-radius:14px;
         padding:1rem 1.1rem; box-shadow:0 1px 3px rgba(15,23,42,.06); height:100%; }
.mcard .label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.05em; }
.mcard .value { color:var(--ink); font-size:1.65rem; font-weight:700; line-height:1.25; }
.mcard .sub { color:var(--muted); font-size:.82rem; }
.pill { display:inline-block; padding:.18rem .7rem; border-radius:999px; font-size:.78rem; font-weight:600; }
.pill-green { background:#dcfce7; color:#166534; }
.pill-amber { background:#fef3c7; color:#92400e; }
.pill-orange{ background:#ffedd5; color:#9a3412; }
.pill-red   { background:#fee2e2; color:#991b1b; }
.pill-grey  { background:#e2e8f0; color:#334155; }
.hazard { border-left:5px solid var(--line); background:#f8fafc; border-radius:8px;
          padding:.7rem 1rem; margin-bottom:.6rem; color:#0f172a; }
.hazard.Low{border-color:#22c55e} .hazard.Medium{border-color:#f59e0b}
.hazard.High{border-color:#f97316} .hazard.Critical{border-color:#dc2626}
.alert { border-radius:12px; padding:1rem 1.2rem; margin-bottom:.8rem; border:1px solid; }
.alert.red{background:#fef2f2;border-color:#fecaca;color:#7f1d1d}
.alert.amber{background:#fffbeb;border-color:#fde68a;color:#78350f}
.alert.green{background:#f0fdf4;border-color:#bbf7d0;color:#14532d}
.demo-banner { background:#fff7ed; border:1px solid #fed7aa; color:#9a3412; padding:.5rem .9rem;
               border-radius:10px; font-size:.85rem; margin-bottom:.8rem; }
.footer-note { color:var(--muted); font-size:.78rem; margin-top:1.5rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
SYSTEM_INSTRUCTION = (
    "You are a senior construction safety officer and structural engineer working in India. "
    "Be precise, conservative about safety, and never fabricate details that are not visible. "
    "If something cannot be assessed from the image, say so."
)

SAFETY_PROMPT = """Analyze this construction site photo for worker PPE compliance and site hazards,
referencing OSHA 1926 (construction standards) and, where structural/concrete work is visible, IS 456:2000.

Return ONLY valid JSON with exactly this schema:
{
  "safety_score": <integer 0-100>,
  "overall_status": "<Compliant | Needs Attention | Non-Compliant | Critical>",
  "workers_detected": <integer>,
  "summary": "<2-3 sentence assessment>",
  "ppe_detection": {
    "helmet":          {"status": "<Detected | Missing | Not Visible>", "note": "<short>"},
    "high_vis_vest":   {"status": "<Detected | Missing | Not Visible>", "note": "<short>"},
    "steel_toe_boots": {"status": "<Detected | Missing | Not Visible>", "note": "<short>"},
    "safety_harness":  {"status": "<Detected | Missing | Not Visible | Not Required>", "note": "<short>"}
  },
  "hazards": [
    {"hazard": "<name>", "severity": "<Low | Medium | High | Critical>",
     "standard": "<e.g. OSHA 1926.501 / IS 456>", "remedy": "<corrective action>"}
  ]
}
If no workers are visible, set workers_detected to 0 and mark PPE items "Not Visible"."""

CRACK_PROMPT = """Analyze this concrete / masonry surface image for cracks and defects, referencing IS 456:2000
and IS 13935 (seismic strengthening/repair) where relevant.

Return ONLY valid JSON with exactly this schema:
{
  "defect_type": "<Hairline | Settlement | Structural Shear | Thermal Expansion | Spalling | No Significant Defect>",
  "confidence": <integer 0-100>,
  "estimated_crack_width_mm": <number>,
  "crack_pattern": "<e.g. diagonal, vertical, map cracking, horizontal>",
  "severity": "<Low | Medium | High | Critical>",
  "urgency": "<Monitor | Repair within 3 months | Repair within 2 weeks | Immediate action>",
  "structural_integrity_score": <integer 0-100>,
  "root_causes": ["<cause>", "..."],
  "recommended_repairs": [
    {"method": "<e.g. Epoxy injection | Structural re-grouting | Surface sealing | Section jacketing>",
     "detail": "<how and when to apply>"}
  ],
  "monitoring_advice": "<crack gauge / tell-tale guidance>",
  "engineer_review_required": <true | false>
}
Crack width is an estimate from visual cues; say so in monitoring_advice if no scale reference exists."""

# --------------------------------------------------------------------------- #
# Demo-mode data (used when no API key is available)
# --------------------------------------------------------------------------- #
DEMO_SAFETY = {
    "safety_score": 64,
    "overall_status": "Needs Attention",
    "workers_detected": 4,
    "summary": "DEMO DATA. Most workers wear helmets and vests, but two lack safety footwear and one "
               "elevated task shows no visible harness. Open edges and loose materials add risk.",
    "ppe_detection": {
        "helmet": {"status": "Detected", "note": "4 of 4 workers wearing hard hats"},
        "high_vis_vest": {"status": "Detected", "note": "3 of 4 workers; one missing"},
        "steel_toe_boots": {"status": "Missing", "note": "2 workers in regular shoes"},
        "safety_harness": {"status": "Missing", "note": "Worker at slab edge without fall arrest"},
    },
    "hazards": [
        {"hazard": "Unprotected slab edge", "severity": "Critical", "standard": "OSHA 1926.501(b)(1)",
         "remedy": "Install guardrails (top rail 42 in) or personal fall arrest before work continues."},
        {"hazard": "Missing safety footwear", "severity": "High", "standard": "OSHA 1926.96",
         "remedy": "Enforce steel-toe boot rule at site gate; stop work for non-compliant workers."},
        {"hazard": "Loose rebar and debris on walkway", "severity": "Medium", "standard": "OSHA 1926.25",
         "remedy": "Daily housekeeping; cap exposed rebar ends; keep a 1.2 m clear walkway."},
        {"hazard": "Unsecured scaffold planks", "severity": "Medium", "standard": "OSHA 1926.451",
         "remedy": "Fully plank platforms, secure ties, and tag scaffold after competent-person inspection."},
    ],
}

DEMO_CRACK = {
    "defect_type": "Structural Shear",
    "confidence": 78,
    "estimated_crack_width_mm": 2.4,
    "crack_pattern": "Diagonal (approx. 45°) near beam-column junction",
    "severity": "High",
    "urgency": "Repair within 2 weeks",
    "structural_integrity_score": 52,
    "root_causes": [
        "Inadequate shear reinforcement or stirrup spacing",
        "Overloading beyond design capacity",
        "Differential settlement transferring shear into the frame",
    ],
    "recommended_repairs": [
        {"method": "Epoxy injection", "detail": "Pressure-inject low-viscosity epoxy for cracks above 0.3 mm after cleaning and port installation."},
        {"method": "Structural re-grouting", "detail": "Use non-shrink cementitious grout around the damaged zone; consider jacketing if reinforcement is exposed."},
    ],
    "monitoring_advice": "DEMO DATA. Fix tell-tale gauges across the crack and log width weekly. Estimated width is visual only.",
    "engineer_review_required": True,
}

DEMO_CHAT = [
    (("crane", "lift", "wind"), "Suspend crane lifts when sustained wind exceeds ~32 km/h (20 mph) or per the manufacturer's limit. Lower booms, secure loose sheeting and formwork, and clear the swing radius."),
    (("cure", "curing", "concrete"), "Cure concrete for at least 7 days (10+ for blended cements) per IS 456 cl. 13.5. Use wet burlap or curing compound, and shade fresh pours in extreme heat."),
    (("rain", "monsoon", "wet"), "Cover fresh concrete with tarpaulin, pause pours in heavy rain, cover cement stock off the ground, and de-energize temporary electrical points."),
    (("harness", "height", "fall"), "Fall protection is required at 6 ft (1.8 m) under OSHA 1926.501. Use full-body harnesses anchored to rated points, plus guardrails on open edges."),
    (("steel", "rebar", "cover"), "Typical clear cover per IS 456 table 16: 20 mm slabs, 25 mm beams, 40 mm columns for mild exposure. Increase for moderate or severe exposure."),
    (("crack", "repair"), "Hairline (<0.3 mm): seal. 0.3-2 mm active or structural: epoxy injection. Wider shear cracks: get an engineer's assessment before repair."),
]

# --------------------------------------------------------------------------- #
# Weather advisories (Tab 4)
# --------------------------------------------------------------------------- #
WEATHER_ADVISORIES = {
    "Normal": {
        "tone": "green", "icon": "✅",
        "headline": "Normal Conditions: Standard Protocols",
        "alerts": [
            "Hold the daily toolbox talk and record attendance.",
            "Complete scaffold and lifting-gear checks before first use.",
            "Maintain hydration points and first-aid readiness.",
        ],
    },
    "High Wind": {
        "tone": "red", "icon": "💨",
        "headline": "High Wind Alert",
        "alerts": [
            "SUSPEND crane and hoist operations when sustained wind exceeds the manufacturer limit (typically ~32 km/h).",
            "Secure or remove loose sheeting, formwork panels, scaffold boards and stacked material.",
            "Stop work on exposed edges, tall scaffolds and roofs; evacuate the tower crane cab.",
            "Check scaffold ties and bracing after the wind event before re-opening.",
        ],
    },
    "Heavy Rain": {
        "tone": "red", "icon": "🌧️",
        "headline": "Heavy Rain Alert",
        "alerts": [
            "ELECTRICAL HAZARD: de-energize temporary supplies, use RCD/GFCI protection, keep cables off wet ground.",
            "Protect fresh concrete with tarpaulin; do not place concrete in heavy rain.",
            "Inspect excavations and trench shoring for water ingress; pump out before entry.",
            "Cover cement and keep it raised; slippery surfaces require anti-skid walkways.",
        ],
    },
    "Extreme Heat": {
        "tone": "amber", "icon": "🔥",
        "headline": "Extreme Heat Advisory",
        "alerts": [
            "Schedule heavy work for early morning; enforce shaded rest breaks and ORS/water every 20 minutes.",
            "CONCRETE CURING: pour at cooler hours, use chilled water/ice, shade surfaces and start curing immediately (IS 456 cl. 13.5).",
            "Watch for heat exhaustion symptoms: dizziness, cramps, confusion. Move affected workers to shade and cool them.",
            "Keep metal tools and formwork shaded to reduce burn risk.",
        ],
    },
}

# --------------------------------------------------------------------------- #
# Valuation model (Tab 3) — transparent planning assumptions, not market quotes
# --------------------------------------------------------------------------- #
STRUCTURE_PROFILES = {
    # cost INR/sqft (Tier 2 baseline), market INR/sqft factor, steel kg/sqft, cement 50 kg bags/sqft
    "Residential Apartment": {"cost": 2100, "market": 1.00, "steel": 4.0, "cement": 0.42},
    "Independent House / Villa": {"cost": 2600, "market": 1.15, "steel": 4.5, "cement": 0.45},
    "Commercial Building": {"cost": 2900, "market": 1.35, "steel": 5.5, "cement": 0.50},
    "Industrial Warehouse": {"cost": 1700, "market": 0.70, "steel": 7.0, "cement": 0.35},
}
TIER_PROFILES = {
    "Tier 1 (Metro: Mumbai, Delhi, Bengaluru)": {"cost": 1.25, "market_rate": 9500},
    "Tier 2 (Indore, Pune, Jaipur)": {"cost": 1.00, "market_rate": 5500},
    "Tier 3 (Smaller cities & towns)": {"cost": 0.82, "market_rate": 3200},
}
CONTINGENCY = 0.08
LAND_VALUE_SHARE = 0.35  # share of completed value attributed to land/location
MILESTONES = [
    (0, "Site prep & foundation"), (15, "Plinth & ground beam"), (30, "Superstructure (columns, slabs)"),
    (60, "Masonry, MEP & plaster"), (80, "Finishing & fixtures"), (95, "Snagging & handover"),
]


def fmt_inr(value: float) -> str:
    """Format an INR amount in Crores / Lakhs."""
    if value >= 1e7:
        return f"₹{value / 1e7:,.2f} Cr"
    if value >= 1e5:
        return f"₹{value / 1e5:,.2f} Lakh"
    return f"₹{value:,.0f}"


def compute_project(structure: str, sqft: float, tier: str, completion: int) -> dict:
    sp, tp = STRUCTURE_PROFILES[structure], TIER_PROFILES[tier]
    cost_rate = sp["cost"] * tp["cost"]
    construction = sqft * cost_rate
    budget = construction * (1 + CONTINGENCY)
    frac = completion / 100
    capex = budget * frac
    full_value = sqft * tp["market_rate"] * sp["market"]
    # land share is realised from day one; built share accrues with completion
    asset_value = full_value * (LAND_VALUE_SHARE + (1 - LAND_VALUE_SHARE) * frac)
    steel_total = sqft * sp["steel"] / 1000  # tonnes
    cement_total = sqft * sp["cement"]  # bags
    return {
        "cost_rate": cost_rate, "budget": budget, "capex": capex, "remaining": budget - capex,
        "full_value": full_value, "asset_value": asset_value,
        "steel_total": steel_total, "steel_used": steel_total * frac,
        "cement_total": cement_total, "cement_used": cement_total * frac,
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def resolve_api_key(sidebar_key: str) -> str:
    """Sidebar input wins; fall back to Streamlit secrets; else empty (demo mode)."""
    if sidebar_key and sidebar_key.strip():
        return sidebar_key.strip()
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


def prepare_image(uploaded) -> tuple[bytes, Image.Image]:
    """Validate, orient, downscale and re-encode an upload as JPEG bytes."""
    img = Image.open(uploaded)
    img.load()
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    img = img.convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), img


def parse_json(text: str) -> dict:
    """Parse model output as JSON, tolerating code fences or stray prose."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def friendly_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if "api key" in low or "api_key" in low or "401" in low or "403" in low or "permission" in low:
        return "The API key was rejected. Check the key in the sidebar."
    if "429" in low or "quota" in low or "rate" in low:
        return "Rate limit or quota reached. Wait a minute and retry, or use demo mode."
    if "timeout" in low or "connection" in low:
        return "Network problem while contacting Gemini. Please retry."
    return f"Analysis failed: {msg[:200]}"


def analyze_image(api_key: str, prompt: str, image_bytes: bytes) -> dict:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), prompt],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    if not response.text:
        raise ValueError("Empty response from model (possibly blocked by safety filters).")
    return parse_json(response.text)


def run_analysis(state_key: str, prompt: str, demo: dict, uploaded, api_key: str, live: bool):
    """Run analysis (live or demo) and store result in session state."""
    try:
        image_bytes, _ = prepare_image(uploaded)
    except Exception as exc:
        st.error(f"Could not read that image: {exc}")
        return
    if not live:
        st.session_state[state_key] = {"data": demo, "demo": True, "time": datetime.now()}
        return
    try:
        with st.spinner("Analyzing image with Gemini…"):
            data = analyze_image(api_key, prompt, image_bytes)
        st.session_state[state_key] = {"data": data, "demo": False, "time": datetime.now()}
    except Exception as exc:
        st.error(friendly_error(exc))


def metric_card(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="mcard"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>')


def pill(text: str, kind: str) -> str:
    return f'<span class="pill pill-{kind}">{text}</span>'


SEVERITY_PILL = {"Low": "green", "Medium": "amber", "High": "orange", "Critical": "red"}
STATUS_PILL = {"Compliant": "green", "Needs Attention": "amber", "Non-Compliant": "orange", "Critical": "red"}
PPE_PILL = {"Detected": "green", "Missing": "red", "Not Visible": "grey", "Not Required": "grey"}
PPE_LABELS = {
    "helmet": "⛑️ Helmet / Hard hat", "high_vis_vest": "🦺 High-vis vest",
    "steel_toe_boots": "🥾 Steel-toe boots", "safety_harness": "🪢 Safety harness",
}


def to_int(value, default=0, lo=0, hi=100) -> int:
    try:
        return max(lo, min(hi, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def score_color(score: int) -> str:
    return "green" if score >= 80 else "amber" if score >= 60 else "orange" if score >= 40 else "red"


def demo_banner(is_demo: bool):
    if is_demo:
        st.markdown('<div class="demo-banner">🧪 <b>Demonstration mode</b>: showing sample output, '
                    'not an analysis of your image. Add a Gemini API key in the sidebar for live results.</div>',
                    unsafe_allow_html=True)


def download_report(label: str, payload: dict, filename: str):
    st.download_button(label, data=json.dumps(payload, indent=2, ensure_ascii=False),
                       file_name=f"{filename}_{datetime.now():%Y%m%d_%H%M%S}.json",
                       mime="application/json")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 🏗️ Control Panel")
    key_input = st.text_input("Google Gemini API key", type="password",
                              placeholder="Paste key (kept in session only)",
                              help="Get a free key at aistudio.google.com. Leave empty for demo mode.")
    API_KEY = resolve_api_key(key_input)
    LIVE = bool(API_KEY) and GENAI_AVAILABLE

    if LIVE:
        st.success("Live mode: Gemini connected")
    elif API_KEY and not GENAI_AVAILABLE:
        st.error("`google-genai` is not installed. Running in demo mode.")
    else:
        st.warning("Demo mode: no API key supplied")

    st.markdown("### Features")
    st.markdown(
        "- 🦺 **Tab 1:** Site Safety & PPE Audit\n"
        "- 🧱 **Tab 2:** Crack Diagnostics\n"
        "- 💰 **Tab 3:** Planning & Valuation\n"
        "- 🌦️ **Tab 4:** Daily Site Assistant"
    )

    with st.expander("Architecture overview"):
        st.markdown(
            f"""
**Flow:** Upload → PIL preprocess (EXIF fix, resize ≤{MAX_IMAGE_SIDE}px) → Gemini (`{MODEL_NAME}`)
with a JSON-only schema → validated render → JSON export.

**Layers**
- *UI:* Streamlit tabs, custom CSS
- *AI:* `google-genai` multimodal + chat
- *Domain logic:* rule-based cost & valuation model
- *State:* `st.session_state` only (no database)
- *Render:* Pollinations AI image URL

**Standards referenced:** OSHA 1926, IS 456:2000, IS 13935
"""
        )

    if st.button("Clear all results", use_container_width=True):
        for k in ("safety", "crack", "chat", "render_url", "render_prompt"):
            st.session_state.pop(k, None)
        st.rerun()
    st.caption(DISCLAIMER)

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    f'<div class="hero"><h1>🏗️ {APP_TITLE}</h1>'
    '<p>AI-powered site safety, structural diagnostics, project valuation and daily advisory. '
    'Skill Wallet Capstone Project 1.</p></div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🦺 Site Safety & PPE Audit", "🧱 Crack Diagnostics",
    "💰 Planning & Valuation", "🌦️ Daily Site Assistant",
])

# --------------------------------------------------------------------------- #
# Tab 1: Safety & PPE
# --------------------------------------------------------------------------- #
with tab1:
    st.subheader("Site Safety & PPE Compliance Audit")
    st.caption("Upload a site photo. The model checks PPE and flags hazards against OSHA 1926 and IS 456.")
    up1 = st.file_uploader("Construction site photo", type=["jpg", "jpeg", "png"], key="up_safety")

    if up1:
        c_img, c_btn = st.columns([2, 1])
        with c_img:
            st.image(up1, caption="Uploaded site photo", use_container_width=True)
        with c_btn:
            if st.button("🔍 Run Safety Audit", type="primary", use_container_width=True):
                run_analysis("safety", SAFETY_PROMPT, DEMO_SAFETY, up1, API_KEY, LIVE)
    else:
        st.info("Upload a JPG or PNG to begin. In demo mode you can upload any image to preview the report format.")

    res = st.session_state.get("safety")
    if res:
        d = res["data"]
        demo_banner(res["demo"])
        score = to_int(d.get("safety_score"))
        status = str(d.get("overall_status", "Unknown"))
        hazards = d.get("hazards") or []
        crit = sum(1 for h in hazards if h.get("severity") in ("High", "Critical"))

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(metric_card("Safety score", f"{score}/100", pill(status, STATUS_PILL.get(status, "grey"))),
                    unsafe_allow_html=True)
        m2.markdown(metric_card("Workers detected", str(d.get("workers_detected", "n/a")), "in frame"),
                    unsafe_allow_html=True)
        m3.markdown(metric_card("Hazards found", str(len(hazards)), "total"), unsafe_allow_html=True)
        m4.markdown(metric_card("High / Critical", str(crit), "need priority action"), unsafe_allow_html=True)

        st.progress(score / 100, text=f"Compliance level: {score}%")
        if d.get("summary"):
            st.markdown(f"**Assessment:** {d['summary']}")

        st.markdown("#### PPE checklist")
        ppe = d.get("ppe_detection") or {}
        cols = st.columns(4)
        for col, (key, label) in zip(cols, PPE_LABELS.items()):
            item = ppe.get(key) or {}
            st_txt = str(item.get("status", "Not Visible"))
            col.markdown(metric_card(label, pill(st_txt, PPE_PILL.get(st_txt, "grey")), item.get("note", "")),
                         unsafe_allow_html=True)

        st.markdown("#### Identified hazards")
        if not hazards:
            st.success("No hazards identified in this image.")
        order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        for h in sorted(hazards, key=lambda x: order.get(x.get("severity"), 9)):
            sev = h.get("severity", "Low")
            st.markdown(
                f'<div class="hazard {sev}"><b>{h.get("hazard", "Hazard")}</b> '
                f'{pill(sev, SEVERITY_PILL.get(sev, "grey"))} '
                f'<span style="color:#64748b;font-size:.8rem">{h.get("standard", "")}</span><br>'
                f'<span>🛠️ <b>Remedy:</b> {h.get("remedy", "n/a")}</span></div>',
                unsafe_allow_html=True)

        download_report("⬇️ Download JSON report", {
            "report": "site_safety_ppe_audit", "generated_at": res["time"].isoformat(),
            "mode": "demo" if res["demo"] else "live", "model": None if res["demo"] else MODEL_NAME,
            "result": d, "disclaimer": DISCLAIMER}, "safety_audit")

# --------------------------------------------------------------------------- #
# Tab 2: Crack Diagnostics
# --------------------------------------------------------------------------- #
with tab2:
    st.subheader("Crack Diagnostics & Structural Integrity")
    st.caption("Upload a close-up of a concrete or masonry surface. Include a coin or ruler for scale if possible.")
    up2 = st.file_uploader("Concrete / wall surface photo", type=["jpg", "jpeg", "png"], key="up_crack")

    if up2:
        c_img, c_btn = st.columns([2, 1])
        with c_img:
            st.image(up2, caption="Uploaded surface", use_container_width=True)
        with c_btn:
            if st.button("🔬 Diagnose Defect", type="primary", use_container_width=True):
                run_analysis("crack", CRACK_PROMPT, DEMO_CRACK, up2, API_KEY, LIVE)
    else:
        st.info("Upload a JPG or PNG of the affected surface.")

    res = st.session_state.get("crack")
    if res:
        d = res["data"]
        demo_banner(res["demo"])
        sev = str(d.get("severity", "Low"))
        integrity = to_int(d.get("structural_integrity_score"))
        try:
            width = f'{float(d.get("estimated_crack_width_mm", 0)):.1f} mm'
        except (TypeError, ValueError):
            width = "n/a"

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(metric_card("Defect type", str(d.get("defect_type", "Unknown")),
                                f'Confidence {to_int(d.get("confidence"))}%'), unsafe_allow_html=True)
        m2.markdown(metric_card("Est. crack width", width, "visual estimate"), unsafe_allow_html=True)
        m3.markdown(metric_card("Severity", pill(sev, SEVERITY_PILL.get(sev, "grey")),
                                str(d.get("crack_pattern", ""))), unsafe_allow_html=True)
        m4.markdown(metric_card("Urgency", str(d.get("urgency", "n/a")), ""), unsafe_allow_html=True)

        st.progress(integrity / 100, text=f"Structural integrity score: {integrity}/100")
        if d.get("engineer_review_required"):
            st.error("⚠️ Licensed structural engineer review is required before further loading or repair.")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Probable root causes")
            for cause in d.get("root_causes") or []:
                st.markdown(f"- {cause}")
        with right:
            st.markdown("#### Recommended engineering repairs")
            for r in d.get("recommended_repairs") or []:
                if isinstance(r, dict):
                    st.markdown(f"**{r.get('method', 'Repair')}**: {r.get('detail', '')}")
                else:
                    st.markdown(f"- {r}")
        if d.get("monitoring_advice"):
            st.info(f"📏 **Monitoring:** {d['monitoring_advice']}")

        download_report("⬇️ Download JSON report", {
            "report": "crack_diagnostics", "generated_at": res["time"].isoformat(),
            "mode": "demo" if res["demo"] else "live", "model": None if res["demo"] else MODEL_NAME,
            "result": d, "disclaimer": DISCLAIMER}, "crack_diagnostics")

# --------------------------------------------------------------------------- #
# Tab 3: Planning & Valuation
# --------------------------------------------------------------------------- #
with tab3:
    st.subheader("Project Planning & Real Estate Valuation")
    st.caption("Indicative estimates from planning-level cost assumptions. Not a certified valuation.")

    i1, i2 = st.columns(2)
    with i1:
        structure = st.selectbox("Structure type", list(STRUCTURE_PROFILES))
        sqft = st.number_input("Built-up area (sq ft)", min_value=100, max_value=1_000_000,
                               value=2000, step=100)
    with i2:
        tier = st.selectbox("Location tier", list(TIER_PROFILES), index=1)
        completion = st.slider("Completion (%)", 0, 100, 45)

    p = compute_project(structure, float(sqft), tier, completion)

    a, b, c, d_ = st.columns(4)
    a.markdown(metric_card("Total project budget", fmt_inr(p["budget"]),
                           f"₹{p['cost_rate']:,.0f}/sqft + {int(CONTINGENCY * 100)}% contingency"),
               unsafe_allow_html=True)
    b.markdown(metric_card("Capex incurred", fmt_inr(p["capex"]), f"{fmt_inr(p['remaining'])} remaining"),
               unsafe_allow_html=True)
    c.markdown(metric_card("Current asset valuation", fmt_inr(p["asset_value"]),
                           f"Completed value {fmt_inr(p['full_value'])}"), unsafe_allow_html=True)
    gain = p["asset_value"] - p["capex"]
    d_.markdown(metric_card("Value vs. spend", fmt_inr(abs(gain)),
                            "above capex" if gain >= 0 else "below capex"), unsafe_allow_html=True)

    st.markdown("#### Material requirements")
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(metric_card("Steel required", f"{p['steel_total']:,.1f} t", "total"), unsafe_allow_html=True)
    s2.markdown(metric_card("Steel consumed", f"{p['steel_used']:,.1f} t",
                            f"{p['steel_total'] - p['steel_used']:,.1f} t to procure"), unsafe_allow_html=True)
    s3.markdown(metric_card("Cement required", f"{p['cement_total']:,.0f} bags", "50 kg bags"),
                unsafe_allow_html=True)
    s4.markdown(metric_card("Cement consumed", f"{p['cement_used']:,.0f} bags",
                            f"{p['cement_total'] - p['cement_used']:,.0f} to procure"), unsafe_allow_html=True)

    st.markdown("#### Milestone progress")
    st.progress(completion / 100, text=f"{completion}% complete")
    ms_cols = st.columns(len(MILESTONES))
    for idx, (col, (start, name)) in enumerate(zip(ms_cols, MILESTONES)):
        end = MILESTONES[idx + 1][0] if idx + 1 < len(MILESTONES) else 100
        if completion >= end:
            badge = pill("Done", "green")
        elif completion >= start:
            badge = pill("In progress", "amber")
        else:
            badge = pill("Upcoming", "grey")
        col.markdown(f"**{name}**<br>{badge}", unsafe_allow_html=True)

    with st.expander("Model assumptions"):
        st.markdown(
            f"- Base cost: ₹{STRUCTURE_PROFILES[structure]['cost']:,}/sqft (Tier 2) × tier factor "
            f"{TIER_PROFILES[tier]['cost']}\n"
            f"- Contingency and approvals: {int(CONTINGENCY * 100)}%\n"
            f"- Completed market value: ₹{TIER_PROFILES[tier]['market_rate']:,}/sqft × structure factor "
            f"{STRUCTURE_PROFILES[structure]['market']}\n"
            f"- Under-construction value = completed value × ({int(LAND_VALUE_SHARE * 100)}% land share + "
            f"{int((1 - LAND_VALUE_SHARE) * 100)}% × completion)\n"
            f"- Steel {STRUCTURE_PROFILES[structure]['steel']} kg/sqft, cement "
            f"{STRUCTURE_PROFILES[structure]['cement']} bags/sqft (typical thumb rules)\n"
            "- Replace constants in `STRUCTURE_PROFILES` / `TIER_PROFILES` with local rates for accuracy."
        )

    st.markdown("#### 🎨 Conceptual 3D architectural rendering")
    style = st.selectbox("Style", ["Modern minimalist", "Contemporary Indian", "Glass facade commercial",
                                   "Industrial steel & concrete"])
    default_prompt = (
        f"Photorealistic 3D architectural render of a {structure.lower()} in an Indian {tier.split(' (')[0]} city, "
        f"{style.lower()} style, about {sqft:,} sq ft, {completion}% constructed, "
        "golden hour lighting, detailed facade, landscaped surroundings, ultra detailed"
    )
    render_prompt = st.text_area("Image prompt (editable)", value=default_prompt, height=90)

    if st.button("🖼️ Generate concept render", type="primary"):
        url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(render_prompt)
               + "?width=1024&height=640&nologo=true")
        try:
            with st.spinner("Rendering concept…"):
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                Image.open(io.BytesIO(resp.content)).verify()
            st.session_state["render_url"] = (url, resp.content)
        except Exception:
            st.session_state.pop("render_url", None)
            st.warning("Render service is busy or unreachable. Try again, or open the link directly:")
            st.markdown(f"[Open in Pollinations]({url})")
    if st.session_state.get("render_url"):
        _, img_bytes = st.session_state["render_url"]
        st.image(img_bytes, caption="Conceptual render (illustrative only)", use_container_width=True)
        st.download_button("⬇️ Download render", img_bytes, file_name="concept_render.jpg", mime="image/jpeg")

# --------------------------------------------------------------------------- #
# Tab 4: Daily Site Assistant
# --------------------------------------------------------------------------- #
def demo_chat_reply(question: str) -> str:
    q = question.lower()
    for keywords, answer in DEMO_CHAT:
        if any(k in q for k in keywords):
            return answer + "\n\n*(Demo mode: add a Gemini key for full conversational answers.)*"
    return ("I'm in demo mode with a small built-in knowledge base (wind, rain, curing, fall protection, "
            "rebar cover, cracks). Add a Gemini API key in the sidebar for open-ended engineering Q&A.")


def live_chat_reply(api_key: str, history: list[dict], weather: str) -> str:
    client = genai.Client(api_key=api_key)
    contents = [
        types.Content(role="user" if m["role"] == "user" else "model",
                      parts=[types.Part.from_text(text=m["content"])])
        for m in history
    ]
    system = (
        "You are a practical construction engineering assistant for Indian building sites. "
        "Answer concisely (under 180 words unless asked for detail), cite relevant IS codes "
        "(IS 456, IS 800, IS 1893, NBC 2016) or OSHA 1926 where applicable, and prioritize safety. "
        f"Today's site weather condition: {weather}. Factor it into advice when relevant. "
        "If a question needs a licensed engineer's judgment, say so."
    )
    response = client.models.generate_content(
        model=MODEL_NAME, contents=contents,
        config=types.GenerateContentConfig(system_instruction=system, temperature=0.4),
    )
    return response.text or "I couldn't generate a response. Please rephrase and try again."


with tab4:
    st.subheader("Daily Site Assistant & Weather-Aware Advisory")
    weather = st.radio("Today's site weather", list(WEATHER_ADVISORIES), horizontal=True,
                       format_func=lambda w: f"{WEATHER_ADVISORIES[w]['icon']} {w}")
    adv = WEATHER_ADVISORIES[weather]
    items = "".join(f"<li>{a}</li>" for a in adv["alerts"])
    st.markdown(f'<div class="alert {adv["tone"]}"><b>{adv["icon"]} {adv["headline"]}</b>'
                f'<ul style="margin:.5rem 0 0 1rem">{items}</ul></div>', unsafe_allow_html=True)

    st.markdown("#### 💬 Engineering assistant")
    chat = st.session_state.setdefault("chat", [])
    if not chat:
        st.caption("Try: “Can we pour a slab this afternoon?”, “What cover do columns need?”, "
                   "“When is a harness mandatory?”")
    for msg in chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a construction engineering question…"):
        chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            if LIVE:
                try:
                    with st.spinner("Thinking…"):
                        reply = live_chat_reply(API_KEY, chat[-20:], weather)
                except Exception as exc:
                    reply = f"⚠️ {friendly_error(exc)}"
            else:
                reply = demo_chat_reply(prompt)
            st.markdown(reply)
        chat.append({"role": "assistant", "content": reply})

st.markdown(f'<div class="footer-note">{DISCLAIMER}</div>', unsafe_allow_html=True)
