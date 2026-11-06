"""Student Feedback Analysis System - Flask backend (SQLite locally, PostgreSQL on Render)."""
import os, io, re, csv, hashlib, sqlite3
from collections import Counter
from datetime import datetime, timedelta
from functools import wraps

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, session, render_template, Response

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback.db"))
CRITERIA = ["teaching", "clarity", "engagement", "materials", "punctuality"]

# ---------------------------------------------------------------- database
def run(sql, args=(), fetch=False):
    if DATABASE_URL:
        import psycopg2, psycopg2.extras
        con = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
        sql = sql.replace("?", "%s")
    else:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        cur.execute(sql, args)
        rows = [dict(r) for r in cur.fetchall()] if fetch else None
        con.commit()
        return rows
    finally:
        con.close()

def init_db():
    pk = "SERIAL PRIMARY KEY" if DATABASE_URL else "INTEGER PRIMARY KEY AUTOINCREMENT"
    run(f"CREATE TABLE IF NOT EXISTS courses (id {pk}, code TEXT UNIQUE NOT NULL, name TEXT NOT NULL, faculty TEXT NOT NULL)")
    run(f"""CREATE TABLE IF NOT EXISTS feedback (id {pk}, course_id INTEGER NOT NULL, student_hash TEXT NOT NULL,
        teaching INTEGER, clarity INTEGER, engagement INTEGER, materials INTEGER, punctuality INTEGER,
        comment TEXT, sentiment REAL, label TEXT, created_at TEXT, UNIQUE(course_id, student_hash))""")
    if not run("SELECT id FROM courses LIMIT 1", fetch=True):
        for c in [("CS101", "Data Structures", "Dr. Rao"), ("CS202", "Database Systems", "Prof. Mehta"),
                  ("MA105", "Linear Algebra", "Dr. Iyer"), ("EC310", "Digital Signal Processing", "Prof. Khan")]:
            run("INSERT INTO courses (code, name, faculty) VALUES (?, ?, ?)", c)

# --------------------------------------------------------------- sentiment
POS = set("""good great excellent amazing awesome clear helpful engaging interesting best love loved enjoy enjoyed enjoyable
useful friendly patient supportive inspiring organized organised knowledgeable fantastic wonderful informative easy
understandable perfect nice punctual motivating practical brilliant approachable""".split())
NEG = set("""bad poor boring confusing confused slow late worst hate hated unclear rude useless irrelevant disorganized
disorganised rushed outdated waste lazy unhelpful monotonous terrible awful dull stressful lacking unprepared weak
noisy difficult confusing""".split())
NEGATE = {"not", "no", "never", "isn't", "wasn't", "don't", "didn't", "can't", "hardly", "without", "isnt", "wasnt", "dont", "didnt", "cant"}
BOOST = {"very": 1.5, "really": 1.4, "extremely": 1.8, "too": 1.3, "so": 1.3, "highly": 1.5}
STOP = set("""the and for that this with was were are but you your have has had they them their from not very really more
much some what when where which will would could should about into than then also just class course sir madam teacher
faculty lecture lectures subject there been being because while does did dont get got feel always every often make made sometimes""".split())

def tokens(text):
    return re.findall(r"[a-z']+", (text or "").lower())

def analyze_sentiment(text):
    """Lexicon sentiment with negation + intensifiers. Returns (score -1..1, label) or (None, None) for no text."""
    toks = tokens(text)
    if not toks:
        return None, None
    score = 0.0
    for i, t in enumerate(toks):
        v = 1 if t in POS else -1 if t in NEG else 0
        if not v:
            continue
        window = toks[max(0, i - 3):i]
        if any(w in NEGATE for w in window):
            v *= -0.8
        mult = BOOST.get(window[-1], 1.0) if window else 1.0
        score += v * mult
    s = round(float(np.tanh(score / 1.5)), 3)
    return s, ("positive" if s > 0.15 else "negative" if s < -0.15 else "neutral")

# --------------------------------------------------------------- analytics
def load_df(course_id=None, faculty=None, label=None):
    sql = ("SELECT f.id, f.course_id, f.teaching, f.clarity, f.engagement, f.materials, f.punctuality, f.comment, "
           "f.sentiment, f.label, f.created_at, c.code, c.name AS course, c.faculty "
           "FROM feedback f JOIN courses c ON c.id = f.course_id WHERE 1=1")
    args = []
    if course_id:
        sql += " AND f.course_id = ?"; args.append(int(course_id))
    if faculty:
        sql += " AND c.faculty = ?"; args.append(faculty)
    if label:
        sql += " AND f.label = ?"; args.append(label)
    df = pd.DataFrame(run(sql + " ORDER BY f.created_at DESC", args, fetch=True))
    if df.empty:
        return df
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    df["overall"] = df[CRITERIA].astype(float).mean(axis=1)
    return df

def top_words(comments, n=12):
    c = Counter(w for t in comments for w in set(tokens(t)) if len(w) > 3 and w not in STOP)
    return [{"word": w, "count": k} for w, k in c.most_common(n)]

def safe_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5 or a.std() == 0 or b.std() == 0:
        return 0.0
    return round(float(np.corrcoef(a, b)[0, 1]), 2)

def analytics(df):
    if df.empty:
        return {"empty": True}
    n = len(df)
    crit = {c: round(float(df[c].mean()), 2) for c in CRITERIA}
    cm = df[df["label"].notna()]
    sent = {k: int((cm["label"] == k).sum()) for k in ("positive", "neutral", "negative")}
    avg_sent = round(float(cm["sentiment"].mean()), 2) if len(cm) else 0.0

    courses, alerts = [], []
    for _, g in df.groupby("course_id"):
        gs = g[g["sentiment"].notna()]
        avg = float(g["overall"].mean())
        s = float(gs["sentiment"].mean()) if len(gs) else None
        health = round(0.7 * avg / 5 * 100 + 0.3 * (s + 1) / 2 * 100, 1) if s is not None else round(avg / 5 * 100, 1)
        neg = float((gs["label"] == "negative").mean()) if len(gs) else 0.0
        weakest = min(CRITERIA, key=lambda c: g[c].mean())
        row = {"course_id": int(g["course_id"].iloc[0]), "code": g["code"].iloc[0], "course": g["course"].iloc[0],
               "faculty": g["faculty"].iloc[0], "responses": len(g), "avg": round(avg, 2),
               "sentiment": round(s, 2) if s is not None else None, "health": health, "weakest": weakest}
        courses.append(row)
        if len(g) >= 3 and (avg < 3.0 or neg >= 0.4):
            alerts.append(f"{row['code']} ({row['faculty']}) needs attention: avg {row['avg']}/5, "
                          f"{round(neg * 100)}% negative comments, weakest area is {weakest}.")
    courses.sort(key=lambda x: -x["health"])

    d = pd.to_datetime(df["created_at"])
    daily = df.assign(day=d.dt.strftime("%Y-%m-%d")).groupby("day")["overall"].agg(["mean", "count"]).reset_index()
    trend = [{"date": r["day"], "avg": round(float(r["mean"]), 2), "n": int(r["count"])} for _, r in daily.iterrows()]

    drivers = []
    for c in CRITERIA:
        others = (df[CRITERIA].astype(float).sum(axis=1) - df[c].astype(float)) / 4
        drivers.append({"criterion": c, "corr": safe_corr(df[c], others)})
    drivers.sort(key=lambda x: -x["corr"])

    best, worst = max(crit, key=crit.get), min(crit, key=crit.get)
    insights = [f"Strongest area: {best} ({crit[best]}/5). Weakest area: {worst} ({crit[worst]}/5)."]
    if drivers[0]["corr"] > 0:
        insights.append(f"'{drivers[0]['criterion']}' moves most with overall satisfaction (r={drivers[0]['corr']}), "
                        "so improving it is likely to lift ratings across the board.")
    if len(courses) > 1:
        insights.append(f"Healthiest course: {courses[0]['code']} ({courses[0]['health']}); "
                        f"lowest: {courses[-1]['code']} ({courses[-1]['health']}).")
    if len(cm) >= 10:
        cs = cm.assign(t=pd.to_datetime(cm["created_at"])).sort_values("t")
        h = len(cs) // 2
        diff = float(cs["sentiment"].iloc[h:].mean() - cs["sentiment"].iloc[:h].mean())
        if abs(diff) >= 0.05:
            insights.append(f"Comment sentiment is {'improving' if diff > 0 else 'declining'} "
                            f"({diff:+.2f} between the earlier and later half of responses).")
    if sent["positive"] + sent["negative"]:
        insights.append(f"{round(sent['positive'] / max(len(cm), 1) * 100)}% of written comments are positive.")

    return {"empty": False, "responses": n, "avg_overall": round(float(df["overall"].mean()), 2),
            "satisfaction": round(float((df["overall"] >= 4).mean() * 100), 1), "avg_sentiment": avg_sent,
            "criteria": crit, "sentiment": sent, "courses": courses, "trend": trend, "drivers": drivers,
            "keywords": {"positive": top_words(cm[cm["label"] == "positive"]["comment"]),
                         "negative": top_words(cm[cm["label"] == "negative"]["comment"])},
            "insights": insights, "alerts": alerts}

# ------------------------------------------------------------------- auth
def admin_required(f):
    @wraps(f)
    def wrapper(*a, **k):
        if not session.get("admin"):
            return jsonify(error="Admin login required"), 401
        return f(*a, **k)
    return wrapper

@app.post("/api/login")
def login():
    if (request.get_json(silent=True) or {}).get("password") == ADMIN_PASSWORD:
        session["admin"] = True
        return jsonify(ok=True)
    return jsonify(error="Wrong password"), 401

@app.post("/api/logout")
def logout():
    session.clear()
    return jsonify(ok=True)

@app.get("/api/me")
def me():
    return jsonify(admin=bool(session.get("admin")))

# ----------------------------------------------------------------- routes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
def index():
    for p in ("templates/index.html", "Templates/index.html", "index.html", "static/index.html"):
        f = os.path.join(BASE_DIR, p)
        if os.path.isfile(f):
            with open(f, encoding="utf-8") as fh:
                return Response(fh.read(), mimetype="text/html")
    return "index.html not found next to app.py", 500

@app.get("/health")
def health():
    return jsonify(status="ok")

@app.get("/api/courses")
def courses_list():
    return jsonify(run("SELECT id, code, name, faculty FROM courses ORDER BY code", fetch=True))

@app.post("/api/courses")
@admin_required
def course_add():
    d = request.get_json(silent=True) or {}
    code, name, fac = (str(d.get(k, "")).strip() for k in ("code", "name", "faculty"))
    if not (code and name and fac):
        return jsonify(error="Code, name and faculty are required"), 400
    if run("SELECT id FROM courses WHERE code = ?", (code.upper(),), fetch=True):
        return jsonify(error="That course code already exists"), 409
    run("INSERT INTO courses (code, name, faculty) VALUES (?, ?, ?)", (code.upper(), name, fac))
    return jsonify(ok=True), 201

@app.delete("/api/courses/<int:cid>")
@admin_required
def course_delete(cid):
    run("DELETE FROM feedback WHERE course_id = ?", (cid,))
    run("DELETE FROM courses WHERE id = ?", (cid,))
    return jsonify(ok=True)

@app.post("/api/sentiment")
def sentiment_preview():
    s, label = analyze_sentiment((request.get_json(silent=True) or {}).get("text", "")[:1000])
    return jsonify(score=s, label=label)

@app.post("/api/feedback")
def feedback_submit():
    d = request.get_json(silent=True) or {}
    roll = str(d.get("roll_no", "")).strip().lower()
    if len(roll) < 3:
        return jsonify(error="Enter a valid roll number (used only to stop duplicate submissions)"), 400
    try:
        cid = int(d.get("course_id"))
        ratings = {c: int(d.get(c)) for c in CRITERIA}
    except (TypeError, ValueError):
        return jsonify(error="Please choose a course and rate every category"), 400
    if not all(1 <= v <= 5 for v in ratings.values()):
        return jsonify(error="Ratings must be between 1 and 5"), 400
    if not run("SELECT id FROM courses WHERE id = ?", (cid,), fetch=True):
        return jsonify(error="Course not found"), 404
    h = hashlib.sha256((app.secret_key + roll).encode()).hexdigest()
    if run("SELECT id FROM feedback WHERE course_id = ? AND student_hash = ?", (cid, h), fetch=True):
        return jsonify(error="You have already submitted feedback for this course"), 409
    comment = str(d.get("comment", "")).strip()[:1000]
    s, label = analyze_sentiment(comment)
    run("INSERT INTO feedback (course_id, student_hash, teaching, clarity, engagement, materials, punctuality, "
        "comment, sentiment, label, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (cid, h, *[ratings[c] for c in CRITERIA], comment, s, label, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")))
    return jsonify(ok=True, sentiment=label), 201

@app.get("/api/analytics")
@admin_required
def analytics_api():
    df = load_df(request.args.get("course_id"), request.args.get("faculty"))
    return jsonify(analytics(df))

@app.get("/api/feedback")
@admin_required
def feedback_list():
    df = load_df(request.args.get("course_id"), request.args.get("faculty"), request.args.get("label"))
    if df.empty:
        return jsonify([])
    df = df.head(200).copy()
    df["overall"] = df["overall"].round(2)
    return jsonify(df.drop(columns=["course_id"]).astype(object).where(df.notna(), None).to_dict("records"))

@app.delete("/api/feedback")
@admin_required
def feedback_clear():
    run("DELETE FROM feedback")
    return jsonify(ok=True)

@app.get("/api/export.csv")
@admin_required
def export_csv():
    df = load_df(request.args.get("course_id"), request.args.get("faculty"))
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["date", "course_code", "course", "faculty", *CRITERIA, "overall", "sentiment", "label", "comment"])
    if not df.empty:
        for _, r in df.iterrows():
            w.writerow([r["created_at"], r["code"], r["course"], r["faculty"], *[r[c] for c in CRITERIA],
                        round(r["overall"], 2), "" if pd.isna(r["sentiment"]) else r["sentiment"], r["label"] or "", r["comment"] or ""])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=feedback_export.csv"})

POS_C = ["Explains concepts very clearly and the examples are helpful", "Great teaching, really engaging and well organized",
         "Very supportive and patient, the notes are useful", "Excellent course, I enjoyed every lecture",
         "Interesting classes and always punctual"]
NEG_C = ["The pace is too fast and the lectures feel confusing", "Study material is outdated and slides are boring",
         "Often late and not very clear while explaining", "Poor planning, the sessions feel rushed and disorganized"]
NEU_C = ["Average course, nothing special", "Okay overall", ""]

@app.post("/api/demo")
@admin_required
def demo():
    rng = np.random.default_rng()
    cs = run("SELECT id FROM courses", fetch=True)
    now, made = datetime.utcnow(), 0
    for c in cs:
        bias = rng.uniform(2.4, 4.6)
        for _ in range(int(rng.integers(12, 22))):
            base = bias + rng.normal(0, 0.6)
            r = {k: int(np.clip(round(base + rng.normal(0, 0.7)), 1, 5)) for k in CRITERIA}
            avg = sum(r.values()) / 5
            pool = POS_C if avg >= 3.8 else NEG_C if avg <= 2.7 else NEU_C
            comment = pool[int(rng.integers(len(pool)))]
            s, label = analyze_sentiment(comment)
            ts = (now - timedelta(days=int(rng.integers(0, 30)), hours=int(rng.integers(0, 24)))).strftime("%Y-%m-%d %H:%M:%S")
            run("INSERT INTO feedback (course_id, student_hash, teaching, clarity, engagement, materials, punctuality, "
                "comment, sentiment, label, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (c["id"], os.urandom(8).hex(), *[r[k] for k in CRITERIA], comment, s, label, ts))
            made += 1
    return jsonify(ok=True, added=made)

try:
    init_db()
except Exception as e:  # pragma: no cover
    print("DB init failed:", e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
