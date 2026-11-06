# Student Feedback Analysis System

Flask + pandas + numpy backend, single-page frontend, SQLite locally / PostgreSQL on Render.

## Run locally
```
pip install -r requirements.txt
python app.py          # http://localhost:5000
```
Admin password: `admin123` (override with the `ADMIN_PASSWORD` env var). Open **Admin Dashboard**, log in, click **Load demo data**.

## Features
- Anonymous student form (roll number is salted+hashed; one response per course per student)
- Live tone detection while typing a comment (negation- and intensifier-aware sentiment engine)
- Dashboard: rating by category, sentiment split, rating trend, praise/criticism keywords
- Satisfaction drivers (numpy correlation), auto-generated insights, low-score alerts
- Course/faculty leaderboard with Health Score (70% ratings + 30% sentiment)
- Filters by course and faculty, CSV export, course management, demo data

## Deploy on Render
1. Push this folder to GitHub.
2. Render > New > Blueprint > pick the repo (uses `render.yaml`: web service + free PostgreSQL).
3. Set `ADMIN_PASSWORD` when prompted. `SECRET_KEY` and `DATABASE_URL` are filled in automatically.
