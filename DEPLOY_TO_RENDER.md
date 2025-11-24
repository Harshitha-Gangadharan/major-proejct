# Deploying this Django app to Render

This document describes how to deploy the `CloudHealthCare` Django project to Render using a managed Postgres database.

1. Prepare repository

- Push your project to a Git provider (GitHub, GitLab, or Bitbucket) and connect it to Render.

2. Create a new Web Service on Render

- In Render dashboard, create a new **Web Service**.
- Connect it to your repository and select the branch (e.g., `main`).
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn $WSGI_MODULE:application --bind 0.0.0.0:$PORT`
- Set Environment Variables (in the Render dashboard or via `render.yaml`):
  - `WSGI_MODULE` = `Cloud.wsgi`
  - `SECRET_KEY` = (a strong secret string)
  - `DATABASE_URL` = (provided automatically when using the managed DB)
  - `ALLOWED_HOSTS` = `your-app.onrender.com`
  - `DEBUG` = `False`

3. Add a Managed Postgres Database

- Option A (recommended): Use the `render.yaml` provided in the repo. When you create the service with the Render CLI or dashboard, add the database in the same project. Render will provide a `DATABASE_URL` that you can use directly.
- Option B: In the Render dashboard go to **New -> PostgreSQL** and create a database. Copy the connection URL and set it to the `DATABASE_URL` environment variable in the web service.

4. Migrate and collect static files

You can run these commands in a Render shell (or run once from CI):

```powershell
# Run migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic --noinput
```

On first deploy, you may want to run the above manually from Render's web shell.

5. Notes and troubleshooting

- `Cloud/settings.py` reads `DATABASE_URL` via `dj-database-url`. Ensure `DATABASE_URL` uses Postgres syntax (postgres://...).
- `whitenoise` is configured to serve static files; `STATIC_ROOT` is set to `staticfiles`.
- If you need to run management commands automatically, add a `render.yaml` job or use the Render shell.
- If you require SSL to your DB set `DISABLE_DB_SSL=False` (default). For local development you can set it to `True`.

6. Environment variables summary

- `SECRET_KEY` — strong random value
- `DATABASE_URL` — Postgres connection URL (set by Render when using a managed DB)
- `ALLOWED_HOSTS` — comma-separated hosts (e.g., `your-app.onrender.com`)
- `DEBUG` — `False` in production
- `WSGI_MODULE` — `Cloud.wsgi`

If you want, I can also create a CI step or GitHub Actions workflow to run migrations and collectstatic on each deploy. 7) GitHub Actions (automatic migrations + deploy)

- I added a workflow at `.github/workflows/deploy-to-render.yml` that runs on `push` to `main`.
- What the workflow does:

  - installs Python and project dependencies
  - runs `python manage.py migrate --noinput`
  - runs `python manage.py collectstatic --noinput`
  - calls Render's Deploy API to trigger a deploy

- Required GitHub repository Secrets (configure in Settings → Secrets):

  - `DATABASE_URL` — full Postgres connection URL for the environment you want the migrations to run against (e.g., the Render managed DB URL). WARNING: This allows GitHub Actions to connect to your DB.
  - `SECRET_KEY` — your Django secret key
  - `ALLOWED_HOSTS` — comma-separated hosts
  - `DEBUG` — `False` in production
  - `RENDER_API_KEY` — a Render API key (create in Render Dashboard → Account → API Keys)
  - `RENDER_SERVICE_ID` — the Render Service ID for your Web Service (found in the service's URL or dashboard)

- Security note: Grant the Actions runner access only to the secrets it needs. If you prefer not to store DB credentials in GitHub, keep migrations manual and run them from Render's shell instead.
