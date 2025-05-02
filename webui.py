import uuid

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

SCRAPE_RESULTS = {}  # job_id: {"status": "pending"/"done"/"error", "result_path": str}

def run_scrape_job(url: str, job_id: str):
    from RAGnificent.core.scraper import MarkdownScraper  # adjust import as needed
    try:
        scraper = MarkdownScraper()
        content = scraper.scrape_website(url)
        out_path = f"static/result_{job_id}.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        SCRAPE_RESULTS[job_id] = {"status": "done", "result_path": out_path}
    except Exception as e:
        SCRAPE_RESULTS[job_id] = {"status": "error", "error": str(e)}

@app.post("/scrape")
async def scrape(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    url = data.get("url")
    if not url:
        return JSONResponse({"error": "Missing URL"}, status_code=400)
    job_id = str(uuid.uuid4())
    SCRAPE_RESULTS[job_id] = {"status": "pending"}
    background_tasks.add_task(run_scrape_job, url, job_id)
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def status(job_id: str):
    if info := SCRAPE_RESULTS.get(job_id):
        return info
    else:
        return JSONResponse({"error": "Job not found"}, status_code=404)

@app.get("/result/{job_id}")
async def result(job_id: str):
    info = SCRAPE_RESULTS.get(job_id)
    if not info or info.get("status") != "done":
        return JSONResponse({"error": "Result not ready"}, status_code=404)
    return FileResponse(info["result_path"], filename=f"result_{job_id}.md")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()
