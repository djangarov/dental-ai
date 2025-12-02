from typing import Annotated
from fastapi import FastAPI, Path, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/{item_id}", response_class=HTMLResponse)
async def read_item(
    request: Request,
    item_id: Annotated[int, Path(title="The ID of the item to get")]
) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request, name="test.html", context={"id": item_id}
    )