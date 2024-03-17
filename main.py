from fastapi import FastAPI, Request, Depends
from streaming import stream
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


@app.get('/myvideo_feed')
async def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return StreamingResponse(stream.gen_frames('object'), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get('/closest_video')
async def closest_video():
    # Video streaming route. Put this in the src attribute of an img tag
    return StreamingResponse(stream.gen_frames('closest_objest'), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get('/video_for_blind')
async def video_for_blind():
    # Video streaming route. Put this in the src attribute of an img tag
    return StreamingResponse(stream.gen_frames('blind'), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get('/video_for_depth')
async def video_for_depth():
    # Video streaming route. Put this in the src attribute of an img tag
    return StreamingResponse(stream.gen_frames('depth'), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/object-detection/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("object_detection.html", {"request": request, "dtype": "object"})

@app.get("/closest-object-detection/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("object_detection.html", {"request": request, "dtype": "closest_object"})

@app.get("/blind/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("object_detection.html", {"request": request, "dtype": "blind"})

@app.get("/depth/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("object_detection.html", {"request": request, "dtype": "depth"})


@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
