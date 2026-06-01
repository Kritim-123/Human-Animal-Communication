from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.database import init_db
from app.routes import clips, dogs, predictions


app = FastAPI(
    title="DogBridge API",
    description="Personalized dog-human likely intent estimation API. It estimates possible intent; it does not translate dog language.",
    version="0.1.0",
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "name": "DogBridge",
        "message": "DogBridge estimates likely intent from dog audio and context; it does not translate dog language.",
    }


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            # Future version: accept small real-time audio chunks, buffer them,
            # extract streaming-safe features, and return rolling likely intent.
            await websocket.send_json(
                {
                    "type": "placeholder_prediction",
                    "received": payload,
                    "message": "Real-time audio inference is not implemented yet.",
                    "possible_intent": "unknown",
                    "confidence": 0.0,
                }
            )
    except WebSocketDisconnect:
        return


app.include_router(dogs.router)
app.include_router(clips.router)
app.include_router(predictions.router)

