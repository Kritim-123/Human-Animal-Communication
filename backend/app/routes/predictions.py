import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import crud, schemas
from app.database import get_db
from app.services.inference_service import ModelNotReadyError, predict_clip


router = APIRouter(tags=["predictions"])


@router.post("/predict/{clip_id}", response_model=schemas.PredictionRead)
def predict(clip_id: int, db: Session = Depends(get_db)):
    clip = crud.get_clip(db, clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="Clip not found.")

    try:
        result = predict_clip(clip.file_path)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    prediction = crud.create_prediction(
        db,
        clip=clip,
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        top_3_json=result["top_3_json"],
        model_version=result["model_version"],
    )
    return schemas.PredictionRead(
        id=prediction.id,
        clip_id=prediction.clip_id,
        predicted_label=prediction.predicted_label,
        confidence=prediction.confidence,
        top_3=json.loads(prediction.top_3_json),
        model_version=prediction.model_version,
        created_at=prediction.created_at,
        message=result["message"],
    )


@router.post("/confirm/{clip_id}", response_model=schemas.AudioClipRead)
def confirm(clip_id: int, confirmation: schemas.ConfirmPrediction, db: Session = Depends(get_db)):
    clip = crud.get_clip(db, clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="Clip not found.")
    return crud.confirm_prediction(db, clip=clip, confirmation=confirmation)


@router.get("/stats/{dog_id}", response_model=schemas.StatsRead)
def stats(dog_id: int, db: Session = Depends(get_db)):
    if crud.get_dog(db, dog_id) is None:
        raise HTTPException(status_code=404, detail="Dog not found.")
    return crud.get_stats(db, dog_id)

