from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app import crud, schemas
from app.database import get_db
from app.services.storage import get_audio_duration_seconds, save_upload


router = APIRouter(tags=["clips"])


@router.post("/clips", response_model=schemas.AudioClipRead, status_code=status.HTTP_201_CREATED)
def upload_clip(
    dog_id: int = Form(...),
    location_context: schemas.LocationContext = Form(schemas.LocationContext.unknown),
    situation_context: schemas.SituationContext = Form(schemas.SituationContext.unknown),
    owner_label: schemas.IntentLabel = Form(schemas.IntentLabel.unknown),
    outcome_label: schemas.IntentLabel | None = Form(None),
    notes: str | None = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if crud.get_dog(db, dog_id) is None:
        raise HTTPException(status_code=404, detail="Dog not found.")
    try:
        saved_path = save_upload(file, dog_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    duration = get_audio_duration_seconds(saved_path)
    return crud.create_clip(
        db,
        dog_id=dog_id,
        file_path=str(saved_path),
        duration_seconds=duration,
        location_context=location_context.value,
        situation_context=situation_context.value,
        owner_label=owner_label.value,
        outcome_label=outcome_label.value if outcome_label else None,
        notes=notes,
    )


@router.get("/clips", response_model=list[schemas.AudioClipRead])
def list_clips(db: Session = Depends(get_db)):
    return crud.list_clips(db)


@router.get("/clips/{clip_id}", response_model=schemas.AudioClipRead)
def get_clip(clip_id: int, db: Session = Depends(get_db)):
    clip = crud.get_clip(db, clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="Clip not found.")
    return clip

