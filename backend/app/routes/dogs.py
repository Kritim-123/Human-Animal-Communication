from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import crud, schemas
from app.database import get_db


router = APIRouter(prefix="/dogs", tags=["dogs"])


@router.post("", response_model=schemas.DogRead, status_code=status.HTTP_201_CREATED)
def create_dog(dog: schemas.DogCreate, db: Session = Depends(get_db)):
    return crud.create_dog(db, dog)


@router.get("", response_model=list[schemas.DogRead])
def list_dogs(db: Session = Depends(get_db)):
    return crud.list_dogs(db)


@router.get("/{dog_id}", response_model=schemas.DogRead)
def get_dog(dog_id: int, db: Session = Depends(get_db)):
    dog = crud.get_dog(db, dog_id)
    if dog is None:
        raise HTTPException(status_code=404, detail="Dog not found.")
    return dog

