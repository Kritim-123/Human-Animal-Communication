# Public Dog Audio Data

DogBridge can use public dog vocalization datasets to learn general canine audio representations. Public data should not be treated as direct evidence for owner-intent labels like `outside_bathroom`, `food_water`, or `attention` unless the dataset includes reliable context and outcome labels.

The main DogBridge intent model should remain personalized and trained on owner-confirmed clips collected through the app.

## Recommended Use

Use public data for:

- Dog sound detection.
- Bark/audio embedding pretraining.
- Individual dog recognition experiments.
- Breed or sex auxiliary tasks when metadata exists.
- Robustness to noisy real-world recordings.

Do not use public data to claim:

- Literal dog-language translation.
- Universal intent prediction.
- Veterinary diagnosis.
- Reliable labels for DogBridge MVP intents unless context/outcome evidence exists.

## Candidate Datasets

| Dataset | Link | Size / Shape | Labels | License / Notes | DogBridge Use |
| --- | --- | --- | --- | --- | --- |
| DogSpeak Dataset | https://huggingface.co/datasets/ArlingtonCL2/DogSpeak_Dataset | 77,202 bark sequences from 156 dogs across 5 breeds | `dog_id`, `breed`, `sex`, filename | CC BY-NC-SA 4.0; non-commercial/share-alike | Best public candidate for canine acoustic representation learning and individual-dog experiments. Not direct intent labels. |
| Barkopedia Individual Dog Recognition Dataset | https://huggingface.co/datasets/ArlingtonCL2/Barkopedia_Individual_Dog_Recognition_Dataset | 8,924 labeled dog bark audio clips | individual dog identity | Check dataset card before use | Useful for personalization and dog identity recognition. Not direct intent labels. |
| BarkMeowDB | https://zenodo.org/records/3563990 | Small dog/cat WAV dataset, about 31.7 MB | dog vs cat | Zenodo record; verify license before redistribution | Useful for smoke tests and dog-vs-non-dog classification only. Too small for intent. |
| Cats vs Dogs Audio Classification | https://www.kaggle.com/datasets/stealthtechnologies/cats-vs-dogs-audio-classification | WAV recordings of cat and dog sounds | cat/dog, sound type depending on file metadata | Kaggle terms; verify before use | Useful for binary animal sound checks, not intent. |
| Audio Cats and Dogs | https://www.kaggle.com/datasets/mmoreaux/audio-cats-and-dogs | WAV files for cat/dog audio classification | cat/dog | Kaggle terms; verify before use | Similar to BarkMeowDB: smoke tests and dog-vs-cat only. |

## Useful Model Resources

| Resource | Link | Why It Matters |
| --- | --- | --- |
| YAMNet transfer learning | https://www.tensorflow.org/tutorials/audio/transfer_learning_audio | YAMNet provides 1,024-dimensional audio embeddings and includes bark-related AudioSet classes. Good next step after MFCC. |
| TensorFlow YAMNet README | https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/README.md | Model background and expected audio format. |
| Hugging Face audio classification | https://huggingface.co/docs/transformers/tasks/audio_classification | Later path for Wav2Vec2 or transformer-based audio classifiers. |
| Wav2Vec2 docs | https://huggingface.co/docs/transformers/model_doc/wav2vec2 | Useful if we explore self-supervised speech/audio representations for dog vocalizations. |
| OpenL3 docs | https://openl3.readthedocs.io/ | Alternative pretrained audio embeddings for comparison with YAMNet. |

## Data Separation Rule

Public data and DogBridge owner-labeled data must stay separate.

Suggested folders:

```text
backend/data/public/
backend/data/public/manifests/
backend/data/raw/
backend/data/processed/
```

Suggested public manifest columns:

- `source_dataset`
- `source_url`
- `license`
- `file_path`
- `filename`
- `dog_id`
- `breed`
- `sex`
- `sound_label`
- `context_label`
- `intent_label`
- `usable_for_intent`
- `notes`

For most public datasets, `usable_for_intent` should be `false`.

## Research Position

The strongest DogBridge framing is:

> DogBridge uses public canine vocalization data to learn general dog sound structure, then learns likely intent from each individual dog through owner-confirmed context and outcomes.

This keeps the project scientifically honest while still letting us benefit from large public audio collections.

## Next Implementation Steps

1. Add a public dataset manifest builder.
2. Support importing DogSpeak metadata without mixing it into DogBridge intent clips.
3. Add optional YAMNet embedding extraction.
4. Train a public-data acoustic representation model.
5. Train dog-specific intent models only from owner-confirmed DogBridge clips.
6. Compare MFCC baseline vs YAMNet embeddings vs transformer embeddings.

