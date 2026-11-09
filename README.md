# Skin Condition Classifier

**A multimodal deep-learning tool that suggests likely skin conditions from a photo and a short symptom questionnaire.**

This is the working artifact from my BSc Data Science final-year project, *NHS Multimodal AI Health Assistant*, which explored how AI could give people preliminary health guidance before they see a clinician.

> ⚠️ **Not a medical device.** This is a research and educational project. Its output is not a diagnosis and must not replace professional medical advice.

## How it works

Most image classifiers look at the photo alone. This model combines two kinds of input, the way a clinician would:

- **The image:** an **EfficientNet-B3** convolutional neural network, pretrained on ImageNet and fine-tuned on skin images, extracts visual features.
- **The context:** a questionnaire captures 38 structured features: age, sex at birth, skin's reaction to sun, where on the body the issue is, how it feels, related symptoms and how long it has lasted.
- **Fusion:** both feature sets are combined in a custom PyTorch `CombinedModel` and passed through fully connected layers to predict one of **20 skin conditions**.

The Streamlit interface walks the user through uploading images and answering the questions, then shows the most likely conditions with confidence scores.

### Conditions covered

Acne · Acute dermatitis · Allergic contact dermatitis · Contact dermatitis · Drug rash · Eczema · Folliculitis · Herpes zoster · Impetigo · Insect bite · Keratosis pilaris · Lichen planus · Pigmented purpuric eruption · Pityriasis rosea · Psoriasis · Scabies · Stasis dermatitis · Tinea · Urticaria · Viral exanthem

## Tech

Python · PyTorch · torchvision (EfficientNet-B3) · OpenCV · scikit-learn · pandas · Streamlit

## Running it

```bash
pip install -r requirements.txt
streamlit run frontend/streamlit_interface.py
```

The trained weights (`final_skin_model.pth`) and preprocessing metadata (`model_metadata.joblib`) are included, so no training is needed to try it. A dev container configuration is provided in `.devcontainer/`.

## Wider project

The full final-year project combined this image model with a text-based symptom checker (BioBERT and a Random Forest) and a knowledge base built from NHS A–Z content, all in a single Streamlit assistant.

## Ideas for improvement

- Store consenting users' images and answers to keep improving the model
- Report per-condition performance and calibrate confidence scores
- Test fairness across skin tones, a known weakness of dermatology datasets
