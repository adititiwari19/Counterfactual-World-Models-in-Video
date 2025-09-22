# URSA Fall 2025: Counterfactual World Models in Video  

Mentor: **Aditi Tiwari (CS @ UIUC)**  
Mentees: **Tia Shrikhande & Abhay Pokhriyal**  

---

## 📌 Project Overview  
This project explores **counterfactual world models in video** — AI systems that do not just predict *what happens next*, but also *what could have happened if something were different*.  

Example:  
- Video shows someone picking up a hammer.  
- Counterfactual question: *“What if they didn’t pick it up?”*  

Through toy experiments, we aim to build a **prototype system** that links narration, visual frames, and counterfactual reasoning.  

---

## 🎯 Goals  
- Learn research skills through hands-on experimentation.  
- Build a **prototype system** and a **final write-up/demo**.  
- Potentially submit to a **workshop-style paper** (stretch goal).  
- Emphasize learning, curiosity, and collaboration.  

---

## 🛠 Tools & Libraries  
We will primarily use **Google Colab** (free GPU runtime) for development.  

Core libraries and models:  
- [OpenCV](https://opencv.org/) → frame extraction, video handling  
- [PyTorch](https://pytorch.org/) → ML backbone  
- [CLIP](https://github.com/openai/CLIP) → image-text alignment  
- [Whisper](https://github.com/openai/whisper) → speech-to-text transcription  
- [VideoMAE / TimeSformer](https://github.com/MCG-NJU/VideoMAE) → video embeddings (later stages)  

---

---

## 🚀 Starter Experiment (Weeks 2–3)  
Goal: Given a short video (5–10s), link narration words to the correct video frame.  

**Steps:**  
1. Pick a toy video (e.g., someone lifting a cup/hammer).  
2. Run **Whisper** → extract transcript.  
3. Use **OpenCV** → extract 1 frame per second.  
4. Use **CLIP** → encode frames + encode a keyword from transcript (e.g., *“hammer”*).  
5. Compare similarity scores across frames.  
6. Plot results: *frame index vs. similarity*.  

Expected outcome: A simple, visible demo where AI finds the right frame for the mentioned object.  

---

## ✅ Week 1 To-Do  
- [ ✅ ] Set up communication channel (Slack/Discord/Group chat).  - Weekly meeting on Thursday over Zoom at 5:30PM
- [ ] Clone this repo & make first commit.  
- [ ] Test a Colab notebook with basic imports (`opencv`, `torch`, `clip`, `whisper`).  
- [ ] Skim background on **World Models** and **Counterfactual Reasoning**.  
- [ ] Share 1–2 related papers or interesting links in our notes.  

---

## 📖 References  
- Hafner et al. (Dreamer): [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603)  
- OpenAI CLIP: [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)  
- OpenAI Whisper: [https://arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356)  

