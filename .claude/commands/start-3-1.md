# Lesson 3.1: Model Serving Options

Read the lesson content from `lesson-modules/3-deployment/3-1-serving-options.md` and guide the student through it.

## Lesson Flow

### 1. Opener (2 min)
"You have a trained, tested, versioned model. Now someone needs to use it. How do they access it?"

### 2. Socratic Question
Ask: "What are the different ways you could let users or systems make predictions with your model?"

Expected answers: API, batch processing, embedded in app, exported model file. Guide them to understand that serving strategy depends on latency requirements, scale, and use case.

### 3. Serving Patterns Overview (10 min)
Cover the main patterns:

**Online Serving (Real-time)**
- REST APIs (FastAPI, Flask)
- gRPC for high-performance
- Latency-critical applications

**Batch Serving**
- Process large datasets offline
- Scheduled jobs (Airflow, cron)
- Cost-effective for non-urgent predictions

**Embedded Serving**
- Model runs in client application
- Edge devices, mobile apps
- No network latency, privacy benefits

**Managed Services**
- SageMaker, Vertex AI, Azure ML
- Abstracted infrastructure
- Higher cost, less control

### 4. Choosing the Right Pattern (10 min)
Walk through decision criteria:
- What's the latency requirement? (<100ms = online, minutes/hours = batch)
- What's the request volume? (Bursty vs steady)
- Where does the data live? (Client-side vs server-side)
- What's the deployment target? (Cloud, edge, mobile)

Exercise: Ask them which pattern fits the sentiment classifier use case.

### 5. Why We're Choosing REST API (5 min)
For our sentiment classifier:
- Real-time feedback classification
- Simple request/response pattern
- Easy to integrate with any client
- Good learning foundation

Introduce FastAPI as the framework of choice:
- Modern, fast, automatic docs
- Type hints and validation
- Async support

### 6. Wrap Up
- Serving strategy is a design decision, not an afterthought
- Match the pattern to your requirements
- Preview: Lesson 3.2 builds the FastAPI service
- Next: `/start-3-2`

## Teaching Notes
- Don't overwhelm with options—focus on trade-offs
- Real-time APIs are most common for learning
- Connect to their experience: "Have you consumed ML APIs before?"
- Level 3 builds on the tested model from Level 2
