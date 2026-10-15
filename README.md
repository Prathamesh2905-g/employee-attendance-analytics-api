# Employee Attendance & Analytics API

A REST API built using FastAPI and MongoDB to manage employee attendance and generate monthly attendance analytics.

## Features

- Employee management
- Employee punch-in and punch-out
- Attendance record correction and history tracking
- Late arrival and overtime calculations
- Half-day attendance tracking
- Monthly employee attendance analytics
- Department-wise attendance summaries
- API documentation through Swagger UI

## Tech Stack

- Python
- FastAPI
- MongoDB
- PyMongo
- Pydantic

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Configure your `.env` file using `.env.example`.
5. Ensure MongoDB is running.
6. Start the application:

   `uvicorn app.main:app --reload`

## API Documentation

After starting the server, open:

`http://127.0.0.1:8000/docs`

## Configuration

Configure `MONGO_URI` and `MONGO_DB` in your local `.env` file.

Never commit real credentials or secrets to the repository.