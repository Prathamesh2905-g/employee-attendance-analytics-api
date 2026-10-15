"""Employee Attendance & Analytics API.

Run from the candidate_kit root:
    uvicorn app.main:app --port 8000
Configuration: MONGO_URI and MONGO_DB (optional local .env).
All timestamps at the HTTP boundary are integer epoch milliseconds.
"""
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
import os
import re
from typing import Optional, Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pymongo import MongoClient, ASCENDING, DESCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

load_dotenv()  # real environment variables take precedence over .env
client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"), serverSelectionTimeoutMS=3000)
db = client[os.getenv("MONGO_DB", "attendance_db")]
UTC = timezone.utc
IST = ZoneInfo("Asia/Kolkata")
PRESENCE = {"PRESENT", "WFH", "ON_DUTY"}
STATUSES = PRESENCE | {"ABSENT", "LEAVE"}
MIN_MS, MAX_MS = 100_000_000_000, 4_102_444_800_000


def ensure_indexes():
    """Create query-supporting and integrity indexes idempotently at startup."""
    db.employees.create_index([("emp_code", ASCENDING)], unique=True, name="uq_employee_code")
    db.employees.create_index([("department", ASCENDING), ("emp_code", ASCENDING)], name="ix_employee_department_code")
    db.employees.create_index([("department", ASCENDING), ("joined_on", ASCENDING)], name="ix_employee_department_joined")
    db.attendance_logs.create_index([("emp_code", ASCENDING), ("date", ASCENDING)], unique=True, name="uq_attendance_employee_date")
    db.attendance_logs.create_index([("date", DESCENDING), ("emp_code", ASCENDING)], name="ix_attendance_date_employee")
    db.attendance_logs.create_index([("emp_code", ASCENDING), ("date", DESCENDING)], name="ix_attendance_employee_date_desc")
    db.attendance_logs.create_index([("status", ASCENDING), ("date", ASCENDING)], name="ix_attendance_status_date")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Fail startup if MongoDB is unavailable; /health also checks readiness later.
    client.admin.command("ping")
    ensure_indexes()
    yield

app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


# ------------------------------ Shared helpers ------------------------------
def round_half_up(value: Any, places: int = 2) -> float:
    quantum = Decimal("1").scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def epoch_ms_to_datetime(value: Any) -> datetime:
    if type(value) is not int or not MIN_MS <= value <= MAX_MS:
        raise ValueError("Timestamp must be an integer epoch-millisecond value in range")
    # Truncate toward the start of the Unix second, including for positive epoch values.
    return datetime.fromtimestamp(value // 1000, tz=UTC)


def datetime_to_epoch_ms(value: Optional[datetime]) -> Optional[int]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)  # PyMongo returns BSON dates as naive UTC by default.
    return int(value.replace(microsecond=0).timestamp() * 1000)


def utc_now_seconds() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)

def aware_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def parse_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError("Date must use YYYY-MM-DD")
    if parsed.isoformat() != value:
        raise ValueError("Date must use YYYY-MM-DD")
    return parsed


def parse_month(value: str) -> tuple[date, date]:
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value):
        raise HTTPException(422, "month must use YYYY-MM")
    first = date.fromisoformat(value + "-01")
    next_month = date(first.year + (first.month == 12), 1 if first.month == 12 else first.month + 1, 1)
    return first, next_month


def parse_clock(value: str) -> time:
    if not isinstance(value, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
        raise ValueError("Time must use HH:MM")
    h, m = map(int, value.split(":"))
    return time(h, m)


def shift_instants(attendance_date: str, shift_start: str, shift_end: str) -> tuple[datetime, datetime]:
    day = parse_date(attendance_date)
    start_clock, end_clock = parse_clock(shift_start), parse_clock(shift_end)
    start_local = datetime.combine(day, start_clock, IST)
    end_day = day + timedelta(days=1) if end_clock <= start_clock else day
    end_local = datetime.combine(end_day, end_clock, IST)
    return start_local.astimezone(UTC), end_local.astimezone(UTC)


def attendance_date_for(punch_in: datetime, employee: dict) -> str:
    local = punch_in.astimezone(IST)
    start_clock, end_clock = parse_clock(employee["shift_start"]), parse_clock(employee["shift_end"])
    day = local.date()
    if end_clock <= start_clock and local.timetz().replace(tzinfo=None) < end_clock:
        day -= timedelta(days=1)
    return day.isoformat()



def compute_late_minutes(
    punch_in: datetime,
    shift_start: str,
    attendance_day: str,
) -> int:
    """Calculate late minutes with a 10-minute grace period."""
    # Convert the actual punch-in instant to Indian Standard Time.
    local_punch_in = punch_in.astimezone(IST)

    # Construct the scheduled shift start for the attendance date.
    shift_start_dt = datetime.strptime(
        f"{attendance_day} {shift_start}",
        "%Y-%m-%d %H:%M",
    ).replace(tzinfo=IST)

    elapsed_seconds = (
        local_punch_in - shift_start_dt
    ).total_seconds()

    # Early arrivals are not late.
    if elapsed_seconds <= 10 * 60:
        return 0

    # Count complete minutes after the 10-minute grace period.
    return int((elapsed_seconds - 10 * 60) // 60)



def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    return round_half_up((punch_out - punch_in).total_seconds() / 3600, 2)


def compute_overtime(punch_out: datetime, shift_end: str, attendance_day: str, shift_start: str) -> int:
    _, scheduled_end = shift_instants(attendance_day, shift_start, shift_end)
    minutes = int((punch_out - scheduled_end).total_seconds() // 60)
    return minutes if minutes >= 30 else 0


def validate_punch_pair(punch_in: Optional[datetime], punch_out: Optional[datetime]):
    if punch_in is None:
        raise HTTPException(422, "Presence status requires punch_in")
    if punch_out is not None:
        seconds = (punch_out - punch_in).total_seconds()
        if seconds <= 0:
            raise HTTPException(422, "punch_out must be after punch_in")
        if seconds > 24 * 3600:
            raise HTTPException(422, "punch_out must be within 24 hours of punch_in")


def calculate_derived(record: dict, employee: dict) -> dict:
    status = record["status"]
    pin, pout = record.get("punch_in"), record.get("punch_out")
    if status in {"ABSENT", "LEAVE"}:
        if pin is not None or pout is not None:
            raise HTTPException(422, "ABSENT and LEAVE records cannot have punch times")
        return {"late_minutes": 0, "work_hours": None, "overtime_minutes": 0, "half_day": False}
    validate_punch_pair(pin, pout)
    late = compute_late_minutes(pin, employee["shift_start"], record["date"])
    if pout is None:
        return {"late_minutes": late, "work_hours": None, "overtime_minutes": 0, "half_day": False}
    hours = compute_work_hours(pin, pout)
    return {"late_minutes": late, "work_hours": hours,
            "overtime_minutes": compute_overtime(pout, employee["shift_end"], record["date"], employee["shift_start"]),
            "half_day": hours < 4.50}


def serialize_history(history: Optional[list]) -> list:
    output = []
    for entry in history or []:
        changes = {}
        for field, pair in (entry.get("changes") or {}).items():
            changes[field] = {k: datetime_to_epoch_ms(v) if field in {"punch_in", "punch_out"} else v
                              for k, v in pair.items()}
        output.append({"at": datetime_to_epoch_ms(entry.get("at")), "by": entry.get("by"),
                       "reason": entry.get("reason"), "changes": changes})
    return output


def serialize_attendance(doc: dict) -> dict:
    return {"emp_code": doc["emp_code"], "date": doc["date"], "status": doc["status"],
            "punch_in": datetime_to_epoch_ms(doc.get("punch_in")),
            "punch_out": datetime_to_epoch_ms(doc.get("punch_out")),
            "work_hours": doc.get("work_hours"), "late_minutes": int(doc.get("late_minutes", 0)),
            "overtime_minutes": int(doc.get("overtime_minutes", 0)), "half_day": bool(doc.get("half_day", False)),
            "history": serialize_history(doc.get("history", []))}


def serialize_employee(doc: dict) -> dict:
    return {"emp_code": doc["emp_code"], "name": doc["name"], "email": doc["email"],
            "department": doc["department"], "shift_start": doc["shift_start"],
            "shift_end": doc["shift_end"], "joined_on": doc["joined_on"],
            "created_at": datetime_to_epoch_ms(doc["created_at"])}


def page_params(page: int, page_size: int):
    if page < 1:
        raise HTTPException(422, "page must be >= 1")
    if not 1 <= page_size <= 100:
        raise HTTPException(422, "page_size must be between 1 and 100")


def month_match(month: str) -> dict:
    first, next_month = parse_month(month)
    return {"date": {"$gte": first.isoformat(), "$lt": next_month.isoformat()}}, first, next_month


def working_days(start: date, end_exclusive: date) -> int:
    count, day = 0, start
    while day < end_exclusive:
        if day.weekday() < 5:
            count += 1
        day += timedelta(days=1)
    return count


def round_rate(value: Optional[float], places: int = 4):
    return None if value is None else round_half_up(value, places)


def clean_explain(value):
    """Convert BSON types in explain output to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): clean_explain(v) for k, v in value.items() if k != "$clusterTime"}
    if isinstance(value, list):
        return [clean_explain(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "binary") and hasattr(value, "generation_time"):
        return str(value)
    return value


# ---------------------------------- Models ----------------------------------
class EmployeeIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = "09:30"
    shift_end: str = "18:30"
    joined_on: str

    @field_validator("shift_start", "shift_end")
    @classmethod
    def valid_clock(cls, value):
        parse_clock(value)
        return value

    @field_validator("joined_on")
    @classmethod
    def valid_joined_date(cls, value):
        parse_date(value)
        return value

    @model_validator(mode="after")
    def different_shift_times(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    emp_code: str
    punched_at: Optional[int] = None
    status: str = "PRESENT"

    @field_validator("punched_at")
    @classmethod
    def valid_timestamp(cls, value):
        if value is not None:
            epoch_ms_to_datetime(value)
        return value

    @field_validator("status")
    @classmethod
    def valid_presence_status(cls, value):
        if value not in PRESENCE:
            raise ValueError("status must be PRESENT, WFH or ON_DUTY")
        return value


class PunchOutIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    emp_code: str
    punched_at: Optional[int] = None

    @field_validator("punched_at")
    @classmethod
    def valid_timestamp(cls, value):
        if value is not None:
            epoch_ms_to_datetime(value)
        return value


class RegularizeIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: Optional[str] = None
    punch_in: Optional[int] = None
    punch_out: Optional[int] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)

    @field_validator("status")
    @classmethod
    def valid_status(cls, value):
        if value is not None and value not in STATUSES:
            raise ValueError("Invalid status")
        return value

    @field_validator("punch_in", "punch_out")
    @classmethod
    def valid_timestamp(cls, value):
        if value is not None:
            epoch_ms_to_datetime(value)
        return value


# ---------------------------------- Routes ----------------------------------
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        return {"status": "ok"}
    except PyMongoError:
        raise HTTPException(503, "MongoDB unavailable")


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = utc_now_seconds()
    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return serialize_employee(doc)


@app.get("/employees")
def list_employees(department: Optional[str] = None, page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100)):
    page_params(page, page_size)
    query = {"department": department} if department is not None else {}
    total = db.employees.count_documents(query)
    docs = db.employees.find(query).sort("emp_code", ASCENDING).skip((page - 1) * page_size).limit(page_size)
    return {"items": [serialize_employee(d) for d in docs], "total": total, "page": page, "page_size": page_size}


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(404, "Employee not found")
    ts = epoch_ms_to_datetime(body.punched_at) if body.punched_at is not None else utc_now_seconds()
    attendance_day = attendance_date_for(ts, emp)
    doc = {"emp_code": body.emp_code, "date": attendance_day, "status": body.status,
           "punch_in": ts, "punch_out": None, "work_hours": None,
           "late_minutes": compute_late_minutes(ts, emp["shift_start"], attendance_day),
           "overtime_minutes": 0, "half_day": False, "history": []}
    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "Already punched in for this attendance date")
    return serialize_attendance(doc)


@app.get("/attendance")
def list_attendance(emp_code: Optional[str] = None, date_from: Optional[str] = None,
                    date_to: Optional[str] = None, status: Optional[str] = None,
                    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100)):
    page_params(page, page_size)
    query = {}
    if emp_code is not None: query["emp_code"] = emp_code
    if status is not None:
        if status not in STATUSES: raise HTTPException(422, "Invalid status")
        query["status"] = status
    if date_from:
        try: parse_date(date_from)
        except ValueError as exc: raise HTTPException(422, str(exc))
    if date_to:
        try: parse_date(date_to)
        except ValueError as exc: raise HTTPException(422, str(exc))
    if date_from and date_to and date_from > date_to:
        raise HTTPException(422, "date_from must be <= date_to")
    if date_from or date_to:
        query["date"] = {}
        if date_from: query["date"]["$gte"] = date_from
        if date_to: query["date"]["$lte"] = date_to
    total = db.attendance_logs.count_documents(query)
    docs = db.attendance_logs.find(query).sort([("date", DESCENDING), ("emp_code", ASCENDING)]).skip((page - 1) * page_size).limit(page_size)
    return {"items": [serialize_attendance(d) for d in docs], "total": total, "page": page, "page_size": page_size}


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp: raise HTTPException(404, "Employee not found")
    ts = epoch_ms_to_datetime(body.punched_at) if body.punched_at is not None else utc_now_seconds()
    # The newest open record at or before this instant is eligible, including overnight shifts.
    record = db.attendance_logs.find_one({"emp_code": body.emp_code, "status": {"$in": list(PRESENCE)},
                                          "punch_in": {"$lte": ts.replace(tzinfo=None)}, "punch_out": None},
                                         sort=[("punch_in", DESCENDING)])
    if not record:
        closed = db.attendance_logs.find_one({"emp_code": body.emp_code, "status": {"$in": list(PRESENCE)},
                                              "punch_in": {"$lte": ts.replace(tzinfo=None)}}, sort=[("punch_in", DESCENDING)])
        if closed: raise HTTPException(409, "Attendance record is already punched out")
        raise HTTPException(404, "No punch-in found before this timestamp")
    record["punch_in"] = aware_utc(record["punch_in"])
    if (ts - record["punch_in"]).total_seconds() <= 0 or (ts - record["punch_in"]).total_seconds() > 86400:
        raise HTTPException(422, "punch_out must be after punch_in and within 24 hours")
    proposed = dict(record); proposed["punch_out"] = ts
    derived = calculate_derived(proposed, emp)
    updated = db.attendance_logs.find_one_and_update(
        {"_id": record["_id"], "punch_out": None},
        {"$set": {"punch_out": ts, **derived}}, return_document=ReturnDocument.AFTER)
    if updated is None: raise HTTPException(409, "Record was punched out concurrently")
    return serialize_attendance(updated)


@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(emp_code: str, date: str, body: RegularizeIn):
    try: parsed_day = parse_date(date)
    except ValueError as exc: raise HTTPException(422, str(exc))
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp: raise HTTPException(404, "Employee not found")
    original = db.attendance_logs.find_one({"emp_code": emp_code, "date": parsed_day.isoformat()})
    if not original: raise HTTPException(404, "Attendance record not found")
    proposed = {k: original.get(k) for k in ["emp_code", "date", "status", "punch_in", "punch_out", "work_hours", "late_minutes", "overtime_minutes", "half_day"]}
    proposed["punch_in"] = aware_utc(proposed.get("punch_in"))
    proposed["punch_out"] = aware_utc(proposed.get("punch_out"))
    if body.status is not None: proposed["status"] = body.status
    if body.punch_in is not None: proposed["punch_in"] = epoch_ms_to_datetime(body.punch_in)
    if body.punch_out is not None: proposed["punch_out"] = epoch_ms_to_datetime(body.punch_out)
    if proposed["status"] in {"ABSENT", "LEAVE"}:
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(422, "Cannot supply punch times for ABSENT or LEAVE")
        proposed["punch_in"] = proposed["punch_out"] = None
    else:
        if proposed.get("punch_in") is None: raise HTTPException(422, "Presence status requires punch_in")
        if proposed["punch_in"].astimezone(IST).date().isoformat() != parsed_day.isoformat() and not (parse_clock(emp["shift_end"]) <= parse_clock(emp["shift_start"]) and proposed["punch_in"].astimezone(IST).time().replace(tzinfo=None) < parse_clock(emp["shift_end"])):
            raise HTTPException(422, "punch_in must belong to the attendance date")
    proposed.update(calculate_derived(proposed, emp))
    fields = ["status", "punch_in", "punch_out", "work_hours", "late_minutes", "overtime_minutes", "half_day"]
    changes = {}
    for field in fields:
        before, after = original.get(field), proposed.get(field)
        if field in {"punch_in", "punch_out"}:
            before = aware_utc(before)
        if before != after:
            changes[field] = {"from": before, "to": after}
    if not changes: raise HTTPException(422, "Request does not change the record")
    now = utc_now_seconds()
    history_entry = {"at": now, "by": body.regularized_by, "reason": body.reason, "changes": changes}
    # Compare-and-swap on fields from the read version. If another editor wins, this update matches no document.
    guard = {"_id": original["_id"], "status": original.get("status"),
             "punch_in": original.get("punch_in"), "punch_out": original.get("punch_out"),
             "history": original.get("history", [])}
    updated = db.attendance_logs.find_one_and_update(guard,
        {"$set": {k: proposed.get(k) for k in fields}, "$push": {"history": history_entry}},
        return_document=ReturnDocument.AFTER)
    if updated is None: raise HTTPException(409, "Record was modified concurrently; retry")
    return serialize_attendance(updated)


# ---------------------------- Analytics pipelines ---------------------------
def employee_monthly_pipeline(emp_code: str, month: str):
    match, first, next_month = month_match(month)
    return [{"$match": {"emp_code": emp_code, **match}},
            {"$addFields": {"_day": {"$dateFromString": {"dateString": "$date"}},
                            "_present": {"$cond": [{"$in": ["$status", list(PRESENCE)]}, 1, 0]}}},
            {"$group": {"_id": None,
                "present_days": {"$sum": {"$cond": [{"$and": [{"$lt": [{"$dayOfWeek": "$_day"}, 7]}, {"$gt": [{"$dayOfWeek": "$_day"}, 1]}, {"$eq": ["$_present", 1]}]}, {"$cond": [{"$eq": ["$half_day", True]}, 0.5, 1]}, 0]}},
                "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
                "late_count": {"$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}},
                "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
                "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}}}}]


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$")):
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp: raise HTTPException(404, "Employee not found")
    _, first, next_month = month_match(month)
    eligible_start = max(first, parse_date(emp["joined_on"]))
    days = working_days(eligible_start, next_month) if eligible_start < next_month else 0
    rows = list(db.attendance_logs.aggregate(employee_monthly_pipeline(emp_code, month)))
    row = rows[0] if rows else {}
    present = round_half_up(row.get("present_days", 0), 2)
    pct = round_half_up(present / days * 100, 2) if days else None
    return {"emp_code": emp_code, "month": month, "working_days": days, "present_days": present,
            "leave_days": row.get("leave_days", 0), "late_count": row.get("late_count", 0),
            "total_late_minutes": row.get("total_late_minutes", 0),
            "total_overtime_minutes": row.get("total_overtime_minutes", 0), "attendance_pct": pct}


@app.get("/analytics/departments/summary")
def department_summary(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = None,
):
    _, first, next_month = month_match(month)

    # Match employees who joined before the next month.
    employee_match = {
        "joined_on": {"$lt": next_month.isoformat()}
    }

    if department is not None:
        employee_match["department"] = department

    pipeline = [
        {"$match": employee_match},

        # Fetch attendance logs for each employee in the selected month.
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"code": "$emp_code"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$emp_code", "$$code"]},
                                    {"$gte": ["$date", first.isoformat()]},
                                    {"$lt": ["$date", next_month.isoformat()]},
                                ]
                            }
                        }
                    }
                ],
                "as": "logs",
            }
        },

        # Keep employees who have no attendance records.
        {
            "$unwind": {
                "path": "$logs",
                "preserveNullAndEmptyArrays": True,
            }
        },

        # Prepare fields used by the aggregation.
        {
            "$addFields": {
                "_day": {
                    "$cond": [
                        {"$ne": [{"$ifNull": ["$logs.date", None]}, None]},
                        {"$dateFromString": {"dateString": "$logs.date"}},
                        None,
                    ]
                },
                "_status": "$logs.status",
                "_late": {"$ifNull": ["$logs.late_minutes", 0]},
                "_hours": "$logs.work_hours",
            }
        },

        # Aggregate monthly statistics by department.
        {
            "$group": {
                "_id": "$department",

                # Count each employee only once, regardless of log count.
                "employee_codes": {"$addToSet": "$emp_code"},

                "present_days": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$ne": ["$_day", None]},
                                    {"$in": ["$_status", list(PRESENCE)]},
                                    {"$lt": [{"$dayOfWeek": "$_day"}, 7]},
                                    {"$gt": [{"$dayOfWeek": "$_day"}, 1]},
                                ]
                            },
                            {
                                "$cond": [
                                    {"$eq": ["$logs.half_day", True]},
                                    0.5,
                                    1,
                                ]
                            },
                            0,
                        ]
                    }
                },

                "hours": {
                    "$push": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$_status", list(PRESENCE)]},
                                    {"$ne": ["$_hours", None]},
                                ]
                            },
                            "$_hours",
                            None,
                        ]
                    }
                },

                "late_count": {
                    "$sum": {
                        "$cond": [
                            {"$gt": ["$_late", 0]},
                            1,
                            0,
                        ]
                    }
                },

                "total_late_minutes": {"$sum": "$_late"},

                "leave_count": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$_status", "LEAVE"]},
                            1,
                            0,
                        ]
                    }
                },

                "on_duty_count": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$_status", "ON_DUTY"]},
                            1,
                            0,
                        ]
                    }
                },
            }
        },

        # Format the response and calculate average work hours.
        {
            "$project": {
                "_id": 0,
                "department": "$_id",
                "headcount": {"$size": "$employee_codes"},
                "present_days": 1,
                "late_count": 1,
                "total_late_minutes": 1,
                "leave_count": 1,
                "on_duty_count": 1,
                "avg_work_hours": {
                    "$cond": [
                        {
                            "$gt": [
                                {
                                    "$size": {
                                        "$filter": {
                                            "input": "$hours",
                                            "as": "h",
                                            "cond": {"$ne": ["$$h", None]},
                                        }
                                    }
                                },
                                0,
                            ]
                        },
                        {
                            "$avg": {
                                "$filter": {
                                    "input": "$hours",
                                    "as": "h",
                                    "cond": {"$ne": ["$$h", None]},
                                }
                            }
                        },
                        None,
                    ]
                },
            }
        },

        {"$sort": {"department": 1}},
    ]

    items = list(db.employees.aggregate(pipeline))

    for item in items:
        item["present_days"] = round_half_up(
            item.get("present_days", 0), 2
        )

        if item.get("avg_work_hours") is not None:
            item["avg_work_hours"] = round_half_up(
                item["avg_work_hours"], 2
            )

    return {
        "month": month,
        "items": items,
    }

@app.get("/analytics/leaderboard/late")
def late_leaderboard(month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
                     limit: int = Query(10, ge=1, le=50), department: Optional[str] = None):
    match, _, _ = month_match(month)
    pipeline = [{"$match": match}, {"$group": {"_id": "$emp_code", "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
                                                 "late_count": {"$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}}}},
      {"$match": {"total_late_minutes": {"$gt": 0}}},
      {"$lookup": {"from": "employees", "localField": "_id", "foreignField": "emp_code", "as": "employee"}},
      {"$unwind": "$employee"},
      *([{"$match": {"employee.department": department}}] if department is not None else []),
      {"$setWindowFields": {"sortBy": {"total_late_minutes": -1}, "output": {"rank": {"$rank": {}}}}},
      {"$match": {"rank": {"$lte": limit}}}, {"$sort": {"total_late_minutes": -1, "_id": 1}},
      {"$project": {"_id": 0, "rank": 1, "emp_code": "$_id", "name": "$employee.name", "department": "$employee.department", "total_late_minutes": 1, "late_count": 1}}]
    return {"month": month, "items": list(db.attendance_logs.aggregate(pipeline))}


@app.get("/analytics/departments/{department}/trend")
def department_trend(department: str, from_: str = Query(..., alias="from"), to: str = Query(...)):
    try: start_day, end_day = parse_date(from_), parse_date(to)
    except ValueError as exc: raise HTTPException(422, str(exc))
    span = (end_day - start_day).days + 1
    if span < 1: raise HTTPException(422, "to must be on or after from")
    if span > 92: raise HTTPException(422, "Date range cannot exceed 92 days")
    if not db.employees.find_one({"department": department}): raise HTTPException(404, "Department not found")
    # Gap filling and the seven-row moving window happen in MongoDB, not in Python.
    pipeline = [
      {"$match": {"date": {"$gte": start_day.isoformat(), "$lte": end_day.isoformat()}}},
      {"$lookup": {"from": "employees", "localField": "emp_code", "foreignField": "emp_code", "as": "employee"}},
      {"$unwind": "$employee"}, {"$match": {"employee.department": department}},
      {"$group": {"_id": "$date", "present_count": {"$sum": {"$cond": [{"$in": ["$status", list(PRESENCE)]}, {"$cond": [{"$eq": ["$half_day", True]}, 0.5, 1]}, 0]}},
                   "late_count": {"$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}}}},
      {"$project": {"_id": 0, "date": {"$dateFromString": {"dateString": "$_id"}}, "present_count": 1, "late_count": 1}},
      {"$densify": {"field": "date", "range": {"step": 1, "unit": "day", "bounds": [datetime.combine(start_day, time.min), datetime.combine(end_day + timedelta(days=1), time.min)]}}},
      {"$set": {"present_count": {"$ifNull": ["$present_count", 0]}, "late_count": {"$ifNull": ["$late_count", 0]}}},
      {"$lookup": {"from": "employees", "let": {"day": "$date"}, "pipeline": [{"$match": {"department": department}},
        {"$match": {"$expr": {"$lte": [{"$dateFromString": {"dateString": "$joined_on"}}, "$$day"]}}}, {"$count": "n"}], "as": "hc"}},
      {"$set": {"headcount": {"$ifNull": [{"$arrayElemAt": ["$hc.n", 0]}, 0],
                     "is_working_day": {"$and": [{"$ne": [{"$dayOfWeek": "$date"}, 1]}, {"$ne": [{"$dayOfWeek": "$date"}, 7]}]}}}},
      {"$set": {"attendance_rate": {"$cond": [{"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]}, {"$divide": ["$present_count", "$headcount"]}, None]},
                "date": {"$dateToString": {"date": "$date", "format": "%Y-%m-%d"}}}},
      {"$setWindowFields": {"sortBy": {"date": 1}, "output": {"moving_avg_7d": {"$avg": "$attendance_rate", "window": {"documents": [-6, 0]}}}}},
      {"$project": {"_id": 0, "date": 1, "is_working_day": 1, "headcount": 1, "present_count": 1, "late_count": 1, "attendance_rate": 1, "moving_avg_7d": 1}},
      {"$sort": {"date": 1}}]
    items = list(db.attendance_logs.aggregate(pipeline))
    for item in items:
        item["attendance_rate"] = round_rate(item["attendance_rate"], 4)
        item["moving_avg_7d"] = round_rate(item["moving_avg_7d"], 4)
    # If no attendance exists at all in the interval, aggregation has no input rows for $densify.
    # A date spine is normally seeded by records; this branch keeps empty periods complete using DB-side $documents.
    if len(items) != span:
        # Seed a date spine in MongoDB and join the department's daily rollup, still no Python date loop.
        spine = [{"$documents": [{"date": datetime.combine(start_day, time.min)}]},
                 {"$densify": {"field": "date", "range": {"step": 1, "unit": "day", "bounds": [datetime.combine(start_day, time.min), datetime.combine(end_day + timedelta(days=1), time.min)]}}},
                 {"$lookup": {"from": "attendance_logs", "let": {"day": {"$dateToString": {"date": "$date", "format": "%Y-%m-%d"}}}, "pipeline": [
                   {"$match": {"$expr": {"$eq": ["$date", "$$day"]}}}, {"$lookup": {"from": "employees", "localField": "emp_code", "foreignField": "emp_code", "as": "employee"}},
                   {"$unwind": "$employee"}, {"$match": {"employee.department": department}},
                   {"$group": {"_id": None, "present_count": {"$sum": {"$cond": [{"$in": ["$status", list(PRESENCE)]}, {"$cond": [{"$eq": ["$half_day", True]}, 0.5, 1]}, 0]}}, "late_count": {"$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}}}}], "as": "daily"}},
                 {"$set": {"present_count": {"$ifNull": [{"$arrayElemAt": ["$daily.present_count", 0]}, 0]}, "late_count": {"$ifNull": [{"$arrayElemAt": ["$daily.late_count", 0]}, 0]}}},
                 {"$lookup": {"from": "employees", "let": {"day": "$date"}, "pipeline": [{"$match": {"department": department}}, {"$match": {"$expr": {"$lte": [{"$dateFromString": {"dateString": "$joined_on"}}, "$$day"]}}}, {"$count": "n"}], "as": "hc"}},
                 {"$set": {"headcount": {"$ifNull": [{"$arrayElemAt": ["$hc.n", 0]}, 0], "is_working_day": {"$and": [{"$ne": [{"$dayOfWeek": "$date"}, 1]}, {"$ne": [{"$dayOfWeek": "$date"}, 7]}]}}}},
                 {"$set": {"attendance_rate": {"$cond": [{"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]}, {"$divide": ["$present_count", "$headcount"]}, None], "date": {"$dateToString": {"date": "$date", "format": "%Y-%m-%d"}}}}},
                 {"$setWindowFields": {"sortBy": {"date": 1}, "output": {"moving_avg_7d": {"$avg": "$attendance_rate", "window": {"documents": [-6, 0]}}}}},
                 {"$project": {"_id": 0, "date": 1, "is_working_day": 1, "headcount": 1, "present_count": 1, "late_count": 1, "attendance_rate": 1, "moving_avg_7d": 1}}, {"$sort": {"date": 1}}]
        items = list(db.command({"aggregate": 1, "pipeline": spine, "cursor": {}})["cursor"]["firstBatch"])
        for item in items:
            item["attendance_rate"] = round_rate(item.get("attendance_rate"), 4)
            item["moving_avg_7d"] = round_rate(item.get("moving_avg_7d"), 4)
    return {"department": department, "items": items}


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(endpoint: str, emp_code: Optional[str] = None, month: Optional[str] = None,
                     department: Optional[str] = None, limit: int = Query(10, ge=1, le=50),
                     date_from: Optional[str] = None, date_to: Optional[str] = None,
                     status: Optional[str] = None, from_: Optional[str] = Query(None, alias="from"),
                     to: Optional[str] = None, page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100)):
    page_params(page, page_size)
    try:
        if endpoint == "attendance_list":
            q = {}
            if emp_code is not None: q["emp_code"] = emp_code
            if status is not None: q["status"] = status
            if date_from or date_to:
                q["date"] = {}
                if date_from: q["date"]["$gte"] = parse_date(date_from).isoformat()
                if date_to: q["date"]["$lte"] = parse_date(date_to).isoformat()
            command = {"find": "attendance_logs", "filter": q, "sort": {"date": -1, "emp_code": 1}, "skip": (page-1)*page_size, "limit": page_size}
            collection = "attendance_logs"
        elif endpoint == "employee_monthly":
            if not emp_code or not month: raise HTTPException(422, "emp_code and month are required")
            pipeline = employee_monthly_pipeline(emp_code, month)
            command = {"aggregate": "attendance_logs", "pipeline": pipeline, "cursor": {}}
            collection = "attendance_logs"
        elif endpoint == "department_summary":
            if not month: raise HTTPException(422, "month is required")
            # Explain the same employee-rooted aggregation used by department_summary.
            _, first, next_month = month_match(month)
            pipeline = [{"$match": {**({"department": department} if department else {}), "joined_on": {"$lt": next_month.isoformat()}}},
                        {"$lookup": {"from": "attendance_logs", "let": {"code": "$emp_code"}, "pipeline": [{"$match": {"$expr": {"$and": [{"$eq": ["$emp_code", "$$code"]}, {"$gte": ["$date", first.isoformat()]}, {"$lt": ["$date", next_month.isoformat()]}]}}}], "as": "logs"}}]
            command = {"aggregate": "employees", "pipeline": pipeline, "cursor": {}}
            collection = "employees"
        elif endpoint == "late_leaderboard":
            if not month: raise HTTPException(422, "month is required")
            match, _, _ = month_match(month)
            pipeline = [{"$match": match}, {"$group": {"_id": "$emp_code", "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}}}}, {"$match": {"total_late_minutes": {"$gt": 0}}}]
            command = {"aggregate": "attendance_logs", "pipeline": pipeline, "cursor": {}}
            collection = "attendance_logs"
        elif endpoint == "department_trend":
            if not department or not from_ or not to: raise HTTPException(422, "department, from and to are required")
            a, b = parse_date(from_), parse_date(to)
            if b < a or (b-a).days+1 > 92: raise HTTPException(422, "Invalid trend date range")
            pipeline = [{"$match": {"date": {"$gte": a.isoformat(), "$lte": b.isoformat()}}}, {"$group": {"_id": "$date", "count": {"$sum": 1}}}]
            command = {"aggregate": "attendance_logs", "pipeline": pipeline, "cursor": {}}
            collection = "attendance_logs"
        else:
            raise HTTPException(422, "Unknown endpoint. Use attendance_list, employee_monthly, department_summary, late_leaderboard or department_trend")
        result = db.command({"explain": command, "verbosity": "executionStats"})
        return {"endpoint": endpoint, "collection": collection, "explain": clean_explain(result)}
    except ValueError as exc:
        raise HTTPException(422, str(exc))
