"""Department registry endpoint."""

from fastapi import APIRouter
from api.schemas import DepartmentInfo, DepartmentsResponse
from config import DEPARTMENTS

router = APIRouter(prefix="/departments", tags=["departments"])


@router.get("", response_model=DepartmentsResponse)
async def list_departments():
    """List all configured departments and their display names."""
    deps = [
        DepartmentInfo(dept_id=k, display_name=v["display_name"])
        for k, v in DEPARTMENTS.items()
    ]
    return DepartmentsResponse(departments=deps)
