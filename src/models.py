# UserProfile class
# Any Pydantic models
# Data schemas

from typing import List, Optional
from pydantic import BaseModel, Field

class UserProfile(BaseModel):
    """User profile specifically designed for medical professionals"""
    name: Optional[str] = Field(default=None, description="The user's preferred name")
    age: Optional[int] = Field(default=None, ge=18, le=100, description="The user's age")
    gender: Optional[str] = Field(default=None, description="The user's gender")
    location: Optional[str] = Field(default=None, description="City/State/Country where the user practices")
    timezone: Optional[str] = Field(default=None, description="The user's timezone")
    language: Optional[str] = Field(default="English", description="Preferred language for communication")
    medical_role: Optional[str] = Field(default=None, description="Primary medical role/profession")
    specialty: Optional[str] = Field(default=None, description="Medical specialty or area of focus")
    subspecialty: Optional[str] = Field(default=None, description="Subspecialty within their field")
    years_of_experience: Optional[int] = Field(default=None, ge=0, description="Years of professional experience")
    clinical_interests: Optional[List[str]] = Field(default=None, description="Specific clinical areas of interest")
    research_interests: Optional[List[str]] = Field(default=None, description="Research areas of interest")
    career_goals: Optional[List[str]] = Field(default=None, description="Professional career objectives")
    medical_software_used: Optional[List[str]] = Field(default=None, description="Electronic health records and medical software used")
    preferred_medical_resources: Optional[List[str]] = Field(default=None, description="Preferred medical references and resources")