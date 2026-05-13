from pydantic import BaseModel, Field

class PlacementPredictorInput(BaseModel):
    branch: str = Field(..., description="Branch of the student", examples=["CSE", "ECE", "EE"])
    college_tier: str = Field(..., description="Tier of the college", examples=["Tier 1", "Tier 2", "Tier 3"])
    cgpa: float = Field(..., description="Cumulative Grade Point Average", examples=[8.5, 9.0])
    backlogs: int = Field(..., description="Number of backlogs", examples=[0, 1, 2])
    coding_skills: float = Field(..., description="Coding skills rating (1-10)", examples=[7, 8, 9])
    dsa_score: float = Field(..., description="Data Structures and Algorithms score (1-10)", examples=[7, 8, 9])
    aptitude_score: float = Field(..., description="Aptitude score (1-100)", examples=[65.0, 85.5])
    communication_skills: float = Field(..., description="Communication skills rating (1-10)", examples=[4.5, 7.5, 9.0])
    ml_knowledge: float = Field(..., description="Machine Learning knowledge rating (1-10)", examples=[6, 7, 8])
    system_design: float = Field(..., description="System design skills rating (1-10)", examples=[5, 6, 7])
    internships: int = Field(..., description="Number of internships completed", examples=[0, 1, 2])
    projects_count: int = Field(..., description="Number of projects completed", examples=[1, 2, 3])
    certifications: int = Field(..., description="Number of certifications obtained", examples=[0, 1, 2])
    hackathons: int = Field(..., description="Number of hackathons participated in", examples=[0, 1, 2])
    open_source_contributions: int = Field(..., description="Number of open source contributions", examples=[0, 1, 2])
    extracurriculars: int = Field(..., description="Number of extracurricular activities", examples=[0, 1, 2])