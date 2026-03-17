from sqlalchemy import select, desc, func, create_engine
from sqlalchemy.orm import Session, DeclarativeBase, Column, Integer, Text, TIMESTAMP
from .config import SUBMIT_DB_URL, SCHEMA_NAME
from .main import ProcessRequest
from typing import List
from datetime import datetime

engine = create_engine(
    SUBMIT_DB_URL, 
    pool_pre_ping=True,  # Checks connection health before using it
    pool_recycle=3600    # Refreshes connections every hour
)

# TODO: copied from Kuba's script -> need to be updated with it
class Base(DeclarativeBase):
    pass

class Submit(Base):
    __tablename__ = "submit"
    __table_args__ = {"schema": SCHEMA_NAME}

    id         = Column(Integer, primary_key=True, autoincrement=True)
    email      = Column(Text)
    pipe_id    = Column(Text)
    transcript = Column(Text)
    created_at = Column(TIMESTAMP, default=datetime.now)


def get_submissions(request: ProcessRequest) -> List[Submit]:
    with Session(engine) as session:
        # Base query is using DISTINCT ON for unique emails
        # This ensures we get exactly ONE (the latest) record per student email
        stmt = (
            select(Submit)
            .distinct(Submit.email)
            .where(Submit.pipe_id == request.pipe_id)
            .order_by(Submit.email, desc(Submit.created_at))
        )

        if request.submit_ids:
            stmt = stmt.where(Submit.id.in_(request.submit_ids))
        if request.student_emails:
            stmt = stmt.where(Submit.email.in_(request.student_emails))

        subquery = stmt.subquery()
        final_stmt = select(Submit).from_statement(
            select(Submit).from_objs(subquery).order_by(func.random())
        )

        if request.head_n_results is not None:
            final_stmt = final_stmt.limit(request.head_n_results)

        results = session.execute(final_stmt).scalars().all()
        return list(results)
