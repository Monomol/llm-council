from sqlalchemy import select, desc, func, create_engine, Column, Integer, Text, TIMESTAMP
from sqlalchemy.orm import Session, DeclarativeBase, aliased
from .config import SUBMIT_DB_URL, SCHEMA_NAME
from .objects import ProcessPayload
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

    def __repr__(self):
        return f"Submit(id='{self.id}', email='{self.email}', pipe_id='{self.pipe_id}', transcript='...', created_at='{self.created_at}')"

def get_submissions(payload: ProcessPayload) -> List[Submit]:
    with Session(engine) as session:
        inner_stmt = (
            select(Submit)
            .where(Submit.pipe_id == payload.pipe_id)
        )

        if payload.student_emails:
            inner_stmt = inner_stmt.where(Submit.email.in_(payload.student_emails))
        
        inner_stmt = (
            inner_stmt
            .distinct(Submit.email)
            .order_by(Submit.email, desc(Submit.created_at))
        )

        subquery = inner_stmt.subquery()
        sub_aliased = aliased(Submit, subquery)
        stmt = select(sub_aliased)

        if payload.submit_ids:
            stmt = stmt.where(sub_aliased.id.in_(payload.submit_ids))

        if payload.random_sample:
            stmt = stmt.order_by(func.random())
        else:
            stmt = stmt.order_by(desc(subquery.c.created_at))

        if payload.head_n_results is not None:
            stmt = stmt.limit(payload.head_n_results)

        results = session.execute(stmt).scalars().all()
        return list(results)
