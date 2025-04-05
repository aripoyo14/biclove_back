from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, Time
)
from sqlalchemy.orm import relationship, declarative_base
import datetime
import pytz
from pydantic import BaseModel, ConfigDict
from typing import List

JST = pytz.timezone("Asia/Tokyo")
Base = declarative_base()

# -------------------------------
# Usersテーブル
# -------------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    affiliation = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    total_thanks = Column(Integer, default=0)
    total_view = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(JST))

    meetings = relationship("Meeting", back_populates="user")
    knowledges = relationship("Knowledge", back_populates="user")
    challenges = relationship("Challenge", back_populates="user")


# -------------------------------
# Meetingテーブル
# -------------------------------
class Meeting(Base):
    __tablename__ = "meeting"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(JST))

    user = relationship("User", back_populates="meetings")
    knowledges = relationship("Knowledge", back_populates="meeting")
    challenges = relationship("Challenge", back_populates="meeting")


# -------------------------------
# Knowledgeテーブル
# -------------------------------
class Knowledge(Base):
    __tablename__ = "knowledge"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    meeting_id = Column(Integer, ForeignKey("meeting.id"), nullable=False)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    thanks_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(JST))

    user = relationship("User", back_populates="knowledges")
    meeting = relationship("Meeting", back_populates="knowledges")
    views = relationship("View", back_populates="knowledge")
    thanks = relationship("Thanks", back_populates="knowledge")
    references_as_original = relationship("Reference", foreign_keys="[Reference.reference_knowledge_id]", back_populates="reference_knowledge")
    references_as_solution = relationship("Reference", foreign_keys="[Reference.solution_knowledge_id]", back_populates="solution_knowledge")


# -------------------------------
# Challengeテーブル
# -------------------------------
class Challenge(Base):
    __tablename__ = "challenge"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    meeting_id = Column(Integer, ForeignKey("meeting.id"), nullable=False)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(JST))

    user = relationship("User", back_populates="challenges")
    meeting = relationship("Meeting", back_populates="challenges")
    solutions = relationship("SolutionKnowledge", back_populates="challenge")


# -------------------------------
# SolutionKnowledgeテーブル
# -------------------------------
class SolutionKnowledge(Base):
    __tablename__ = "solution_knowledge"
    id = Column(Integer, primary_key=True, index=True)
    challenge_id = Column(Integer, ForeignKey("challenge.id"), nullable=False)
    solution_knowledge = Column(Text, nullable=False)

    challenge = relationship("Challenge", back_populates="solutions")
    references = relationship("Reference", back_populates="solution_knowledge")


# -------------------------------
# Referenceテーブル
# -------------------------------
class Reference(Base):
    __tablename__ = "reference"
    id = Column(Integer, primary_key=True, index=True)
    solution_knowledge_id = Column(Integer, ForeignKey("solution_knowledge.id"), nullable=False)
    reference_knowledge_id = Column(Integer, ForeignKey("knowledge.id"), nullable=False)
    solution_knowledge = relationship("SolutionKnowledge", back_populates="references")
    reference_knowledge = relationship("Knowledge", back_populates="references_as_original")


# -------------------------------
# Thanksテーブル
# -------------------------------
class Thanks(Base):
    __tablename__ = "thanks"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    knowledge_id = Column(Integer, ForeignKey("knowledge.id"), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(JST))

    knowledge = relationship("Knowledge", back_populates="thanks")


# -------------------------------
# Viewテーブル
# -------------------------------
class View(Base):
    __tablename__ = "view"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    knowledge_id = Column(Integer, ForeignKey("knowledge.id"), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(JST))

    knowledge = relationship("Knowledge", back_populates="views")
    
